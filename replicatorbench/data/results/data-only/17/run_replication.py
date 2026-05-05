import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "/app/data/data.csv"
OUTPUT_JSON = "/app/data/replication_results.json"
OUTPUT_CSV = "/app/data/replication_results.csv"
SUMMARY_TXT = "/app/data/replication_summary.txt"


def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        sys.stderr.write(f"ERROR: Expected data at {DATA_PATH}. Place data.csv there and rerun.\n")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Subset to strong democracies (Polity IV >= 9)
    if "democ" not in df.columns:
        sys.stderr.write("ERROR: 'democ' column not found in data.\n")
        sys.exit(1)

    df_strong = df[df["democ"] >= 9].copy()

    # Define variables
    outcome = "gov_consumption"
    key_iv = "sd_gov"  # Polarization measure: SD of the 'private vs government ownership' question

    required_cols: List[str] = [
        outcome, key_iv,
        "log_gdp_per_capita",  # economic development
        "trade_share",          # openness
        "age_15_64", "age_65_plus",  # demographics
        "federal", "oecd",     # institutions
        # region and colonial controls
        "africa", "laam", "asiae",
        "col_uka", "col_espa", "col_otha",
        # optional identifiers
        "country", "year", "wave"
    ]

    missing_cols = [c for c in required_cols if c not in df_strong.columns]
    if missing_cols:
        sys.stderr.write(f"WARNING: Missing expected columns: {missing_cols}. Proceeding with available ones.\n")

    # Keep only available columns
    cols_present = [c for c in required_cols if c in df_strong.columns]
    work = df_strong[cols_present].copy()

    # Create wave dummies if wave exists
    wave_cols: List[str] = []
    if "wave" in work.columns:
        dummies = pd.get_dummies(work["wave"], prefix="wave", drop_first=True)
        wave_cols = list(dummies.columns)
        work = pd.concat([work.drop(columns=["wave"]), dummies], axis=1)

    # Drop rows with missing values in any modeling columns
    model_cols = [c for c in work.columns if c in [outcome, key_iv, "log_gdp_per_capita", "trade_share", "age_15_64", "age_65_plus", "federal", "oecd", "africa", "laam", "asiae", "col_uka", "col_espa", "col_otha"]] + wave_cols
    model_cols = [c for c in model_cols if c in work.columns]
    work_clean = work.dropna(subset=model_cols)

    # Build design matrices
    y = work_clean[outcome].astype(float)
    X_cols = []
    for c in [key_iv, "log_gdp_per_capita", "trade_share", "age_15_64", "age_65_plus", "federal", "oecd", "africa", "laam", "asiae", "col_uka", "col_espa", "col_otha"]:
        if c in work_clean.columns:
            X_cols.append(c)
    X_cols += [c for c in wave_cols if c in work_clean.columns]

    if key_iv not in X_cols:
        sys.stderr.write("ERROR: Key independent variable 'sd_gov' not available for modeling.\n")
        sys.exit(1)

    X = work_clean[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust SE (HC1)
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC1')

    # Extract focal coefficient
    coef = results.params.get(key_iv, np.nan)
    se = results.bse.get(key_iv, np.nan)
    pval = results.pvalues.get(key_iv, np.nan)

    # Prepare outputs
    out_json = {
        "model": "OLS with HC1 robust SEs",
        "dependent_variable": outcome,
        "key_independent_variable": key_iv,
        "controls": [c for c in X_cols if c != key_iv],
        "n_obs": int(results.nobs),
        "r_squared": float(results.rsquared),
        "coef_sd_gov": float(coef) if pd.notnull(coef) else None,
        "se_sd_gov": float(se) if pd.notnull(se) else None,
        "p_value_sd_gov": float(pval) if pd.notnull(pval) else None,
        "countries_in_sample": sorted(work_clean["country"].dropna().unique().tolist()) if "country" in work_clean.columns else None,
        "year_range": [int(work_clean["year"].min()), int(work_clean["year"].max())] if "year" in work_clean.columns else None,
    }

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out_json, f, indent=2)

    # Save CSV row
    pd.DataFrame([
        {
            "n_obs": out_json["n_obs"],
            "r_squared": out_json["r_squared"],
            "coef_sd_gov": out_json["coef_sd_gov"],
            "se_sd_gov": out_json["se_sd_gov"],
            "p_value_sd_gov": out_json["p_value_sd_gov"],
        }
    ]).to_csv(OUTPUT_CSV, index=False)

    # Save a human-readable summary
    summary_lines = []
    summary_lines.append("Replication: Effect of polarization (sd_gov) on government consumption in strong democracies")
    summary_lines.append(f"N = {out_json['n_obs']}")
    summary_lines.append(f"R-squared = {out_json['r_squared']:.4f}")
    summary_lines.append(f"coef(sd_gov) = {out_json['coef_sd_gov']:.4f} (SE = {out_json['se_sd_gov']:.4f}, p = {out_json['p_value_sd_gov']:.4g})")
    summary_lines.append("Controls: " + ", ".join(out_json["controls"]))

    with open(SUMMARY_TXT, "w") as f:
        f.write("\n".join(summary_lines) + "\n\n")
        f.write(str(results.summary()))

    print("Saved results to:")
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_CSV}")
    print(f"- {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
