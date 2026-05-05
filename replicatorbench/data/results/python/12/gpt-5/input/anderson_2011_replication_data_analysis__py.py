#!/usr/bin/env python3
"""
Python replication entry for Anderson (2011) low-caste village dominance and agricultural income per acre.

Models:
 1) Full sample: raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode)*C(caste), cluster SE at vill_id
 2) Exploratory: net_inc_per_acre with the same covariates, cluster SE at vill_id
 3) Subset (UP/Bihar: stcode in {2, 15}): raw_inc_per_acre with same covariates, cluster SE at vill_id

Inputs:
 - analysis_data.dta (searched in /app/data/, then ./replication_data/)

Outputs:
 - /app/data/anderson_2011_results.csv (key coefficients and stats)
 - /app/data/anderson_2011_results.txt (model summaries)
"""

import os
import sys
import textwrap
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import pyreadstat
except Exception as e:
    pyreadstat = None


def find_data_path() -> Optional[str]:
    candidates = [
        "/app/data/analysis_data.dta",
        os.path.join(os.path.dirname(__file__), "replication_data", "analysis_data.dta"),
        os.path.join(os.getcwd(), "replication_data", "analysis_data.dta"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_data() -> pd.DataFrame:
    path = find_data_path()
    if path is None:
        raise FileNotFoundError(
            "analysis_data.dta not found in /app/data or ./replication_data."
        )
    if pyreadstat is None:
        raise ImportError("pyreadstat is required to read .dta files but is not installed.")
    df, meta = pyreadstat.read_dta(path)
    # Ensure standard column names are present
    required = [
        "raw_inc_per_acre", "net_inc_per_acre", "literate_hh", "land_owned",
        "locaste_land_v", "stcode", "caste", "vill_id"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df


def run_model(df: pd.DataFrame, outcome: str, model_name: str, subset: Optional[pd.Series] = None):
    dfx = df.copy()
    if subset is not None:
        dfx = dfx.loc[subset].copy()
    # Drop NA rows for selected variables
    vars_needed = [outcome, "literate_hh", "land_owned", "locaste_land_v", "stcode", "caste", "vill_id"]
    dfx = dfx.dropna(subset=vars_needed)
    # Ensure vill_id is not float with NaNs
    if dfx.shape[0] == 0:
        raise ValueError(f"No observations after filtering for model {model_name}.")

    formula = f"{outcome} ~ literate_hh + land_owned + locaste_land_v + C(stcode)*C(caste)"
    groups = dfx["vill_id"].astype("int64", errors="ignore")
    model = smf.ols(formula=formula, data=dfx)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    # Extract key stats for locaste_land_v
    term = "locaste_land_v"
    if term not in res.params.index:
        # In case of perfect collinearity (shouldn't happen), record NaNs
        coef = np.nan
        se = np.nan
        pval = np.nan
        tval = np.nan
        ci_low, ci_high = np.nan, np.nan
    else:
        coef = res.params[term]
        se = res.bse[term]
        pval = res.pvalues[term]
        tval = res.tvalues[term]
        ci_low, ci_high = res.conf_int().loc[term].tolist()

    summary_row = {
        "model": model_name,
        "outcome": outcome,
        "nobs": int(res.nobs),
        "coef_locaste_land_v": coef,
        "se_locaste_land_v": se,
        "pval_locaste_land_v": pval,
        "t_locaste_land_v": tval,
        "ci_low_locaste_land_v": ci_low,
        "ci_high_locaste_land_v": ci_high,
        "r2": res.rsquared,
        "cov_type": res.cov_type,
    }

    return res, summary_row


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    print("Loading data...")
    df = load_data()

    # Optional: confirm low-caste sample attribute if present
    if "locaste" in df.columns:
        unique_lc = pd.Series(df["locaste"].unique()).dropna().tolist()
        print(f"locaste unique values: {unique_lc}")

    results: List[Tuple] = []

    # Model 1: Full sample, raw_inc_per_acre
    print("Fitting Model 1: Full sample, outcome=raw_inc_per_acre")
    res1, row1 = run_model(df, outcome="raw_inc_per_acre", model_name="full_raw")
    results.append(("Model 1 (full_raw)", res1, row1))

    # Model 2: Exploratory, net_inc_per_acre
    print("Fitting Model 2: Full sample, outcome=net_inc_per_acre")
    res2, row2 = run_model(df, outcome="net_inc_per_acre", model_name="full_net")
    results.append(("Model 2 (full_net)", res2, row2))

    # Model 3: Subset (UP/Bihar stcode in {2,15}), raw_inc_per_acre
    print("Fitting Model 3: UP/Bihar subset (stcode in {2,15}), outcome=raw_inc_per_acre")
    subset_up_bihar = df["stcode"].isin([2, 15])
    res3, row3 = run_model(df, outcome="raw_inc_per_acre", model_name="subset_up_bihar_raw", subset=subset_up_bihar)
    results.append(("Model 3 (subset_up_bihar_raw)", res3, row3))

    # Save CSV with key coefficients
    out_csv = "/app/data/anderson_2011_results.csv"
    ensure_dir(out_csv)
    summary_rows = [r for _, _, r in results]
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    # Save text summaries
    out_txt = "/app/data/anderson_2011_results.txt"
    lines = []
    header = textwrap.dedent(
        """
        Replication: Anderson (2011) - Low-caste village dominance and agricultural income per acre
        Models estimated with OLS, cluster-robust SE at village level (vill_id)
        Specification: outcome ~ literate_hh + land_owned + locaste_land_v + C(stcode)*C(caste)
        Key parameter of interest: coefficient on locaste_land_v
        """
    ).strip()
    lines.append(header)

    for label, res, row in results:
        lines.append("\n" + "=" * 80)
        lines.append(label)
        lines.append("-" * 80)
        lines.append(res.summary().as_text())
        lines.append("\nKey coefficient (locaste_land_v):")
        lines.append(
            f" coef={row['coef_locaste_land_v']:.6g}, se={row['se_locaste_land_v']:.6g}, p={row['pval_locaste_land_v']:.6g}, "
            f"t={row['t_locaste_land_v']:.6g}, CI=[{row['ci_low_locaste_land_v']:.6g}, {row['ci_high_locaste_land_v']:.6g}]"
        )

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved results to: {out_csv} and {out_txt}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
