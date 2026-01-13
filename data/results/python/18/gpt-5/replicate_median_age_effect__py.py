#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    # Paths
    # Attempt to resolve dataset path robustly
    candidate_paths = [
        "/app/data/original/18/0112_gpt5_2/replication_data/replication_data_mkk9.csv",
        "/app/data/original/18/0112_gpt5/replication_data/replication_data_mkk9.csv"
    ]
    data_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        # Search recursively under /app/data
        search_root = "/app/data"
        target_name = "replication_data_mkk9.csv"
        for root, dirs, files in os.walk(search_root):
            if target_name in files:
                data_path = os.path.join(root, target_name)
                break
    if data_path is None:
        raise FileNotFoundError("Could not locate replication_data_mkk9.csv under /app/data. Please ensure it is mounted.")
    out_dir = "/app/data"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Basic sanity checks and cleaning analogous to Stata script
    required_cols = ["entrepreneurship", "median_age", "year", "country", "cy_cell"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing values in key columns (Stata: drop if median_age == 'NA'; here: drop NaNs)
    before = len(df)
    df_clean = df.copy()

    # Ensure correct dtypes
    # median_age should be numeric
    df_clean["median_age"] = pd.to_numeric(df_clean["median_age"], errors="coerce")
    df_clean["entrepreneurship"] = pd.to_numeric(df_clean["entrepreneurship"], errors="coerce")
    df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
    # weights and country
    df_clean["cy_cell"] = pd.to_numeric(df_clean["cy_cell"], errors="coerce")
    df_clean["country"] = df_clean["country"].astype(str)

    df_clean = df_clean.dropna(subset=["median_age", "entrepreneurship", "year", "country", "cy_cell"]).copy()

    # Ensure weights are non-negative; drop non-positive weights if any
    if (df_clean["cy_cell"] <= 0).any():
        dropped = (df_clean["cy_cell"] <= 0).sum()
        print(f"Warning: Dropping {dropped} rows with non-positive weights (cy_cell <= 0)")
        df_clean = df_clean[df_clean["cy_cell"] > 0].copy()

    after = len(df_clean)
    print(f"Cleaned data: kept {after} of {before} rows (dropped {before - after}).")

    # Model: WLS with year fixed effects and country-clustered robust SEs
    formula = "entrepreneurship ~ median_age + C(year)"
    print(f"Fitting WLS model: {formula} with analytic weights and country-clustered SEs")

    model = smf.wls(formula=formula, data=df_clean, weights=df_clean["cy_cell"])
    # Cluster-robust covariance by country
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df_clean["country"]})

    # Save summary
    summary_txt_path = os.path.join(out_dir, "median_age_regression_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(res.summary().as_text())
    print(f"Saved model summary to: {summary_txt_path}")

    # Save tidy results
    ci = res.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]
    results_df = pd.DataFrame({
        "term": res.params.index,
        "estimate": res.params.values,
        "std_error": res.bse.values,
        "t_value": res.tvalues.values,
        "p_value": res.pvalues.values,
    }).set_index("term").join(ci)

    results_csv_path = os.path.join(out_dir, "median_age_regression_results.csv")
    results_df.to_csv(results_csv_path)
    print(f"Saved tidy regression results to: {results_csv_path}")

    # Additional diagnostics and context metrics
    n_obs = int(res.nobs)
    n_countries = df_clean["country"].nunique()
    n_years = df_clean["year"].nunique()
    sd_median_age = float(df_clean["median_age"].std(ddof=1))
    beta_med_age = float(res.params.get("median_age", np.nan))
    # Predicted change for a one SD decrease in median_age
    effect_for_minus1sd = beta_med_age * (-sd_median_age)

    extras = {
        "n_obs": n_obs,
        "n_countries": int(n_countries),
        "n_years": int(n_years),
        "sd_median_age": sd_median_age,
        "beta_median_age": beta_med_age,
        "predicted_change_for_minus1SD_median_age": effect_for_minus1sd,
        "weighting": "analytic weights (approximated via WLS weights=cy_cell)",
        "cov_type": "cluster",
        "cluster_group": "country"
    }

    extras_json_path = os.path.join(out_dir, "median_age_regression_extras.json")
    with open(extras_json_path, "w") as f:
        json.dump(extras, f, indent=2)
    print(f"Saved extras to: {extras_json_path}")

    # Also print focal coefficient line for quick inspection
    if "median_age" in results_df.index:
        med_row = results_df.loc["median_age"]
        print("Focal coefficient (median_age):")
        print(med_row.to_dict())
    else:
        print("Warning: 'median_age' term not found in results.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
