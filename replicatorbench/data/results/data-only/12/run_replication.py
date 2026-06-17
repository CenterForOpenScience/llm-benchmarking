import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    # Paths
    data_path = os.environ.get("APP_DATA", "/app/data")
    input_file = os.path.join(data_path, "analysis_data.dta")
    summary_out = os.path.join(data_path, "regression_summary.txt")
    csv_out = os.path.join(data_path, "regression_table.csv")
    json_out = os.path.join(data_path, "results_lcv_income.json")

    # Load data
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Expected data file not found at {input_file}. Place 'analysis_data.dta' in /app/data")

    df = pd.read_stata(input_file)

    # Expected columns
    needed_cols = [
        "net_inc_per_acre",  # outcome
        "locaste_land_v",    # low-caste dominated village indicator (0/1)
        "literate_hh",       # control
        "land_owned",        # control
        "stcode",            # state fixed effects
        "vill_id"            # cluster id (village)
    ]
    for col in needed_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataset. Columns available: {list(df.columns)}")

    # If a 'locaste' indicator exists (1 for lower-caste household), keep sample as provided
    # Dataset appears already restricted to lower-caste households (locaste==1 for all rows observed)

    # Drop missing in variables used
    model_cols = ["net_inc_per_acre", "locaste_land_v", "literate_hh", "land_owned", "stcode", "vill_id"]
    d = df[model_cols].dropna()

    # Ensure correct dtypes
    d["stcode"] = d["stcode"].astype("category")

    # Specify model: OLS with state FE and controls
    formula = "net_inc_per_acre ~ locaste_land_v + literate_hh + land_owned + C(stcode)"

    # Fit OLS with heteroskedasticity-robust SE (HC1)
    ols_model = smf.ols(formula, data=d).fit(cov_type="HC1")

    # Also compute village-clustered SEs for sensitivity
    try:
        clusters = d["vill_id"].astype(int)
        ols_cluster = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    except Exception as e:
        warnings.warn(f"Cluster-robust estimation failed: {e}")
        ols_cluster = None

    # Extract focal coefficient (effect of residing in low-caste dominated village)
    coef_name = "locaste_land_v"
    results = {
        "n_obs": int(ols_model.nobs),
        "outcome": "net_inc_per_acre",
        "focal_variable": coef_name,
        "controls": ["literate_hh", "land_owned", "C(stcode)"],
        "estimator": "OLS (HC1 robust SE)",
        "coefficient": float(ols_model.params.get(coef_name, np.nan)),
        "std_error": float(ols_model.bse.get(coef_name, np.nan)),
        "t_value": float(ols_model.tvalues.get(coef_name, np.nan)),
        "p_value": float(ols_model.pvalues.get(coef_name, np.nan)),
        "conf_int": list(map(float, ols_model.conf_int().loc[coef_name].values)) if coef_name in ols_model.params.index else [np.nan, np.nan]
    }

    # If clustered results available, add them
    if ols_cluster is not None and coef_name in ols_cluster.params.index:
        results.update({
            "estimator_cluster": "OLS (village-clustered SE)",
            "coefficient_cluster": float(ols_cluster.params.get(coef_name, np.nan)),
            "std_error_cluster": float(ols_cluster.bse.get(coef_name, np.nan)),
            "t_value_cluster": float(ols_cluster.tvalues.get(coef_name, np.nan)),
            "p_value_cluster": float(ols_cluster.pvalues.get(coef_name, np.nan)),
            "conf_int_cluster": list(map(float, ols_cluster.conf_int().loc[coef_name].values))
        })

    # Save textual summary
    with open(summary_out, "w") as f:
        f.write("=== OLS with HC1 robust SEs ===\n")
        f.write(ols_model.summary().as_text())
        f.write("\n\n")
        if ols_cluster is not None:
            f.write("=== OLS with village-clustered SEs ===\n")
            f.write(ols_cluster.summary().as_text())

    # Save coefficient table to CSV
    table_rows = []
    for name in ols_model.params.index:
        row = {
            "term": name,
            "coef": ols_model.params[name],
            "se_hc1": ols_model.bse[name],
            "p_hc1": ols_model.pvalues[name]
        }
        if ols_cluster is not None and name in ols_cluster.params.index:
            row.update({
                "se_cluster": ols_cluster.bse[name],
                "p_cluster": ols_cluster.pvalues[name]
            })
        table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(csv_out, index=False)

    # Save JSON results
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # Print a concise message for logs
    print(json.dumps({
        "message": "Replication OLS completed",
        "results_file": json_out,
        "summary_file": summary_out,
        "csv_file": csv_out,
        "coef_locaste_land_v": results.get("coefficient"),
        "se_locaste_land_v": results.get("std_error"),
        "n_obs": results.get("n_obs")
    }))


if __name__ == "__main__":
    main()
