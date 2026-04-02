#!/usr/bin/env python3
"""
Replication script for: So Hard to Say Goodbye? Turnover Intention among U.S. Federal Employees
Focal claim: Higher overall job satisfaction is associated with lower intention to leave one's agency.

This script:
- Loads the estimation dataset from /app/data
- Fits a logistic regression for LeavingAgency on JobSat and controls
- Uses cluster-robust (by Agency) standard errors
- Saves coefficient tables and a text summary to /app/data

Assumptions:
- The dataset file is available at /app/data/Estimation Data - Pitts (126zz).csv
- Column names follow those seen in the provided extract

Outputs:
- /app/data/turnover_logit_summary.txt: model summary
- /app/data/turnover_replication_results.json: structured results for comparison
- /app/data/turnover_replication_coefs.csv: coefficient table with robust SEs and p-values
"""

import json
import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    data_path = os.path.join("/app/data", "Estimation Data - Pitts (126zz).csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Required dataset not found at {data_path}. Ensure you mount ./data to /app/data and include the CSV."
        )

    df = pd.read_csv(data_path)

    # Define variables consistent with available columns
    outcome = "LeavingAgency"
    controls = [
        "Over40",
        "NonMinority",
        "JobSat",
        "SatPay",
        "SatAdvan",
        "PerfCul",
        "Empowerment",
        "RelSup",
        "Relcow",
        "Over40xSatAdvan",
    ]
    cluster_var = "Agency"

    # Subset to required columns and drop missing (listwise deletion)
    needed_cols = [outcome, cluster_var] + controls
    df_model = df[needed_cols].dropna().copy()

    # Ensure binary outcome is 0/1
    if not set(df_model[outcome].unique()).issubset({0, 1}):
        raise ValueError(f"Outcome {outcome} must be binary 0/1.")

    # Design matrices
    X = sm.add_constant(df_model[controls], has_constant='add')
    y = df_model[outcome].astype(int)

    # Fit Logit with cluster-robust SEs by Agency
    groups = df_model[cluster_var]
    model = sm.Logit(y, X)
    res = model.fit(disp=0, maxiter=200, cov_type='cluster', cov_kwds={'groups': groups})

    # Build outputs
    coef = res.params
    bse = res.bse
    pvalues = res.pvalues

    coef_table = pd.DataFrame({
        'coef': coef,
        'std_err_cluster': bse,
        'z': coef / bse,
        'p_value_cluster': pvalues,
    })
    coef_table.index.name = 'variable'

    # Save text summary
    summary_text = res.summary().as_text()
    with open(os.path.join("/app/data", "turnover_logit_summary.txt"), "w") as f:
        f.write(summary_text)

    # Focal coefficient info
    focal_var = "JobSat"
    focal_info: Dict[str, float] = {
        "coef": float(coef.get(focal_var, np.nan)),
        "std_err_cluster": float(bse.get(focal_var, np.nan)),
        "p_value_cluster": float(pvalues.get(focal_var, np.nan)),
        "direction": "negative" if coef.get(focal_var, np.nan) < 0 else "positive",
    }

    # Predicted probability at means (for reference)
    xbar = X.mean(axis=0)
    linpred = float(np.dot(xbar.values if hasattr(xbar, 'values') else xbar, coef.loc[X.columns]))
    prob_at_means = 1.0 / (1.0 + np.exp(-linpred))

    results_obj = {
        "model": "Logit: LeavingAgency ~ const + Over40 + NonMinority + JobSat + SatPay + SatAdvan + PerfCul + Empowerment + RelSup + Relcow + Over40xSatAdvan",
        "n_obs": int(df_model.shape[0]),
        "cluster": cluster_var,
        "cov_type": "cluster",
        "focal_variable": focal_var,
        "focal_result": focal_info,
        "prob_at_means": prob_at_means,
        "notes": "Listwise deletion across analysis columns; cluster-robust SEs by Agency; dataset: Estimation Data - Pitts (126zz).csv"
    }

    # Save JSON results
    with open(os.path.join("/app/data", "turnover_replication_results.json"), "w") as f:
        json.dump(results_obj, f, indent=2)

    # Save full coefficient table
    coef_table.to_csv(os.path.join("/app/data", "turnover_replication_coefs.csv"))

    # Print focal summary to stdout for convenience
    print(json.dumps(results_obj, indent=2))


if __name__ == "__main__":
    main()
