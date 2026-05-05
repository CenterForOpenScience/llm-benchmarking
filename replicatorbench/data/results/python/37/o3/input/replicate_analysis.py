#!/usr/bin/env python3
"""Replication analysis script for attention-intimacy study.
Reads dataset, constructs intimacy scale (atot) following original SPSS syntax, then
runs independent samples t-test comparing attention vs. no-attention conditions.
Outputs key statistics to stdout and writes a simple results text file in /app/data.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def reverse_code(series: pd.Series) -> pd.Series:
    """Reverse-codes 7-point Likert scale: 7->1, 6->2, ..., 1->7"""
    mapping = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    return series.map(mapping)


def cohend(x, y):
    """Compute Cohen's d for independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof)
    return (x.mean() - y.mean()) / pooled_std

# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def main():
    # Path to dataset inside container
    data_path = os.getenv("DATA_FILE", "/app/data/ohtsubo data.csv")

    if not os.path.exists(data_path):
        sys.stderr.write(f"Dataset not found at {data_path}. Set DATA_FILE environment variable if needed.\n")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Reverse-code a4 into a4r if not already present
    if "a4r" not in df.columns:
        if "a4" not in df.columns:
            raise ValueError("Column 'a4' not found in dataset, cannot create 'a4r'.")
        df["a4r"] = reverse_code(df["a4"])

    # Compute atot (sum of a1..a6 with a4 reversed)
    required_cols = ["a1", "a2", "a3", "a4r", "a5", "a6"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scale computation: {missing}")

    df["atot"] = df[required_cols].sum(axis=1)

    # Drop rows with missing values in key variables
    analysis_df = df.dropna(subset=["condition", "atot"])

    # Split by condition
    group1 = analysis_df[analysis_df["condition"] == 1]["atot"]
    group0 = analysis_df[analysis_df["condition"] == 0]["atot"]

    # Independent samples t-test (Welch's)
    t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)

    # Cohen's d
    d_val = cohend(group1, group0)

    # Prepare output
    summary = (
        f"Replication of attention-intimacy effect\n"
        f"Dataset: {os.path.basename(data_path)}\n"
        f"Sample sizes -> Attention (1): {len(group1)}, No Attention (0): {len(group0)}\n"
        f"Means (SD) -> Attention: {group1.mean():.2f} ({group1.std(ddof=1):.2f}), "
        f"No Attention: {group0.mean():.2f} ({group0.std(ddof=1):.2f})\n"
        f"Welch t-test: t = {t_stat:.3f}, p = {p_val:.3g}\n"
        f"Cohen's d = {d_val:.2f}\n"
    )

    print(summary)

    # Save to results file
    results_path = "/app/data/replication_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Results written to {results_path}")

if __name__ == "__main__":
    main()
