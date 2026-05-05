#!/usr/bin/env python3
"""
Python translation of key analysis from the original Stata .do file (data cleaning and analysis.do).

Scope:
- Replicate the focal nonparametric comparison of first-stage effort between OS (one-stage) and TS (two-stage) treatments using the provided cleaned dataset FinalData.csv.
- Preserve logic of: `ranksum effort_t1, by(treatment)` where treatment is coded 1=OS, 2=TS.

Notes:
- All IO is performed in /app/data per execution environment requirements.
- This script assumes a cleaned dataset `FinalData.csv` is present in /app/data with at least the columns: `treat` and `effort_t1`.
- The original .do file also contained raw-data cleaning instructions for z-Tree exports. For this replication, we operate directly on the already-cleaned FinalData.csv provided in replication_data.

Outputs written to /app/data:
- effort_t1_by_treatment_summary.csv: Descriptive statistics of effort_t1 by treatment
- mannwhitney_effort_t1_OS_vs_TS.json: Test results from Mann-Whitney U test (two-sided) and simple effect-size metrics
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DATA_PATH = "/app/data/FinalData.csv"
OUT_SUMMARY_CSV = "/app/data/effort_t1_by_treatment_summary.csv"
OUT_TEST_JSON = "/app/data/mannwhitney_effort_t1_OS_vs_TS.json"


@dataclass
class GroupStats:
    n: int
    mean: float
    median: float
    std: float
    min: float
    q1: float
    q3: float
    max: float


def compute_group_stats(x: pd.Series) -> GroupStats:
    x_clean = x.dropna().astype(float)
    return GroupStats(
        n=int(x_clean.shape[0]),
        mean=float(x_clean.mean()) if x_clean.shape[0] else float("nan"),
        median=float(x_clean.median()) if x_clean.shape[0] else float("nan"),
        std=float(x_clean.std(ddof=1)) if x_clean.shape[0] > 1 else float("nan"),
        min=float(x_clean.min()) if x_clean.shape[0] else float("nan"),
        q1=float(x_clean.quantile(0.25)) if x_clean.shape[0] else float("nan"),
        q3=float(x_clean.quantile(0.75)) if x_clean.shape[0] else float("nan"),
        max=float(x_clean.max()) if x_clean.shape[0] else float("nan"),
    )


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U statistic.
    r_rb = 2U/(n1*n2) - 1
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return 2.0 * u_stat / (n1 * n2) - 1.0


def main() -> int:
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}. Ensure FinalData.csv is placed under /app/data.")
        return 1

    df = pd.read_csv(DATA_PATH)

    required_cols = {"treat", "effort_t1"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"ERROR: Missing required columns in dataset: {sorted(missing_cols)}")
        return 1

    # Validate treatment coding: 1=OS, 2=TS (as per DataDictionary.csv)
    valid_treats = {1, 2}
    treat_values = set(pd.unique(df["treat"].dropna().astype(int)))
    unexpected = treat_values - valid_treats
    if unexpected:
        print(f"WARNING: Unexpected treatment codes found: {sorted(unexpected)}. Proceeding with 1 and 2 only.")

    # Subset for analysis and drop missing
    work = df.loc[df["treat"].isin([1, 2]), ["treat", "effort_t1"]].dropna()

    # Split groups
    os_mask = work["treat"] == 1
    ts_mask = work["treat"] == 2
    os_vals = work.loc[os_mask, "effort_t1"].astype(float)
    ts_vals = work.loc[ts_mask, "effort_t1"].astype(float)

    # Compute descriptive stats
    stats_os = compute_group_stats(os_vals)
    stats_ts = compute_group_stats(ts_vals)

    summary_df = pd.DataFrame(
        [
            {"treatment": "OS (1)", **asdict(stats_os)},
            {"treatment": "TS (2)", **asdict(stats_ts)},
        ]
    )
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    # Mann-Whitney U test (two-sided), OS vs TS
    # By default, scipy.stats.mannwhitneyu uses a one-sided alternative; specify two-sided
    mw_res = mannwhitneyu(os_vals, ts_vals, alternative="two-sided")
    u_stat = float(mw_res.statistic)
    p_value = float(mw_res.pvalue)

    # Effect sizes
    n_os = int(stats_os.n)
    n_ts = int(stats_ts.n)
    rb = rank_biserial_from_u(u_stat, n_os, n_ts)

    results: Dict[str, object] = {
        "test": "Mann-Whitney U (two-sided)",
        "outcome": "effort_t1",
        "grouping": {"variable": "treat", "OS": 1, "TS": 2},
        "n_os": n_os,
        "n_ts": n_ts,
        "mean_os": stats_os.mean,
        "mean_ts": stats_ts.mean,
        "median_os": stats_os.median,
        "median_ts": stats_ts.median,
        "u_stat": u_stat,
        "p_value": p_value,
        "rank_biserial_correlation": rb,
        "expected_direction": "TS > OS",
        "alpha": 0.05,
        "reject_null_at_alpha": bool(p_value < 0.05),
    }

    with open(OUT_TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also print a concise summary to stdout
    print("Analysis complete. Key results:")
    print(summary_df.to_string(index=False))
    print("\nMann-Whitney U test (two-sided):")
    print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
