import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "/app/data/1-s2.0-S1090513816301118-mmc1.csv"
OUTPUT_PATH = "/app/data/results_task2.json"


def safe_group(series: pd.Series, mask: pd.Series) -> pd.Series:
    return pd.to_numeric(series[mask], errors="coerce").dropna()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if "MiniK_Total" not in df.columns or "DSM5_Total" not in df.columns:
        raise KeyError("Required columns 'MiniK_Total' and/or 'DSM5_Total' not found.")

    # Build groups: fast (<= -1) vs slow (>= 1); exclude -1 < score < 1
    mini = pd.to_numeric(df["MiniK_Total"], errors="coerce")
    dsm = pd.to_numeric(df["DSM5_Total"], errors="coerce")

    fast_mask = mini <= -1
    slow_mask = mini >= 1

    fast = safe_group(dsm, fast_mask)
    slow = safe_group(dsm, slow_mask)

    results: Dict[str, Any] = {
        "task": "Task2",
        "analysis": "Mann-Whitney U test on DSM5_Total by MiniK groups (fast<=-1 vs slow>=1); exclusions in (-1,1)",
        "dataset": os.path.basename(DATA_PATH),
        "group_definition": {
            "fast": "MiniK_Total <= -1",
            "slow": "MiniK_Total >= 1",
            "excluded_range": "-1 < MiniK_Total < 1"
        },
        "group_ns": {"fast": int(fast.shape[0]), "slow": int(slow.shape[0])},
        "group_medians": {"fast": float(np.median(fast)) if fast.shape[0] else np.nan,
                           "slow": float(np.median(slow)) if slow.shape[0] else np.nan}
    }

    # Assumption checks
    # Levene's test for equal variances (center=median to be robust)
    if fast.shape[0] > 0 and slow.shape[0] > 0:
        lev_stat, lev_p = stats.levene(fast, slow, center='median')
        results["levene"] = {"stat": float(lev_stat), "p": float(lev_p)}
    else:
        results["levene"] = {"stat": np.nan, "p": np.nan}

    # Shapiro-Wilk normality tests by group
    def shapiro_safe(x: pd.Series) -> Dict[str, Any]:
        if x.shape[0] < 3:
            return {"W": np.nan, "p": np.nan, "n": int(x.shape[0])}
        try:
            W, p = stats.shapiro(x)
            return {"W": float(W), "p": float(p), "n": int(x.shape[0])}
        except Exception:
            return {"W": np.nan, "p": np.nan, "n": int(x.shape[0])}

    results["shapiro"] = {"fast": shapiro_safe(fast), "slow": shapiro_safe(slow)}

    # Mann-Whitney U test (two-sided)
    if fast.shape[0] > 0 and slow.shape[0] > 0:
        try:
            U, p = stats.mannwhitneyu(fast, slow, alternative='two-sided')
            results["mannwhitney"] = {"U": float(U), "p": float(p)}
        except ValueError as e:
            results["mannwhitney"] = {"U": np.nan, "p": np.nan, "error": str(e)}
    else:
        results["mannwhitney"] = {"U": np.nan, "p": np.nan, "error": "Insufficient data in one or both groups."}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
