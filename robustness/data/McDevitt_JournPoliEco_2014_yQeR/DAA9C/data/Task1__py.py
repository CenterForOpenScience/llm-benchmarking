import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# IO paths
DATA_FILE = Path("/app/data/final_data.dta")
OUT_JSON = Path("/app/data/task1_result.json")


def main():
    # Load data
    df = pd.read_stata(DATA_FILE)
    # Keep required columns
    df2 = df[["complaints_2008", "first_A"]].dropna(subset=["complaints_2008", "first_A"])  # per analyst, use overall complaints and first_A

    # Group descriptives
    g0 = df2.loc[df2["first_A"] == 0, "complaints_2008"]
    g1 = df2.loc[df2["first_A"] == 1, "complaints_2008"]

    ratio = float(np.nan)
    if g0.mean() and not np.isclose(g0.mean(), 0.0):
        ratio = float(g1.mean() / g0.mean())

    # Welch's t-test (two-sided) between groups
    ttest_res = stats.ttest_ind(g1, g0, equal_var=False, nan_policy="omit")

    result = {
        "task": "Task1",
        "analysis": "Group comparison of overall complaints by first_A (A/number name)",
        "n_total": int(len(df2)),
        "group_descriptives": {
            "first_A_0": {
                "n": int(g0.count()),
                "mean": float(g0.mean()),
                "std": float(g0.std(ddof=1)),
                "min": float(g0.min()),
                "max": float(g0.max()),
            },
            "first_A_1": {
                "n": int(g1.count()),
                "mean": float(g1.mean()),
                "std": float(g1.std(ddof=1)),
                "min": float(g1.min()),
                "max": float(g1.max()),
            },
        },
        "mean_ratio_firstA1_over_0": ratio,
        "welch_t_test": {
            "t_stat": float(ttest_res.statistic),
            "p_value_two_sided": float(ttest_res.pvalue),
        },
    }

    # Emit outputs
    print(json.dumps(result, indent=2))
    OUT_JSON.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
