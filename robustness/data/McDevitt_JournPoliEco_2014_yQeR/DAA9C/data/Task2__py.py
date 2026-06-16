import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Replicates JMP Oneway t Test comparing complaints_2008 across first_A groups
DATA_FILE = Path("/app/data/final_data.dta")
OUT_JSON = Path("/app/data/task2_result.json")


def main():
    df = pd.read_stata(DATA_FILE)
    # Ensure first_A is treated as categorical; values already 0/1
    df2 = df[["complaints_2008", "first_A"]].dropna(subset=["complaints_2008", "first_A"])  # Illinois only per dataset

    g0 = df2.loc[df2["first_A"] == 0, "complaints_2008"].astype(float)
    g1 = df2.loc[df2["first_A"] == 1, "complaints_2008"].astype(float)

    # Welch two-sample t-test (dual-sided) analogous to JMP t Test
    t_res = stats.ttest_ind(g1, g0, equal_var=False, nan_policy="omit")

    result = {
        "task": "Task2",
        "analysis": "Oneway t-test of complaints_2008 by first_A (dual-sided)",
        "n_total": int(len(df2)),
        "n_first_A_0": int(g0.count()),
        "n_first_A_1": int(g1.count()),
        "group_means": {"first_A_0": float(g0.mean()), "first_A_1": float(g1.mean())},
        "t_stat": float(t_res.statistic),
        "p_value_two_sided": float(t_res.pvalue),
    }

    print(json.dumps(result, indent=2))
    OUT_JSON.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
