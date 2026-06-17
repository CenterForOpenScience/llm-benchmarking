import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# Paths (IO must use /app/data)
HOUSEHOLD_PATH = os.environ.get("HOUSEHOLD_PATH", "/app/data/AEJApp-2009-0289-data/household.dta")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/data/outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def welch_ttest(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Welch's t-test via scipy
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')

    # Welch-Satterthwaite degrees of freedom
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se_x2 = vx / nx if nx > 0 else np.nan
    se_y2 = vy / ny if ny > 0 else np.nan
    denom = (se_x2 + se_y2) ** 2
    num = (se_x2**2) / (nx - 1) if nx > 1 else 0.0
    num += (se_y2**2) / (ny - 1) if ny > 1 else 0.0
    df = denom / num if num != 0 else np.nan

    return {
        "t_stat": float(t_stat) if t_stat is not None else np.nan,
        "p_value": float(p_val) if p_val is not None else np.nan,
        "df": float(df) if not np.isnan(df) else None,
        "n_group1": int(nx),
        "n_group2": int(ny),
        "mean_group1": float(np.mean(x)) if nx > 0 else None,
        "mean_group2": float(np.mean(y)) if ny > 0 else None,
        "sd_group1": float(np.std(x, ddof=1)) if nx > 1 else None,
        "sd_group2": float(np.std(y, ddof=1)) if ny > 1 else None,
    }


def run_task_ttest_caste_income(task_name: str):
    df = pd.read_stata(HOUSEHOLD_PATH)

    # Ensure required columns exist
    if not set(["totinc", "caste"]).issubset(df.columns):
        raise ValueError("Required columns 'totinc' and 'caste' not found in household data.")

    # Clean caste strings
    df["caste"] = df["caste"].astype(str).str.strip()

    # Define grouping: low caste = 'ST/SC'; other castes combined as comparison group
    low_label = "ST/SC"
    if low_label not in set(df["caste"].unique()):
        raise ValueError("Expected caste category 'ST/SC' not found; cannot construct low-caste vs other grouping.")

    df_valid = df.loc[df["totinc"].notna() & df["caste"].notna(), ["totinc", "caste"]].copy()
    g1 = df_valid.loc[df_valid["caste"] == low_label, "totinc"].values
    g2 = df_valid.loc[df_valid["caste"] != low_label, "totinc"].values

    res = welch_ttest(g1, g2)
    res.update({
        "task": task_name,
        "dataset": os.path.basename(HOUSEHOLD_PATH),
        "outcome": "totinc",
        "group_var": "caste",
        "group1_label": low_label,
        "group2_label": "not_" + low_label,
        "test": "Welch_t_test",
        "alternative": "two-sided"
    })

    out_path = os.path.join(OUTPUT_DIR, f"{task_name.lower()}_ttest_caste_income.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps(res))


if __name__ == "__main__":
    # Task1: t-test income by caste (ST/SC vs others)
    run_task_ttest_caste_income("Task1")

    # Task2: per instruction, do not use controls; replicate the raw t-test of income by caste
    run_task_ttest_caste_income("Task2")
