import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# Reproducibility
RNG_SEED = int(os.environ.get("PY_SEED", 123))
rng = np.random.default_rng(RNG_SEED)

# Paths (all IO via /app/data)
DATA_FILE = "/workspace/replication_data/analysis-data.csv"
OUT_TEST_SUMMARY = "/app/data/replication_test_summary.csv"
OUT_RESULTS_JSON = "/app/data/replication_results.json"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found at {path}")
    df = pd.read_csv(path)
    return df


def compute_subject_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to Test phase
    df_test = df[df["phase"] == "Test"].copy()

    # Aggregate per subject and condition
    grouped = (
        df_test.groupby(["subject", "condition"], as_index=False)
        .agg(
            acc_2d=("correct", "mean"),
            acc_1d_a=("correct_1d_a", "mean"),
            acc_1d_b=("correct_1d_b", "mean"),
        )
    )
    # 1D score is the max across the two 1D-consistent rules
    grouped["acc_1d"] = grouped[["acc_1d_a", "acc_1d_b"]].max(axis=1)
    return grouped


def summarize_groups(test_summary: pd.DataFrame) -> pd.DataFrame:
    summary = (
        test_summary.groupby("condition", as_index=False)
        .agg(M=("acc_1d", "mean"), SD=("acc_1d", "std"), N=("acc_1d", "size"))
    )
    return summary


def pooled_sd(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return np.sqrt(sp2)


def diff_means_ci_equal_var(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    nx, ny = len(x), len(y)
    sp = pooled_sd(x, y)
    diff = x.mean() - y.mean()
    se = sp * np.sqrt(1.0 / nx + 1.0 / ny)
    df = nx + ny - 2
    tcrit = stats.t.ppf(1 - alpha / 2.0, df)
    lo, hi = diff - tcrit * se, diff + tcrit * se
    return diff, (lo, hi), se, df, sp


def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, rng=None):
    rng = rng or np.random.default_rng()
    nx, ny = len(x), len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        diffs[b] = xb.mean() - yb.mean()
    lo = np.quantile(diffs, alpha / 2.0)
    hi = np.quantile(diffs, 1 - alpha / 2.0)
    return (lo, hi)


def main():
    df = load_data(DATA_FILE)

    # Sanity checks
    # Expect 32 test trials per subject
    n_test = (df["phase"] == "Test").sum()
    # If possible, infer N subjects
    N_subjects = None
    try:
        N_subjects = int(n_test / 32)
    except Exception:
        pass

    test_summary = compute_subject_scores(df)
    test_summary.to_csv(OUT_TEST_SUMMARY, index=False)

    # Split by condition labels expected in data
    conds = sorted(test_summary["condition"].unique())
    if not {"contingent", "full-information"}.issubset(set(conds)):
        raise ValueError(f"Expected conditions 'contingent' and 'full-information' in data, found: {conds}")

    x = test_summary.loc[test_summary["condition"] == "contingent", "acc_1d"].to_numpy()
    y = test_summary.loc[test_summary["condition"] == "full-information", "acc_1d"].to_numpy()

    # Two-sample t-test with equal variances to mirror original R code
    t_res = stats.ttest_ind(x, y, equal_var=True)

    # Mean difference and 95% CI (equal variance)
    diff, (ci_lo, ci_hi), se, df, sp = diff_means_ci_equal_var(x, y, alpha=0.05)

    # Cohen's d (pooled SD)
    d = diff / sp if sp > 0 else np.nan

    # Bootstrap CI for robustness
    boot_lo, boot_hi = bootstrap_diff_ci(x, y, n_boot=10000, alpha=0.05, rng=rng)

    # Group summaries
    group_summary = summarize_groups(test_summary)
    group_stats = {
        row["condition"]: {"mean": float(row["M"]), "sd": float(row["SD"]), "n": int(row["N"]) }
        for _, row in group_summary.iterrows()
    }

    results = {
        "n_subjects_inferred": int(N_subjects) if N_subjects is not None else None,
        "group_stats_acc_1d": group_stats,
        "mean_diff_contingent_minus_fullinfo": float(diff),
        "t_stat": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
        "df": int(df),
        "pooled_sd": float(sp),
        "se_diff": float(se),
        "ci95_diff_equal_var": [float(ci_lo), float(ci_hi)],
        "cohens_d": float(d),
        "bootstrap_ci95_diff": [float(boot_lo), float(boot_hi)],
        "notes": "acc_1d is the subject-level 1D score computed as the maximum of mean(correct_1d_a) and mean(correct_1d_b) across 32 test trials."
    }

    with open(OUT_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Also print a concise summary
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
