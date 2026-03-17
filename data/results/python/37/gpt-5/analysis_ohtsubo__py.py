import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
INPUT_CSV = os.path.join(DATA_DIR, "ohtsubo data.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "ohtsubo_ttest_results.json")
OUTPUT_TXT = os.path.join(DATA_DIR, "ohtsubo_ttest_summary.txt")


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    # Cronbach's alpha for items in columns
    items = df_items.to_numpy(dtype=float)
    k = items.shape[1]
    if k < 2:
        return float("nan")
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    alpha = (k / (k - 1.0)) * (1 - (item_vars.sum() / total_var))
    return float(alpha)


def cohen_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    # Pooled SD Cohen's d for two independent groups
    nx = len(x)
    ny = len(y)
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if sp2 <= 0:
        return float("nan")
    sp = np.sqrt(sp2)
    d = (x.mean() - y.mean()) / sp
    return float(d)


def hedges_g(d: float, nx: int, ny: int) -> float:
    # Small-sample bias correction
    df = nx + ny - 2
    if df <= 0 or np.isnan(d):
        return float("nan")
    J = 1 - (3 / (4 * df - 1))
    return float(J * d)


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input data not found at {INPUT_CSV}. Ensure the file is mounted under /app/data.")

    df = pd.read_csv(INPUT_CSV)

    # Ensure required columns exist
    required_cols = ["a1", "a2", "a3", "a4", "a5", "a6", "condition"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Reverse-code a4: SPSS recode was (7->1, 6->2, ..., 1->7) => a4r = 8 - a4
    df["a4r"] = 8 - df["a4"]

    # Compute total intimacy as in SPSS: atot = a1 + a2 + a3 + a4r + a5 + a6
    df["atot"] = df[["a1", "a2", "a3", "a4r", "a5", "a6"]].sum(axis=1)

    # Also compute mean-scale version for interpretability on 1-7 scale
    df["aint_mean"] = df["atot"] / 6.0

    # Drop rows with any missing in analysis variables
    analysis_df = df.dropna(subset=["atot", "aint_mean", "condition"]).copy()

    # Enforce binary groups 0/1
    analysis_df = analysis_df[analysis_df["condition"].isin([0, 1])]

    g1 = analysis_df.loc[analysis_df["condition"] == 1, "atot"].to_numpy(dtype=float)
    g0 = analysis_df.loc[analysis_df["condition"] == 0, "atot"].to_numpy(dtype=float)

    # T-tests (Welch and pooled)
    t_welch, p_welch = stats.ttest_ind(g1, g0, equal_var=False)
    t_pooled, p_pooled = stats.ttest_ind(g1, g0, equal_var=True)

    # Degrees of freedom
    nx, ny = len(g1), len(g0)
    df_pooled = nx + ny - 2
    # Welch-Satterthwaite df
    vx = g1.var(ddof=1)
    vy = g0.var(ddof=1)
    welch_df = (vx / nx + vy / ny) ** 2 / (((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1))

    # Effect sizes
    d = cohen_d_independent(g1, g0)
    g = hedges_g(d, nx, ny)

    # Group stats for atot and mean-scale
    def stats_for(col):
        x1 = analysis_df.loc[analysis_df["condition"] == 1, col]
        x0 = analysis_df.loc[analysis_df["condition"] == 0, col]
        return {
            "n_attention_1": int(x1.shape[0]),
            "n_attention_0": int(x0.shape[0]),
            "mean_attention_1": float(x1.mean()),
            "mean_attention_0": float(x0.mean()),
            "sd_attention_1": float(x1.std(ddof=1)),
            "sd_attention_0": float(x0.std(ddof=1)),
        }

    atot_stats = stats_for("atot")
    aint_stats = stats_for("aint_mean")

    # Reliability (Cronbach's alpha) for items used (after reverse-coding a4)
    alpha = cronbach_alpha(analysis_df[["a1", "a2", "a3", "a4r", "a5", "a6"]])

    results = {
        "input_file": INPUT_CSV,
        "n_total": int(analysis_df.shape[0]),
        "outcome": "Intimacy total score (atot) = a1+a2+a3+a4r+a5+a6; aint_mean = atot/6",
        "grouping": "condition (1 = attention; 0 = no attention)",
        "cronbach_alpha_items6": alpha,
        "group_stats_atot": atot_stats,
        "group_stats_aint_mean": aint_stats,
        "t_test_atot": {
            "welch": {"t": float(t_welch), "df": float(welch_df), "p_value": float(p_welch)},
            "pooled": {"t": float(t_pooled), "df": int(df_pooled), "p_value": float(p_pooled)}
        },
        "effect_sizes": {
            "cohens_d_pooled": d,
            "hedges_g": g
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    lines = []
    lines.append("Ohtsubo replication t-test on intimacy by attention condition")
    lines.append(f"Input: {INPUT_CSV}")
    lines.append(f"N total (used): {results['n_total']}")
    lines.append(f"Cronbach alpha (6 items): {alpha:.3f}")
    lines.append("")
    lines.append("Group stats (atot sum):")
    for k, v in atot_stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Group stats (aint_mean 1-7 scale):")
    for k, v in aint_stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("T-tests on atot:")
    lines.append(f"  Welch t({welch_df:.2f}) = {t_welch:.3f}, p = {p_welch:.4g}")
    lines.append(f"  Pooled t({df_pooled}) = {t_pooled:.3f}, p = {p_pooled:.4g}")
    lines.append(f"Effect sizes: Cohen's d (pooled) = {d:.3f}, Hedges' g = {g:.3f}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Analysis complete. Results written to:")
    print("  ", OUTPUT_JSON)
    print("  ", OUTPUT_TXT)


if __name__ == "__main__":
    main()
