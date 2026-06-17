import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12345)

DATA_PATH = "/app/data/RiskData.dta"
ARTIFACTS_DIR = "/app/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def kendall_tau_b(x, y):
    # Pure Python/Numpy Kendall's tau-b implementation
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.shape[0]
    C = D = tie_x = tie_y = tie_both = 0
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx == 0 and dy == 0:
                tie_both += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            else:
                prod = dx * dy
                if prod > 0:
                    C += 1
                elif prod < 0:
                    D += 1
    Tx = tie_x + tie_both
    Ty = tie_y + tie_both
    denom = math.sqrt((C + D + Tx) * (C + D + Ty))
    if denom == 0:
        return 0.0
    return (C - D) / denom


def bootstrap_ci_stat(x, y, stat_fn, n_boot=2000, ci=0.95, random_state=12345):
    rng = np.random.default_rng(random_state)
    n = len(x)
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        vals.append(stat_fn(x[samp], y[samp]))
    vals = np.array(vals)
    lower = float(np.quantile(vals, (1 - ci) / 2))
    upper = float(np.quantile(vals, 1 - (1 - ci) / 2))
    return [lower, upper]


def permutation_p_value(x, y, stat_fn, alternative="less", n_perm=2000, random_state=12345):
    rng = np.random.default_rng(random_state)
    observed = stat_fn(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        s = stat_fn(x, y_perm)
        if alternative == "less":
            if s <= observed:
                count += 1
        elif alternative == "greater":
            if s >= observed:
                count += 1
        else:  # two-sided
            if abs(s) >= abs(observed):
                count += 1
    p = (count + 1) / (n_perm + 1)
    return float(p), float(observed)


def pearson_r(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    xm = x - x.mean()
    ym = y - y.mean()
    num = np.sum(xm * ym)
    den = math.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if den == 0:
        return 0.0
    return num / den


def main():
    # Load data
    df = pd.read_stata(DATA_PATH)

    # Exclude non-participants (age==99)
    df = df[df["age"] != 99].copy()

    # Compute sums for PV (Risk11..Risk110) and RV (Risk21..Risk210)
    risk1_cols = [f"Risk1{i}" for i in list(range(1,10)) + [10]]
    risk2_cols = [f"Risk2{i}" for i in list(range(1,10)) + [10]]
    for col in risk1_cols + risk2_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    df["risk1_sum"] = df[risk1_cols].sum(axis=1)
    df["risk2_sum"] = df[risk2_cols].sum(axis=1)
    df["risk_avg_safe"] = (df["risk1_sum"] + df["risk2_sum"]) / 2.0

    # Decision errors in LV (exclude item 5):
    lv_cols_needed = ["Risk31","Risk32","Risk33","Risk34","Risk36","Risk37","Risk38","Risk39","Risk310"]
    for col in lv_cols_needed:
        if col not in df.columns:
            raise ValueError(f"Missing expected LV column: {col}")

    df["decision_errors_lv"] = (
        (1 - df["Risk31"]) + (1 - df["Risk32"]) + (1 - df["Risk33"]) + (1 - df["Risk34"]) +
        df["Risk36"] + df["Risk37"] + df["Risk38"] + df["Risk39"] + df["Risk310"]
    )

    # Drop rows with missing in constructed variables
    df2 = df.dropna(subset=["risk_avg_safe","decision_errors_lv"]).copy()
    x = df2["risk_avg_safe"].to_numpy()
    y = df2["decision_errors_lv"].to_numpy()

    # Kendall's tau-b with bootstrap CI and permutation p-value (one-sided less)
    tau = kendall_tau_b(x, y)
    ci_kendall_boot = bootstrap_ci_stat(x, y, kendall_tau_b, n_boot=2000, ci=0.95, random_state=12345)
    p1_kendall, _ = permutation_p_value(x, y, kendall_tau_b, alternative="less", n_perm=3000, random_state=54321)

    # Robustness: Kendall PV-only and RV-only (permutation p-values only)
    x_pv = df2["risk1_sum"].to_numpy()
    tau_pv = kendall_tau_b(x_pv, y)
    p1_pv, _ = permutation_p_value(x_pv, y, kendall_tau_b, alternative="less", n_perm=2000, random_state=111)

    x_rv = df2["risk2_sum"].to_numpy()
    tau_rv = kendall_tau_b(x_rv, y)
    p1_rv, _ = permutation_p_value(x_rv, y, kendall_tau_b, alternative="less", n_perm=2000, random_state=222)

    # Pearson r with bootstrap CI and permutation p-value (one-sided less)
    r = pearson_r(x, y)
    ci_pearson = bootstrap_ci_stat(x, y, pearson_r, n_boot=3000, ci=0.95, random_state=13579)
    p1_r, _ = permutation_p_value(x, y, pearson_r, alternative="less", n_perm=5000, random_state=333)

    # Save numerical results
    results = {
        "n_obs": int(len(df2)),
        "kendall": {
            "tau_b": float(tau),
            "p_value_one_sided_less": float(p1_kendall),
            "ci_95_bootstrap": ci_kendall_boot
        },
        "kendall_pv_only": {
            "tau_b": float(tau_pv),
            "p_value_one_sided_less": float(p1_pv)
        },
        "kendall_rv_only": {
            "tau_b": float(tau_rv),
            "p_value_one_sided_less": float(p1_rv)
        },
        "pearson": {
            "r": float(r),
            "p_value_one_sided_less": float(p1_r),
            "ci_95_bootstrap": ci_pearson
        }
    }

    with open(os.path.join(ARTIFACTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Also write a CSV summary table
    table_rows = [
        ["kendall_tau_b", results["kendall"]["tau_b"], "NA", str(results["kendall"]["ci_95_bootstrap"]), "NA", "one-sided p (less)=" + str(results["kendall"]["p_value_one_sided_less"])],
        ["kendall_tau_b_pv_only", results["kendall_pv_only"]["tau_b"], "NA", "NA", "NA", "one-sided p (less)=" + str(results["kendall_pv_only"]["p_value_one_sided_less"])],
        ["kendall_tau_b_rv_only", results["kendall_rv_only"]["tau_b"], "NA", "NA", "NA", "one-sided p (less)=" + str(results["kendall_rv_only"]["p_value_one_sided_less"])],
        ["pearson_r", results["pearson"]["r"], "NA", str(results["pearson"]["ci_95_bootstrap"]), "NA", "one-sided p (less)=" + str(results["pearson"]["p_value_one_sided_less"])],
    ]
    df_table = pd.DataFrame(table_rows, columns=["statistic", "value", "standard_error", "confidence_interval", "p_value", "notes"])
    df_table.to_csv(os.path.join(ARTIFACTS_DIR, "table_correlations.csv"), index=False)

    # Visualization: scatter/count with simple linear fit line (no LOWESS to avoid native deps)
    counts = df2.groupby(["risk_avg_safe", "decision_errors_lv"]).size().reset_index(name="count")

    plt.figure(figsize=(7, 5))
    sizes = 20 + 10 * np.log1p(counts["count"])  # scale sizes
    plt.scatter(counts["risk_avg_safe"], counts["decision_errors_lv"], s=sizes, alpha=0.7, edgecolor='k')

    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    plt.plot(x_line, y_line, color='tab:blue', linewidth=2)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("Average number of safe choices in RV and PV")
    plt.ylabel("Number of decision errors in LV")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(ARTIFACTS_DIR, "figure_scatter_fit.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Print a concise summary to stdout
    print("Analysis complete. N=", len(df2))
    print("Kendall tau-b:", tau, "; one-sided (less) p=", p1_kendall, "; 95% boot CI:", ci_kendall_boot)
    print("Pearson r:", r, "; one-sided (less) p=", p1_r, "; 95% boot CI:", ci_pearson)


if __name__ == "__main__":
    main()
