import os
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

DATA_CANDIDATES = [
    "/workspace/replication_data/data_gerhold.csv",
    "/app/data/original/4/python/replication_data/data_gerhold.csv"
]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("Data file not found in any of: " + ", ".join(DATA_CANDIDATES))
OUT_DIR = "/workspace/replication_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def f_test_equal_variance(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    s1 = np.var(x, ddof=1)
    s2 = np.var(y, ddof=1)
    # ratio as larger variance over smaller variance
    if s1 >= s2:
        F = s1 / s2 if s2 > 0 else np.inf
        df1, df2 = n1 - 1, n2 - 1
    else:
        F = s2 / s1 if s1 > 0 else np.inf
        df1, df2 = n2 - 1, n1 - 1
    if not np.isfinite(F):
        return F, 0.0
    # two-sided p-value
    p_one = stats.f.sf(F, df1, df2)
    p_two = 2 * min(p_one, 1 - p_one)
    # since F >= 1 by construction, p_one <= 0.5, so p_two ~ 2*p_one
    p_two = max(min(p_two, 1.0), 0.0)
    return F, p_two


def ttest_with_ci(x: np.ndarray, y: np.ndarray, equal_var: bool = True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    res = stats.ttest_ind(x, y, equal_var=equal_var)
    diff = m1 - m2
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df if df > 0 else np.nan
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2)) if n1 > 0 and n2 > 0 else np.nan
    else:
        se = math.sqrt(v1 / n1 + v2 / n2) if n1 > 0 and n2 > 0 else np.nan
        # Welch-Satterthwaite df
        num = (v1 / n1 + v2 / n2) ** 2
        den = 0.0
        if n1 > 1:
            den += (v1 / n1) ** 2 / (n1 - 1)
        if n2 > 1:
            den += (v2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else np.nan
    if np.isnan(se) or np.isnan(df) or df <= 0:
        ci = [np.nan, np.nan]
    else:
        tcrit = stats.t.ppf(0.975, df)
        ci = [diff - tcrit * se, diff + tcrit * se]
    return {
        "n_female": int(n1),
        "n_male": int(n2),
        "mean_female": float(m1),
        "mean_male": float(m2),
        "diff_female_minus_male": float(diff),
        "t_stat": float(res.statistic) if np.isfinite(res.statistic) else res.statistic,
        "p_value": float(res.pvalue) if np.isfinite(res.pvalue) else res.pvalue,
        "df": float(df) if np.isfinite(df) else df,
        "se_diff": float(se) if np.isfinite(se) else se,
        "ci_95": ci,
        "equal_var": bool(equal_var),
    }


def cohen_d_pooled(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df if df > 0 else np.nan
    sp = math.sqrt(sp2) if sp2 > 0 else np.nan
    if np.isnan(sp) or sp == 0:
        return np.nan
    return (m1 - m2) / sp


def two_proportion_test(k1: int, n1: int, k2: int, n2: int):
    p1 = k1 / n1 if n1 > 0 else np.nan
    p2 = k2 / n2 if n2 > 0 else np.nan
    diff = p1 - p2 if (np.isfinite(p1) and np.isfinite(p2)) else np.nan
    # pooled standard error for z-test
    p_pool = (k1 + k2) / (n1 + n2) if (n1 + n2) > 0 else np.nan
    if any(np.isnan(val) for val in [p1, p2, p_pool]):
        return {
            "p_female": p1, "p_male": p2, "diff_pp": diff,
            "z_stat": np.nan, "p_value": np.nan,
            "ci_95_unpooled": [np.nan, np.nan]
        }
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else np.nan
    pval = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    # unpooled CI
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = stats.norm.ppf(0.975)
    ci = [diff - zcrit * se_unpooled, diff + zcrit * se_unpooled] if se_unpooled > 0 else [np.nan, np.nan]
    return {
        "p_female": p1, "p_male": p2, "diff_pp": diff,
        "z_stat": z, "p_value": pval,
        "ci_95_unpooled": ci
    }


def main():
    df = pd.read_csv(DATA_PATH)
    # Ensure relevant columns exist
    required_cols = ["gender", "female", "mh_anxiety_1", "mh_anxiety_3"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Exclude gender == 3 (treat as string per original R script)
    df["gender_str"] = df["gender"].astype(str)
    df = df[df["gender_str"] != "3"].copy()

    # Construct robust female binary indicator
    # Prefer 'female' column if present; coerce to numeric and binarize
    fem_num = pd.to_numeric(df["female"], errors="coerce")
    if fem_num.notna().any():
        df["female_bin"] = (fem_num >= 0.5).astype(int)
    else:
        # Fallback to 'gender' if needed (assume 2=female, 1=male)
        if "gender" in df.columns:
            df["female_bin"] = (pd.to_numeric(df["gender"], errors="coerce") == 2).astype(int)
        else:
            raise ValueError("Cannot determine female indicator: missing/invalid 'female' and 'gender'.")

    # Split groups using female_bin
    female_group = df[df["female_bin"] == 1].copy()
    male_group = df[df["female_bin"] == 0].copy()

    # Clean types for outcome variables
    for c in ["mh_anxiety_1", "mh_anxiety_3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Define groups using robust female_bin constructed above
    female_group = df[df["female_bin"] == 1].copy()
    male_group = df[df["female_bin"] == 0].copy()

    # F-test for variance equality and t-test for mh_anxiety_1
    x1 = female_group["mh_anxiety_1"].dropna().to_numpy()
    y1 = male_group["mh_anxiety_1"].dropna().to_numpy()

    F1, pF1 = f_test_equal_variance(x1, y1)
    equal_var1 = (pF1 > 0.05) if np.isfinite(pF1) else False
    t1 = ttest_with_ci(x1, y1, equal_var=equal_var1)
    d1 = cohen_d_pooled(x1, y1)

    # Exploratory: mh_anxiety_3
    x3 = female_group["mh_anxiety_3"].dropna().to_numpy()
    y3 = male_group["mh_anxiety_3"].dropna().to_numpy()
    F3, pF3 = f_test_equal_variance(x3, y3)
    equal_var3 = (pF3 > 0.05) if np.isfinite(pF3) else False
    t3 = ttest_with_ci(x3, y3, equal_var=equal_var3)
    d3 = cohen_d_pooled(x3, y3)

    # Proportion agreeing (4 or 5) on mh_anxiety_1
    agree_f = np.sum(x1 >= 4)
    agree_m = np.sum(y1 >= 4)
    prop_test = two_proportion_test(int(agree_f), len(x1), int(agree_m), len(y1))

    # Descriptives
    desc = {
        "overall": {
            "N": int(df.shape[0])
        },
        "female": {
            "N": int(female_group.shape[0]),
            "mh_anxiety_1_mean": float(np.nanmean(x1)) if len(x1) else np.nan,
            "mh_anxiety_1_sd": float(np.nanstd(x1, ddof=1)) if len(x1) > 1 else np.nan,
            "mh_anxiety_3_mean": float(np.nanmean(x3)) if len(x3) else np.nan,
            "mh_anxiety_3_sd": float(np.nanstd(x3, ddof=1)) if len(x3) > 1 else np.nan,
        },
        "male": {
            "N": int(male_group.shape[0]),
            "mh_anxiety_1_mean": float(np.nanmean(y1)) if len(y1) else np.nan,
            "mh_anxiety_1_sd": float(np.nanstd(y1, ddof=1)) if len(y1) > 1 else np.nan,
            "mh_anxiety_3_mean": float(np.nanmean(y3)) if len(y3) else np.nan,
            "mh_anxiety_3_sd": float(np.nanstd(y3, ddof=1)) if len(y3) > 1 else np.nan,
        },
    }

    results = {
        "hypothesis": "Women (female==1) will report higher worry about COVID-19 than men (female==0), operationalized as higher mean mh_anxiety_1 and higher agree proportion (4/5).",
        "filters": {"excluded_gender_eq_3": True},
        "tests": {
            "variance_test_mh_anxiety_1": {"F": F1, "p_value": pF1},
            "t_test_mh_anxiety_1": t1,
            "cohens_d_mh_anxiety_1": d1,
            "variance_test_mh_anxiety_3": {"F": F3, "p_value": pF3},
            "t_test_mh_anxiety_3": t3,
            "cohens_d_mh_anxiety_3": d3,
            "two_prop_test_agree_mh_anxiety_1": prop_test,
        },
        "descriptives": desc,
        "notes": "Equal variances determined by F-test p>0.05; two-proportion z uses pooled SE for test and unpooled SE for 95% CI.",
    }

    # Save JSON
    out_json = os.path.join(OUT_DIR, "results_Gerhold_covid_Azg9_0948_final.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Also save a brief CSV summary for mh_anxiety_1
    rows = []
    rows.append({
        "group": "female", "n": t1["n_female"], "mean": t1["mean_female"],
        "sd": float(np.nanstd(x1, ddof=1)) if len(x1) > 1 else np.nan,
        "agree_rate_4or5": prop_test["p_female"],
    })
    rows.append({
        "group": "male", "n": t1["n_male"], "mean": t1["mean_male"],
        "sd": float(np.nanstd(y1, ddof=1)) if len(y1) > 1 else np.nan,
        "agree_rate_4or5": prop_test["p_male"],
    })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "summary_mh_anxiety_1.csv"), index=False)

    # Print concise summary to stdout
    print("Replication results (Gerhold_covid_Azg9_0948):")
    print(f"N: total={desc['overall']['N']}, female={desc['female']['N']}, male={desc['male']['N']}")
    print("mh_anxiety_1: t-test (female - male)")
    print(f"  F-test p={pF1:.4g} -> equal_var={equal_var1}")
    print(f"  diff={t1['diff_female_minus_male']:.4f}, t={t1['t_stat']:.4f}, df={t1['df']:.2f}, p={t1['p_value']:.4g}, CI95={t1['ci_95']}")
    print(f"  Cohen's d={d1:.4f}")
    print("Agree (4/5) on mh_anxiety_1: two-proportion z-test (female - male)")
    print(f"  diff_pp={prop_test['diff_pp']:.4f}, z={prop_test['z_stat']:.4f}, p={prop_test['p_value']:.4g}, CI95_unpooled={prop_test['ci_95_unpooled']}")
    print("Exploratory mh_anxiety_3: t-test (female - male)")
    print(f"  F-test p={pF3:.4g} -> equal_var={equal_var3}")
    print(f"  diff={t3['diff_female_minus_male']:.4f}, t={t3['t_stat']:.4f}, df={t3['df']:.2f}, p={t3['p_value']:.4g}, CI95={t3['ci_95']}")


if __name__ == "__main__":
    main()
