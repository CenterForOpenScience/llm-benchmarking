import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "/app/data/data_gerhold.csv"
OUTPUT_JSON = "/app/data/replication_results.json"


def load_and_prepare(path):
    df = pd.read_csv(path)
    # Ensure Germany subset (dataset appears to be only Germany already)
    df = df[df.get("CountryofLiving", "Germany") == "Germany"].copy()

    # Build binary female indicator: prefer provided column; backfill from gender if missing
    if "female" in df.columns:
        df["female_bin"] = df["female"]
    else:
        df["female_bin"] = np.nan

    if "gender" in df.columns:
        # Infer mapping from observed pattern: gender==2 -> female, gender==1 -> male, 3 -> other/unknown
        df.loc[df["gender"] == 2, "female_bin"] = 1.0
        df.loc[df["gender"] == 1, "female_bin"] = 0.0

    # Keep only binary gender observations
    df = df[df["female_bin"].isin([0.0, 1.0])].copy()

    # Normalize weights (if present)
    if "weight_new" in df.columns:
        w = df["weight_new"].astype(float)
        # Avoid division by zero or nan
        w = w.replace([np.inf, -np.inf], np.nan).fillna(w.median())
        w = w / w.mean()
        df["w_norm"] = w
    elif "weight_sample" in df.columns:
        w = df["weight_sample"].astype(float)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(w.median())
        w = w / w.mean()
        df["w_norm"] = w
    else:
        df["w_norm"] = 1.0

    return df


def analyze_gender_diff(df, var):
    res = {"variable": var}
    if var not in df.columns:
        res["available"] = False
        return res
    res["available"] = True

    # Drop missing var values
    d = df[[var, "female_bin", "w_norm"]].dropna().copy()

    # Unweighted group stats
    g = d.groupby("female_bin")[var]
    stats_unw = {
        "n_female": int(g.count().get(1.0, 0)),
        "n_male": int(g.count().get(0.0, 0)),
        "mean_female": float(g.mean().get(1.0, np.nan)),
        "mean_male": float(g.mean().get(0.0, np.nan)),
        "sd_female": float(g.std(ddof=1).get(1.0, np.nan)),
        "sd_male": float(g.std(ddof=1).get(0.0, np.nan)),
    }
    # Welch's t-test
    female_vals = d.loc[d["female_bin"] == 1.0, var].values
    male_vals = d.loc[d["female_bin"] == 0.0, var].values
    if len(female_vals) > 1 and len(male_vals) > 1:
        t_stat, p_val = stats.ttest_ind(female_vals, male_vals, equal_var=False, nan_policy="omit")
    else:
        t_stat, p_val = (np.nan, np.nan)
    stats_unw.update({"t_stat": float(t_stat) if t_stat == t_stat else np.nan, "p_value": float(p_val) if p_val == p_val else np.nan})

    # Weighted difference using WLS: outcome ~ const + female
    y = d[var].astype(float).values
    X = sm.add_constant(d["female_bin"].astype(float).values)
    w = d["w_norm"].astype(float).values
    try:
        model = sm.WLS(y, X, weights=w)
        fit = model.fit()
        # Robust standard errors
        fit_rob = fit.get_robustcov_results(cov_type="HC1")
        coef_female = float(fit_rob.params[1])
        se_female = float(fit_rob.bse[1])
        t_female = float(fit_rob.tvalues[1])
        p_female = float(fit_rob.pvalues[1])
        stats_w = {
            "coef_female": coef_female,
            "se_female": se_female,
            "t_stat": t_female,
            "p_value": p_female,
            "n_used": int(d.shape[0]),
        }
    except Exception as e:
        stats_w = {"error": str(e)}

    res["unweighted"] = stats_unw
    res["weighted_WLS"] = stats_w
    return res


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")

    df = load_and_prepare(DATA_PATH)

    variables = ["mh_anxiety_1", "mh_anxiety_3"]
    results = {
        "description": "Gender differences in anxiety items (Likert 1-5) among adults living in Germany.",
        "n_total": int(df.shape[0]),
        "variables": [],
    }

    for var in variables:
        res_var = analyze_gender_diff(df, var)
        results["variables"].append(res_var)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Also, print a concise summary
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
