import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# IO paths in container
DATA_PATH = "/app/data/AlTammemi_Survey_deidentify.csv"
OUT_JSON = "/app/data/results_ordinal_motivation.json"
OUT_CSV = "/app/data/results_ordinal_motivation_or.csv"


def load_and_prepare():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Expected data file not found at {DATA_PATH}. Mount or copy it to /app/data.")
    df = pd.read_csv(DATA_PATH)

    # Timestamp filter to remove pre-collection test sessions (mirror Rmd logic)
    df["created_dt"] = pd.to_datetime(df["created"], errors="coerce")
    cutoff = pd.Timestamp("2021-01-19 17:50:32")
    df = df[df["created_dt"] > cutoff].copy()

    # Impute specific Kessler follow-ups (per Analysis_updated.Rmd):
    # If Kessler_3 is NA and Kessler_2 == 1 -> Kessler_3 = 1
    # If Kessler_6 is NA and Kessler_5 == 1 -> Kessler_6 = 1
    for prev_c, curr_c in [("Kessler_2", "Kessler_3"), ("Kessler_5", "Kessler_6")]:
        if prev_c in df.columns and curr_c in df.columns:
            mask = df[curr_c].isna() & (df[prev_c] == 1)
            df.loc[mask, curr_c] = 1

    # Compute Kessler total and categories
    k_cols = [f"Kessler_{i}" for i in range(1, 11)]
    for c in k_cols:
        if c not in df.columns:
            raise KeyError(f"Missing Kessler item: {c}")
    df["Kessler_total"] = df[k_cols].sum(axis=1, min_count=len(k_cols))

    def k_cat(total):
        if pd.isna(total):
            return np.nan
        if total < 20:
            return "No Distress"
        if 20 <= total <= 24:
            return "Low Distress"
        if 25 <= total <= 29:
            return "Moderate Distress"
        return "Severe Distress"

    df["Kessler_categories"] = df["Kessler_total"].apply(k_cat)

    # Outcome: online_learning_1 (assumed ordinal: 1=lowest motivation ... 4=highest)
    if "online_learning_1" not in df.columns:
        raise KeyError("Missing column: online_learning_1")

    # Keep complete cases for outcome and distress categories
    dfa = df.dropna(subset=["online_learning_1", "Kessler_categories"]).copy()

    # Cast outcome to categorical ordered
    # Ensure integer codes from 1..4
    dfa["online_learning_1"] = dfa["online_learning_1"].astype(int)
    valid_levels = sorted(dfa["online_learning_1"].unique())
    # Sanity: they should be [1,2,3,4]
    if min(valid_levels) < 1:
        raise ValueError("Unexpected coding for online_learning_1; expected 1..4")

    # Predictor: Kessler_categories (baseline No Distress)
    cats = ["No Distress", "Low Distress", "Moderate Distress", "Severe Distress"]
    dfa["Kessler_categories"] = pd.Categorical(dfa["Kessler_categories"], categories=cats)
    X = pd.get_dummies(dfa["Kessler_categories"], drop_first=True)
    # Columns will be: Low Distress, Moderate Distress, Severe Distress (baseline No Distress)

    y = dfa["online_learning_1"].astype(int)

    # Fit proportional odds ordinal logistic model
    model = OrderedModel(y, X, distr="logit")
    res = model.fit(method="bfgs", maxiter=200, disp=False)

    # Extract coefficients for distress levels and transform to OR (higher vs lower motivation)
    params = res.params
    conf_int = res.conf_int()

    # Note: params include threshold (cut) parameters and slope coefficients.
    slope_names = [name for name in params.index if name in X.columns]

    out_rows = []
    for name in slope_names:
        est = params[name]
        lo, hi = conf_int.loc[name]
        out_rows.append({
            "predictor": name,
            "estimate_or": float(np.exp(est)),
            "ci_low": float(np.exp(lo)),
            "ci_high": float(np.exp(hi))
        })

    # Identify severe vs no distress row
    severe_row = next((r for r in out_rows if r["predictor"] == "Severe Distress"), None)

    # Save CSV and JSON
    pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False)

    summary = {
        "n_total": int(len(df)),
        "n_analytic": int(len(dfa)),
        "outcome": "online_learning_1 (ordinal 1=low .. 4=high)",
        "predictors": list(X.columns),
        "baseline": "No Distress",
        "model": "Ordered logistic (proportional odds)",
        "key_result_severe_vs_no": severe_row,
        "notes": "OR < 1 implies lower odds of higher motivation relative to baseline (No Distress)."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Also print key result to stdout for logging
    if severe_row is not None:
        try:
            pval = float(res.pvalues.get("Severe Distress", np.nan))
        except Exception:
            pval = np.nan
        print(
            f"[ORDINAL] Severe vs No Distress OR={severe_row['estimate_or']:.3f} "
            f"95% CI=[{severe_row['ci_low']:.3f}, {severe_row['ci_high']:.3f}] "
            f"p-value={pval if not np.isnan(pval) else 'NA'}"
        )

    print(f"Ordinal model complete. Analytic N={len(dfa)}. Results saved to:\n - {OUT_JSON}\n - {OUT_CSV}")


if __name__ == "__main__":
    load_and_prepare()
