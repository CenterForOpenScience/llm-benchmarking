import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime

# Paths
DATA_PATH = "/app/data/AlTammemi_Survey_deidentify.csv"
OUT_JSON = "/app/data/results_mnlogit.json"
OUT_CSV = "/app/data/results_mnlogit_or.csv"


def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Expected data file not found at {DATA_PATH}. Make sure to mount data to /app/data.")
    df = pd.read_csv(DATA_PATH)

    # Parse and filter by created timestamp to remove pre-collection test sessions
    # Mirrors R: as.POSIXct(DF$created, tz = "America/New_York") > as.POSIXct("2021-01-19 17:50:32", tz = "America/New_York")
    # Here we parse naively and compare to naive cutoff; this retains the intended ordering filter.
    df["created_dt"] = pd.to_datetime(df["created"], errors="coerce")
    cutoff = pd.Timestamp("2021-01-19 17:50:32")
    df = df[df["created_dt"] > cutoff].copy()

    # Impute specific Kessler follow-up items per R code comments:
    # If Kessler_3 is NA and Kessler_2 == 1, set Kessler_3 = 1
    # If Kessler_6 is NA and Kessler_5 == 1, set Kessler_6 = 1
    for col_prev, col_curr in [("Kessler_2", "Kessler_3"), ("Kessler_5", "Kessler_6")]:
        if col_prev in df.columns and col_curr in df.columns:
            mask = df[col_curr].isna() & (df[col_prev] == 1)
            df.loc[mask, col_curr] = 1

    # Compute Kessler total and categories
    k_cols = [f"Kessler_{i}" for i in range(1, 11)]
    for c in k_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Sum across all 10 items; if any are missing, result is NaN (min_count enforces this)
    df["Kessler_total"] = df[k_cols].sum(axis=1, min_count=len(k_cols))

    # Categorize per thresholds
    def categorize(total):
        if pd.isna(total):
            return np.nan
        if total < 20:
            return "No Distress"
        elif 20 <= total <= 24:
            return "Low Distress"
        elif 25 <= total <= 29:
            return "Moderate Distress"
        else:
            return "Severe Distress"

    df["Kessler_categories"] = df["Kessler_total"].apply(categorize)

    # Complete cases for outcome and predictor
    needed = ["Kessler_categories", "online_learning_1"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    dfa = df.dropna(subset=needed).copy()

    # Prepare data for multinomial logit: outcome is Kessler_categories (base: No Distress)
    # Predictor: online_learning_1 (treated as numeric ordinal per original analysis)
    cats = ["No Distress", "Low Distress", "Moderate Distress", "Severe Distress"]
    y_cat = pd.Categorical(dfa["Kessler_categories"], categories=cats)
    y = y_cat.codes  # 0..J-1, with -1 for NaN (already dropped)

    X = dfa[["online_learning_1"]].astype(float)
    X = sm.add_constant(X, has_constant="add")

    # Fit MNLogit    # Fit MNLogit
    model = sm.MNLogit(y, X)
    result = model.fit(method="newton", maxiter=200, disp=False)

    # Extract odds ratios (relative risk ratios) and 95% CI using params and bse (robust to conf_int format differences)
    params = result.params.copy()  # shape: (J-1) x K
    bse = result.bse.copy()

    from scipy.stats import norm
    zcrit = norm.ppf(0.975)

    lo = params - zcrit * bse
    hi = params + zcrit * bse

    or_df = np.exp(params)
    ci_low = np.exp(lo)
    ci_high = np.exp(hi)

    # Build a tidy table    # Build a tidy table
    or_table = []
    row_cats = cats[1:]  # rows correspond to categories vs base (No Distress)
    for i, row in enumerate(or_df.index):
        cat_name = row_cats[i] if i < len(row_cats) else str(row)
        for coef_name in or_df.columns:
            or_table.append({
                "contrast": f"{cat_name} vs No Distress",
                "term": coef_name,
                "estimate_rr": float(or_df.loc[row, coef_name]),
                "ci_low": float(ci_low.loc[row, coef_name]),
                "ci_high": float(ci_high.loc[row, coef_name])
            })

    # Save CSV
    or_df_out = pd.DataFrame(or_table)
    or_df_out.to_csv(OUT_CSV, index=False)

    # Prepare JSON summary, focusing on the key contrast (Severe vs No Distress) and predictor term
    severe_row = next((r for r in or_table if r["contrast"] == "Severe Distress vs No Distress" and r["term"] == "online_learning_1"), None)

    summary = {
        "n_total": int(len(df)),
        "n_analytic": int(len(dfa)),
        "outcome_levels": cats,
        "base_level": cats[0],
        "predictor": "online_learning_1",
        "model": "Multinomial Logit (statsmodels.MNLogit)",
        "key_result_severe_vs_no": severe_row,
        "all_results": or_table
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Print key RRR for logging
    if severe_row is not None:
        print(
            f"[MNLOGIT] Severe vs No Distress RRR for online_learning_1 = {severe_row['estimate_rr']:.3f} "
            f"95% CI=[{severe_row['ci_low']:.3f}, {severe_row['ci_high']:.3f}]"
        )

    print(f"Analysis complete. Analytic N={len(dfa)}. Results saved to:\n - {OUT_JSON}\n - {OUT_CSV}")


if __name__ == "__main__":
    main()
