import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "/app/data/Benin2012survey.csv"
OUT_PATH = "/app/data/task2_results.json"


def helmert_contrasts(levels):
    k = len(levels)
    cols = [f"C{i}" for i in range(1, k)]
    mapping = {}
    for j, lvl in enumerate(levels):
        row = []
        for i in range(1, k):
            if j < i:
                row.append(-1)
            elif j == i:
                row.append(i)
            else:
                row.append(0)
        mapping[lvl] = dict(zip(cols, row))
    return mapping, cols


def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Remove missing DV as in the R script
    df = df[~df["BoniVote"].isna()].copy()

    # Restrict to Benin-related survey experiment data: already satisfied by file
    # Keep all five passage levels
    levels = ["Control", "Femme", "Nago", "Bariba", "FonFemme"]
    df = df[df["passage"].isin(levels)].copy()

    # Drop rows with missing FonGroup
    if "FonGroup" not in df.columns:
        raise KeyError("Required column 'FonGroup' not found in data.")
    df = df[~df["FonGroup"].isna()].copy()

    # Ensure correct order of levels
    mapping, cols = helmert_contrasts(levels)

    # Build helmert-coded columns for passage
    for c in cols:
        df[f"passage_{c}"] = df["passage"].map(lambda v: mapping.get(v, {}).get(c, np.nan))

    # Sum contrast for FonGroup: -0.5 (non-Fon) vs +0.5 (Fon)
    df["FonGroup_c"] = df["FonGroup"].apply(lambda x: 0.5 if x == 1 else (-0.5 if x == 0 else np.nan))

    # DV
    y = df["BoniVote"].astype(float)

    # Design matrix: intercept + passage_C1.. + FonGroup_c + interactions
    X_parts = []
    for c in cols:
        X_parts.append(df[f"passage_{c}"])
    X = pd.concat(X_parts, axis=1)
    X.columns = [f"passage_{c}" for c in cols]

    # Add FonGroup_c
    X["FonGroup_c"] = df["FonGroup_c"]

    # Interactions
    for c in cols:
        X[f"int_{c}_FG"] = X[f"passage_{c}"] * X["FonGroup_c"]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Drop any remaining NAs
    model_df = pd.concat([y, X], axis=1).dropna()
    y2 = model_df["BoniVote"].astype(float)
    X2 = model_df.drop(columns=["BoniVote"])  

    # Fit logistic regression
    res = sm.Logit(y2, X2).fit(disp=False, maxiter=200)

    # Prepare compact results
    out = {
        "n_obs": int(res.nobs),
        "params": {k: float(v) for k, v in res.params.items()},
        "pvalues": {k: float(v) for k, v in res.pvalues.items()},
        "bse": {k: float(v) for k, v in res.bse.items()},
        "llf": float(res.llf),
        "df_model": float(res.df_model),
        "df_resid": float(res.df_resid),
        "notes": "Key test for claim in 5-level model: interaction on int_C4_FG (fourth Helmert contrast x FonGroup)."
    }

    # Write results
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # Also print a short console summary
    key_term = "int_C4_FG" if "int_C4_FG" in out["pvalues"] else None
    if key_term:
        print(f"Task2: p-value for {key_term} = {out['pvalues'][key_term]:.4g}")
    else:
        print("Task2: Key interaction term not found in results.")


if __name__ == "__main__":
    main()
