import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_PATH = "/app/data/final_data.dta"
OUT_PATH = "/app/data/results_task1.json"


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


def main():
    df = pd.read_stata(DATA_PATH)

    # Ensure numeric types where needed
    if not np.issubdtype(df["complaints_2008"].dtype, np.number):
        df["complaints_2008"] = pd.to_numeric(df["complaints_2008"], errors="coerce")
    if not np.issubdtype(df["num_names_2008"].dtype, np.number):
        df["num_names_2008"] = pd.to_numeric(df["num_names_2008"], errors="coerce")
    if not np.issubdtype(df["first_A"].dtype, np.number):
        df["first_A"] = pd.to_numeric(df["first_A"], errors="coerce")

    # Recode first_A into descriptive categories as in the R code
    df["first_A_cat"] = np.where(df["first_A"] == 1, "A or number", "rest")

    # Standardize outcome and numeric covariate (R's scale uses ddof=1)
    df = df.copy()
    df["complaints_z"] = zscore(df["complaints_2008"].astype(float))
    df["num_names_z"] = zscore(df["num_names_2008"].astype(float))

    # Drop rows with missing values in variables used
    model_df = df[["complaints_z", "first_A_cat", "num_names_z"]].dropna()

    formula = "complaints_z ~ C(first_A_cat) + num_names_z"
    model = smf.ols(formula, data=model_df).fit()

    # Identify the parameter corresponding to first_A
    fa_params = [p for p in model.params.index if p.startswith("C(first_A_cat)")]
    # There should be exactly one because first_A has 2 levels
    fa_param = fa_params[0] if fa_params else None

    results_out = {
        "n": int(model.nobs),
        "model": "OLS",
        "formula": formula,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "coef_first_A": None,
        "se_first_A": None,
        "t_first_A": None,
        "p_first_A": None,
        "param_first_A": fa_param,
        "params": {k: float(v) for k, v in model.params.items()},
        "bse": {k: float(v) for k, v in model.bse.items()},
        "tvalues": {k: float(v) for k, v in model.tvalues.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
    }

    if fa_param is not None:
        results_out["coef_first_A"] = float(model.params[fa_param])
        results_out["se_first_A"] = float(model.bse[fa_param])
        results_out["t_first_A"] = float(model.tvalues[fa_param])
        results_out["p_first_A"] = float(model.pvalues[fa_param])

    with open(OUT_PATH, "w") as f:
        json.dump(results_out, f, indent=2)

    # Also print a concise summary to stdout
    print(model.summary())
    if fa_param is not None:
        print(f"First_A effect ({fa_param}): coef={results_out['coef_first_A']:.4f}, p={results_out['p_first_A']:.4g}")


if __name__ == "__main__":
    main()
