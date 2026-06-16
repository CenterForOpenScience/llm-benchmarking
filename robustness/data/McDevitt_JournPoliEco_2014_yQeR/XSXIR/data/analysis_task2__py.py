import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "/app/data/final_data.dta"
OUT_PATH = "/app/data/results_task2.json"


def main():
    df = pd.read_stata(DATA_PATH)

    # Ensure numeric
    for col in ["complaints_2008", "num_names_2008", "first_A"]:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recode first_A
    df["first_A_cat"] = np.where(df["first_A"] == 1, "A or number", "rest")

    # Model: standardized outcome on first_A and standardized num_names, report F-test for first_A term
    df = df.copy()
    df["complaints_z"] = (df["complaints_2008"] - df["complaints_2008"].mean()) / df["complaints_2008"].std(ddof=1)
    df["num_names_z"] = (df["num_names_2008"] - df["num_names_2008"].mean()) / df["num_names_2008"].std(ddof=1)

    model_df = df[["complaints_z", "first_A_cat", "num_names_z"]].dropna()

    formula = "complaints_z ~ C(first_A_cat) + num_names_z"
    model = smf.ols(formula, data=model_df).fit()

    # Compute Type II-like F test for first_A term by comparing full and reduced models
    reduced = smf.ols("complaints_z ~ num_names_z", data=model_df).fit()
    df1 = (len(model.params) - len(reduced.params))
    df2 = int(model_df.shape[0] - len(model.params))
    ssr_full = np.sum(model.resid ** 2)
    ssr_reduced = np.sum(reduced.resid ** 2)
    msr_diff = (ssr_reduced - ssr_full) / df1
    mse_full = ssr_full / df2
    F_stat = msr_diff / mse_full
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)

    # Extract the coefficient for first_A dummy
    fa_params = [p for p in model.params.index if p.startswith("C(first_A_cat)")]
    fa_param = fa_params[0] if fa_params else None

    results_out = {
        "n": int(model.nobs),
        "model": "OLS",
        "formula": formula,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "F_first_A": float(F_stat),
        "F_df1": int(df1),
        "F_df2": int(df2),
        "F_p": float(p_value),
        "coef_first_A": float(model.params[fa_param]) if fa_param else None,
        "p_first_A_coef": float(model.pvalues[fa_param]) if fa_param else None,
        "param_first_A": fa_param,
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results_out, f, indent=2)

    print(model.summary())
    print(f"F-test for first_A: F({df1}, {df2}) = {F_stat:.3f}, p = {p_value:.3g}")
    if fa_param:
        print(f"First_A effect ({fa_param}): coef={model.params[fa_param]:.4f}, p={model.pvalues[fa_param]:.4g}")


if __name__ == "__main__":
    main()
