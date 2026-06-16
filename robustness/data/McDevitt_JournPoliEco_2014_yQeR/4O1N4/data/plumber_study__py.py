import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

# All IO must use /app/data per run policy
DATA_PATH = "/app/data/final_data.dta"
OUT_JSON = "/app/data/results_task1.json"


def effect_code(series):
    # Expect binary 0/1; map to -0.5 / 0.5
    return series.map({0: -0.5, 1: 0.5}).astype(float)


def add_constructed_vars(df):
    df = df.copy()
    # Continuous centers
    df["ad_spendingC"] = df["ad_spending"] - df["ad_spending"].mean()
    df["firm_ageC"] = df["firm_age"] - df["firm_age"].mean()
    df["log_emp_size"] = np.log(df["emp_size"].replace(0, np.nan))
    df["log_emp_size"] = df["log_emp_size"].replace([-np.inf, np.inf], np.nan)
    df["emp_sizeC"] = df["log_emp_size"] - df["log_emp_size"].mean()
    df["num_names_2008C"] = df["num_names_2008"] - df["num_names_2008"].mean()

    # Effect-coded binaries
    for col in ["first_A", "multiple_names", "chicago", "on_google"]:
        if col in df.columns:
            df[f"{col}_ec"] = effect_code(df[col].astype(int))
    return df


def robust_glm_poisson(formula, data):
    # Fit GLM Poisson with robust covariance directly
    model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit(cov_type="HC0")
    return model, model


def summarise_glm_results(rob):
    # Ensure we work with pandas objects robustly
    params = rob.params
    bse = rob.bse
    pvals = rob.pvalues
    conf = rob.conf_int(alpha=0.05)
    # IRR and CI for IRR from link-scale CI
    irr = np.exp(params)
    # conf may be a DataFrame with columns [0, 1]
    if hasattr(conf, "iloc"):
        ci_lower = conf.iloc[:, 0]
        ci_upper = conf.iloc[:, 1]
    else:
        # fallback to numpy array
        ci_lower = conf[:, 0]
        ci_upper = conf[:, 1]
    irr_ci_lower = np.exp(ci_lower)
    irr_ci_upper = np.exp(ci_upper)
    # Robust SE for IRR via delta method: SE_IRR = exp(beta) * SE_beta
    irr_se = irr * bse
    out = {}
    for name in params.index:
        out[name] = {
            "coef": float(params[name]),
            "se_robust": float(bse[name]),
            "p_value": float(pvals[name]),
            "ci_lower": float(ci_lower[name] if hasattr(ci_lower, "__getitem__") else ci_lower[params.index.get_loc(name)]),
            "ci_upper": float(ci_upper[name] if hasattr(ci_upper, "__getitem__") else ci_upper[params.index.get_loc(name)]),
            "irr": float(irr[name]),
            "irr_se_delta": float(irr_se[name]),
            "irr_ci_lower": float(irr_ci_lower[name] if hasattr(irr_ci_lower, "__getitem__") else irr_ci_lower[params.index.get_loc(name)]),
            "irr_ci_upper": float(irr_ci_upper[name] if hasattr(irr_ci_upper, "__getitem__") else irr_ci_upper[params.index.get_loc(name)]),
        }
    return out


def fit_negative_binomial(formula, data):
    # Use statsmodels discrete NegativeBinomial (NB2)
    y, X = dmatrices(formula, data, return_type="dataframe")
    # Ensure endog is a 1-d array
    endog = np.asarray(y).ravel()
    mod = sm.NegativeBinomial(endog, X)
    res = mod.fit(disp=False)
    return res, X.columns.tolist()


def summarise_nb_results(res, exog_names):
    # Robustly extract parameters and uncertainty; fall back to NaNs if not available
    params = res.params
    # bse can be unavailable if Hessian inversion fails
    try:
        bse = res.bse
    except Exception:
        bse = pd.Series(np.nan, index=params.index)
    try:
        conf = res.conf_int()
    except Exception:
        conf = pd.DataFrame({0: np.nan, 1: np.nan}, index=params.index)
    try:
        pvals = res.pvalues
    except Exception:
        pvals = pd.Series(np.nan, index=params.index)
    irr = np.exp(params)
    # Handle conf whether DataFrame or ndarray
    if hasattr(conf, "iloc"):
        ci_lower = conf.iloc[:, 0]
        ci_upper = conf.iloc[:, 1]
    else:
        ci_lower = conf[:, 0]
        ci_upper = conf[:, 1]
    irr_ci_lower = np.exp(ci_lower)
    irr_ci_upper = np.exp(ci_upper)
    irr_se = irr * bse
    out = {}
    for name in params.index:
        out[name] = {
            "coef": float(params[name]),
            "se": float(bse[name] if name in bse.index else np.nan),
            "p_value": float(pvals[name] if name in pvals.index else np.nan),
            "ci_lower": float(ci_lower[name] if hasattr(ci_lower, "__getitem__") else ci_lower[list(params.index).index(name)]),
            "ci_upper": float(ci_upper[name] if hasattr(ci_upper, "__getitem__") else ci_upper[list(params.index).index(name)]),
            "irr": float(irr[name]),
            "irr_se_delta": float(irr_se[name] if name in bse.index else np.nan),
            "irr_ci_lower": float(irr_ci_lower[name] if hasattr(irr_ci_lower, "__getitem__") else irr_ci_lower[list(params.index).index(name)]),
            "irr_ci_upper": float(irr_ci_upper[name] if hasattr(irr_ci_upper, "__getitem__") else irr_ci_upper[list(params.index).index(name)]),
        }
    return out


def main():
    df = pd.read_stata(DATA_PATH)
    # Ensure expected columns exist
    required_cols = [
        "complaints_2008", "first_A", "firm_age", "emp_size", "num_names_2008",
        "multiple_names", "chicago", "ad_spending", "on_google"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_constructed_vars(df)

    results = {}

    # Model 0: Poisson first_A only
    formula0 = "complaints_2008 ~ first_A_ec"
    m0, m0r = robust_glm_poisson(formula0, df)
    results["poisson_model0_firstA_only"] = summarise_glm_results(m0r)

    # Model 1: Poisson controls only (no first_A)
    formula1 = "complaints_2008 ~ firm_ageC + emp_sizeC + num_names_2008C + ad_spendingC"
    m1, m1r = robust_glm_poisson(formula1, df)
    results["poisson_model1_controls_only"] = summarise_glm_results(m1r)

    # Model 2: Poisson with some controls + first_A
    formula2 = "complaints_2008 ~ first_A_ec + firm_ageC + emp_sizeC + num_names_2008C + ad_spendingC"
    m2, m2r = robust_glm_poisson(formula2, df)
    results["poisson_model2_with_firstA_controls"] = summarise_glm_results(m2r)

    # Model 3: Poisson with all controls
    formula3 = (
        "complaints_2008 ~ first_A_ec + firm_ageC + emp_sizeC + num_names_2008C + "
        "multiple_names_ec + chicago_ec + ad_spendingC + on_google_ec"
    )
    m3, m3r = robust_glm_poisson(formula3, df)
    results["poisson_model3_all_controls"] = summarise_glm_results(m3r)

    # Negative Binomial with same specification as model 3
    nb_res, exog_names = fit_negative_binomial(formula3, df)
    results["neg_binom_model_all_controls"] = summarise_nb_results(nb_res, exog_names)

    # Save JSON results
    with open(OUT_JSON, "w") as f:
        json.dump({
            "model_summaries": results,
            "notes": {
                "data_path": DATA_PATH,
                "poisson_cov_type": "HC0",
                "nb_model": "statsmodels.discrete.count_model.NegativeBinomial (NB2)",
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()
