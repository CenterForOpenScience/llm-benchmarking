import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# All IO must use /app/data per run policy
DATA_PATH = "/app/data/final_data.dta"
OUT_JSON = "/app/data/results_task2.json"


def effect_code(series):
    return series.map({0: -0.5, 1: 0.5}).astype(float)


def add_constructed_vars(df):
    df = df.copy()
    df["ad_spendingC"] = df["ad_spending"] - df["ad_spending"].mean()
    df["firm_ageC"] = df["firm_age"] - df["firm_age"].mean()
    df["log_emp_size"] = np.log(df["emp_size"].replace(0, np.nan))
    df["log_emp_size"] = df["log_emp_size"].replace([-np.inf, np.inf], np.nan)
    df["emp_sizeC"] = df["log_emp_size"] - df["log_emp_size"].mean()
    df["num_names_2008C"] = df["num_names_2008"] - df["num_names_2008"].mean()

    for col in ["first_A", "multiple_names", "chicago", "on_google"]:
        if col in df.columns:
            df[f"{col}_ec"] = effect_code(df[col].astype(int))
    return df


def robust_glm_poisson(formula, data):
    model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit(cov_type="HC0")
    return model


def summarise_glm_results(rob):
    params = rob.params
    bse = rob.bse
    pvals = rob.pvalues
    conf = rob.conf_int(alpha=0.05)
    irr = np.exp(params)
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


def main():
    df = pd.read_stata(DATA_PATH)
    df = add_constructed_vars(df)

    results = {}

    # Poisson first_A only
    formula0 = "complaints_2008 ~ first_A_ec"
    m0r = robust_glm_poisson(formula0, df)
    results["poisson_model0_firstA_only"] = summarise_glm_results(m0r)

    # Poisson controls only (no first_A)
    formula1 = "complaints_2008 ~ firm_ageC + emp_sizeC + num_names_2008C + ad_spendingC"
    m1r = robust_glm_poisson(formula1, df)
    results["poisson_model1_controls_only"] = summarise_glm_results(m1r)

    # Poisson with some controls + first_A
    formula2 = "complaints_2008 ~ first_A_ec + firm_ageC + emp_sizeC + num_names_2008C + ad_spendingC"
    m2r = robust_glm_poisson(formula2, df)
    results["poisson_model2_with_firstA_controls"] = summarise_glm_results(m2r)

    # Poisson with all controls
    formula3 = (
        "complaints_2008 ~ first_A_ec + firm_ageC + emp_sizeC + num_names_2008C + "
        "multiple_names_ec + chicago_ec + ad_spendingC + on_google_ec"
    )
    m3r = robust_glm_poisson(formula3, df)
    results["poisson_model3_all_controls"] = summarise_glm_results(m3r)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "model_summaries": results,
            "notes": {
                "data_path": DATA_PATH,
                "poisson_cov_type": "HC0"
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()
