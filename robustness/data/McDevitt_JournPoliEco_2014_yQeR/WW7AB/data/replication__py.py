import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
from statsmodels.discrete.count_model import ZeroInflatedPoisson

DATA_PATH = "/app/data/final_data.dta"
OUT_PATH_JSON = "/app/data/task1_results.json"


def load_data(path=DATA_PATH):
    df = pd.read_stata(path, convert_categoricals=False)
    return df.copy()


def ttest_by_firstA(df):
    a1 = df.loc[df["first_A"] == 1, "complaints_2008"].astype(float)
    a0 = df.loc[df["first_A"] == 0, "complaints_2008"].astype(float)
    # Stata's default two-sample t test assumes equal variances. We report both.
    t_eq, p_eq = stats.ttest_ind(a1, a0, equal_var=True, nan_policy="omit")
    t_uneq, p_uneq = stats.ttest_ind(a1, a0, equal_var=False, nan_policy="omit")
    return {
        "n_A1": int(a1.notna().sum()),
        "n_A0": int(a0.notna().sum()),
        "mean_A1": float(np.nanmean(a1)),
        "mean_A0": float(np.nanmean(a0)),
        "t_equal_var": float(t_eq),
        "p_equal_var": float(p_eq),
        "t_unequal_var": float(t_uneq),
        "p_unequal_var": float(p_uneq)
    }


def chi2_binary_by_firstA(df):
    df = df.copy()
    df["complaints_2008_bin"] = (df["complaints_2008"].astype(float) > 0).astype(int)
    tab = pd.crosstab(df["complaints_2008_bin"], df["first_A"]).astype(int)
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    return {
        "table": tab.values.tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof)
    }


def fit_logit_models(df):
    df = df.copy()
    df["complaints_2008_bin"] = (df["complaints_2008"].astype(float) > 0).astype(int)
    res = {}
    # Simple
    m1 = smf.logit("complaints_2008_bin ~ first_A", data=df).fit(disp=False)
    b = m1.params["first_A"]
    se = m1.bse["first_A"]
    z = b / se
    p = m1.pvalues["first_A"]
    res["logit_simple"] = {"coef_first_A": float(b), "se": float(se), "z": float(z), "p": float(p)}
    # With controls
    formula = "complaints_2008_bin ~ first_A + ad_spend_k + firm_age + chicago + emp_size + multiple_names"
    m2 = smf.logit(formula, data=df).fit(disp=False)
    b = m2.params["first_A"]
    se = m2.bse["first_A"]
    z = b / se
    p = m2.pvalues["first_A"]
    res["logit_full"] = {"coef_first_A": float(b), "se": float(se), "z": float(z), "p": float(p)}
    return res


def fit_poisson_models(df):
    df = df.copy()
    res = {}
    # Simple
    m1 = smf.glm("complaints_2008 ~ first_A", data=df, family=sm.families.Poisson()).fit()
    pearson_chi2 = m1.pearson_chi2
    df_resid = m1.df_resid
    od = pearson_chi2 / df_resid if df_resid > 0 else np.nan
    res["poisson_simple"] = {
        "coef_first_A": float(m1.params.get("first_A", np.nan)),
        "se": float(m1.bse.get("first_A", np.nan)),
        "z": float(m1.tvalues.get("first_A", np.nan)),
        "p": float(m1.pvalues.get("first_A", np.nan)),
        "overdispersion_pearson_chi2_by_df": float(od)
    }
    # With controls
    formula = "complaints_2008 ~ first_A + ad_spend_k + firm_age + chicago + emp_size + multiple_names"
    m2 = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
    pearson_chi2 = m2.pearson_chi2
    df_resid = m2.df_resid
    od = pearson_chi2 / df_resid if df_resid > 0 else np.nan
    res["poisson_full"] = {
        "coef_first_A": float(m2.params.get("first_A", np.nan)),
        "se": float(m2.bse.get("first_A", np.nan)),
        "z": float(m2.tvalues.get("first_A", np.nan)),
        "p": float(m2.pvalues.get("first_A", np.nan)),
        "overdispersion_pearson_chi2_by_df": float(od)
    }
    return res


def _margins_at_means_count(model, cols, df_base, first_A_values=(0, 1), is_zip=False, zip_infl_cols=None):
    # Build explicit exog matrices with a constant column to match training design
    # cols: columns used in the count equation (excluding constant)
    # zip_infl_cols: columns used in the inflation equation (excluding constant)
    means = df_base[cols].mean(numeric_only=True)
    out = {}
    for v in first_A_values:
        # Build count exog row with explicit constant
        ex = means.copy()
        ex["first_A"] = v
        ex_count = pd.DataFrame([[1.0] + [ex[c] for c in cols]], columns=["const"] + cols)
        if not is_zip:
            mu = float(model.predict(exog=ex_count)[0])
        else:
            # Build inflation exog row with explicit constant
            if zip_infl_cols is None:
                infl_means = pd.Series({}, dtype=float)
                infl_means["first_A"] = v
                infl_cols = ["first_A"]
            else:
                infl_means = df_base[zip_infl_cols].mean(numeric_only=True)
                infl_means["first_A"] = v
                infl_cols = list(zip_infl_cols)
            ex_infl = pd.DataFrame([[1.0] + [infl_means[c] for c in infl_cols]], columns=["const"] + infl_cols)
            mu = float(model.predict(exog=ex_count, exog_infl=ex_infl, which="mean")[0])
        out[f"first_A_{v}"] = mu
    return out


def fit_zip_models(df):
    df = df.copy()
    res = {}
    # Simple: count ~ first_A; inflate ~ first_A
    y = df["complaints_2008"].values
    cols_simple = ["first_A"]
    X = sm.add_constant(df[cols_simple])
    X_infl = sm.add_constant(df[cols_simple])
    zip1 = ZeroInflatedPoisson(y, X, exog_infl=X_infl, inflation="logit").fit(disp=False, maxiter=200)
    params = zip1.params
    b_count = float(params.get("first_A", np.nan))
    b_infl = float(params.get("inflate_first_A", np.nan))
    se = zip1.bse
    z = zip1.tvalues
    res["zip_simple"] = {
        "coef_count_first_A": b_count,
        "se_count_first_A": float(se.get("first_A", np.nan)),
        "z_count_first_A": float(z.get("first_A", np.nan)),
        "coef_infl_first_A": b_infl,
        "se_infl_first_A": float(se.get("inflate_first_A", np.nan)),
        "z_infl_first_A": float(z.get("inflate_first_A", np.nan)),
        "margins_atmeans": _margins_at_means_count(zip1, cols=cols_simple, df_base=df, is_zip=True, zip_infl_cols=cols_simple)
    }
    # Full: count ~ first_A + controls; inflate ~ same set
    cols = ["first_A", "ad_spend_k", "firm_age", "chicago", "emp_size", "multiple_names"]
    X = sm.add_constant(df[cols])
    X_infl = sm.add_constant(df[cols])
    zip2 = ZeroInflatedPoisson(y, X, exog_infl=X_infl, inflation="logit").fit(disp=False, maxiter=400)
    params = zip2.params
    res["zip_full"] = {
        "coef_count_first_A": float(params.get("first_A", np.nan)),
        "se_count_first_A": float(zip2.bse.get("first_A", np.nan)),
        "z_count_first_A": float(zip2.tvalues.get("first_A", np.nan)),
        "coef_infl_first_A": float(params.get("inflate_first_A", np.nan)),
        "se_infl_first_A": float(zip2.bse.get("inflate_first_A", np.nan)),
        "z_infl_first_A": float(zip2.tvalues.get("inflate_first_A", np.nan)),
        "margins_atmeans": _margins_at_means_count(zip2, cols=cols, df_base=df, is_zip=True, zip_infl_cols=cols)
    }
    return res


def fit_nb_models(df):
    df = df.copy()
    res = {}
    y = df["complaints_2008"].values
    # Simple
    cols = ["first_A"]
    X = sm.add_constant(df[cols])
    nb1 = NBDiscrete(y, X).fit(disp=False, maxiter=200)
    b = nb1.params["first_A"]
    se = nb1.bse["first_A"]
    z = b / se
    p = nb1.pvalues["first_A"]
    res["nb_simple"] = {
        "coef_first_A": float(b), "se": float(se), "z": float(z), "p": float(p),
        "alpha": float(np.exp(nb1.params[-1])) if hasattr(nb1, "params") else float("nan"),
        "margins_atmeans": _margins_at_means_count(nb1, cols=cols, df_base=df)
    }
    # Full
    cols = ["first_A", "ad_spend_k", "firm_age", "chicago", "emp_size", "multiple_names"]
    X = sm.add_constant(df[cols])
    nb2 = NBDiscrete(y, X).fit(disp=False, maxiter=400)
    b = nb2.params["first_A"]
    se = nb2.bse["first_A"]
    z = b / se
    p = nb2.pvalues["first_A"]
    res["nb_full"] = {
        "coef_first_A": float(b), "se": float(se), "z": float(z), "p": float(p),
        "alpha": float(np.exp(nb2.params[-1])) if hasattr(nb2, "params") else float("nan"),
        "margins_atmeans": _margins_at_means_count(nb2, cols=cols, df_base=df)
    }
    return res


def main():
    df = load_data()
    results = {}
    results["ttest_complaints_by_first_A"] = ttest_by_firstA(df)
    results["chi2_binary_complaints_by_first_A"] = chi2_binary_by_firstA(df)
    results.update(fit_logit_models(df))
    results.update(fit_poisson_models(df))
    results.update(fit_zip_models(df))
    results.update(fit_nb_models(df))
    # Write results
    with open(OUT_PATH_JSON, "w") as f:
        json.dump(results, f, indent=2)
    # Also print a short summary to stdout
    print(json.dumps({
        "nb_simple_z_first_A": results.get("nb_simple", {}).get("z"),
        "nb_full_z_first_A": results.get("nb_full", {}).get("z"),
        "zip_simple_margins": results.get("zip_simple", {}).get("margins_atmeans"),
        "zip_full_margins": results.get("zip_full", {}).get("margins_atmeans")
    }, indent=2))


if __name__ == "__main__":
    main()
