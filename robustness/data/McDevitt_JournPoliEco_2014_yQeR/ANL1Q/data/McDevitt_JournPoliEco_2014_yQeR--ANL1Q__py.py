import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# All IO must use /app/data
DATA_PATH = "/app/data/final_data.dta"

def main():
    # Load data
    df = pd.read_stata(DATA_PATH)

    # Ensure key variables are present
    required_cols = [
        'complaints_2008','first_A','multiple_names','on_google',
        'ad_spend_k','firm_age','chicago','emp_size'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Dichotomized analysis of complaints (robust against outliers)
    df['complaints_binary'] = np.where(df['complaints_2008'] == 0, 0, 1)
    ct = pd.crosstab(df['first_A'], df['complaints_binary'])
    chi2, p_chi2, dof, exp = stats.chi2_contingency(ct.values)
    print("RESULT|chi_square_2x2|statistic|{:.6f}".format(chi2))
    print("RESULT|chi_square_2x2|p_value|{:.6g}".format(p_chi2))
    # Percentages
    try:
        # first_A assumed 0/1
        n_nonA = int(ct.loc[0, :].sum()) if 0 in ct.index else int(ct.iloc[0, :].sum())
        n_nonA_pos = int(ct.loc[0, 1]) if (0 in ct.index and 1 in ct.columns) else None
        n_A = int(ct.loc[1, :].sum()) if 1 in ct.index else None
        n_A_pos = int(ct.loc[1, 1]) if (1 in ct.index and 1 in ct.columns) else None
        if n_nonA and n_nonA_pos is not None:
            print("RESULT|percent_at_least_one_complaint_nonA|value|{:.6f}".format(n_nonA_pos / n_nonA))
        if n_A and n_A_pos is not None:
            print("RESULT|percent_at_least_one_complaint_A|value|{:.6f}".format(n_A_pos / n_A))
    except Exception as e:
        print(f"WARN|percentage_calc_failed|{e}")

    # Poisson GLM: complaints_2008 ~ first_A
    pois = smf.glm(formula='complaints_2008 ~ first_A', data=df, family=sm.families.Poisson()).fit()
    print("RESULT|poisson_glm|coef_first_A|{:.6f}".format(pois.params.get('first_A', np.nan)))
    print("RESULT|poisson_glm|z_first_A|{:.6f}".format(pois.tvalues.get('first_A', np.nan)))
    print("RESULT|poisson_glm|p_first_A|{:.6g}".format(pois.pvalues.get('first_A', np.nan)))
    # Goodness of fit (p<.05 indicates bad fit)
    dev = pois.deviance
    df_resid = pois.df_resid
    gof_p = 1 - stats.chi2.cdf(dev, df_resid)
    print("RESULT|poisson_glm|gof_p|{:.6g}".format(gof_p))

    # Negative Binomial GLM: complaints_2008 ~ first_A
    nb = smf.glm(formula='complaints_2008 ~ first_A', data=df, family=sm.families.NegativeBinomial()).fit()
    print("RESULT|negbin_glm|coef_first_A|{:.6f}".format(nb.params.get('first_A', np.nan)))
    print("RESULT|negbin_glm|z_first_A|{:.6f}".format(nb.tvalues.get('first_A', np.nan)))
    print("RESULT|negbin_glm|p_first_A|{:.6g}".format(nb.pvalues.get('first_A', np.nan)))
    # Goodness of fit (proxy): deviance test
    dev_nb = nb.deviance
    df_resid_nb = nb.df_resid
    gof_p_nb = 1 - stats.chi2.cdf(dev_nb, df_resid_nb)
    print("RESULT|negbin_glm|gof_p|{:.6g}".format(gof_p_nb))

    # Model comparison: Poisson vs NB (LR test)
    try:
        ll_p = pois.llf
        ll_nb = nb.llf
        lr_stat = 2 * (ll_nb - ll_p)
        lr_p = stats.chi2.sf(lr_stat, df=1)
        print("RESULT|model_comparison|LR_NB_vs_Poisson_stat|{:.6f}".format(lr_stat))
        print("RESULT|model_comparison|LR_NB_vs_Poisson_p|{:.6g}".format(lr_p))
    except Exception as e:
        print(f"WARN|model_comparison_failed|{e}")

    # Zero-inflated models (optional if available)
    try:
        from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
        # exogenous with intercept
        exog = sm.add_constant(df[['first_A']])
        # Inflation with intercept only
        zip_model = ZeroInflatedPoisson(df['complaints_2008'], exog, exog_infl=np.ones((len(df), 1)), inflation='logit')
        zip_res = zip_model.fit(method='bfgs', disp=0)
        print("RESULT|zip|coef_first_A|{:.6f}".format(zip_res.params[exog.columns.get_loc('first_A')]))
        print("RESULT|zip|p_first_A|{:.6g}".format(zip_res.pvalues[exog.columns.get_loc('first_A')]))

        zinb_model = ZeroInflatedNegativeBinomialP(df['complaints_2008'], exog, exog_infl=np.ones((len(df), 1)), inflation='logit')
        zinb_res = zinb_model.fit(method='bfgs', disp=0)
        print("RESULT|zinb|coef_first_A|{:.6f}".format(zinb_res.params[exog.columns.get_loc('first_A')]))
        print("RESULT|zinb|p_first_A|{:.6g}".format(zinb_res.pvalues[exog.columns.get_loc('first_A')]))
    except Exception as e:
        print(f"WARN|zero_inflated_models_skipped_or_failed|{e}")

    # NB with controls
    formula_controls = 'complaints_2008 ~ first_A + multiple_names + on_google + ad_spend_k + firm_age + chicago + emp_size'
    nb_c = smf.glm(formula=formula_controls, data=df, family=sm.families.NegativeBinomial()).fit()
    print("RESULT|negbin_glm_controls|coef_first_A|{:.6f}".format(nb_c.params.get('first_A', np.nan)))
    print("RESULT|negbin_glm_controls|z_first_A|{:.6f}".format(nb_c.tvalues.get('first_A', np.nan)))
    print("RESULT|negbin_glm_controls|p_first_A|{:.6g}".format(nb_c.pvalues.get('first_A', np.nan)))

    # t-test replication of Table 2: complaints_2008 by first_A
    grp_A = df.loc[df['first_A'] == 1, 'complaints_2008']
    grp_nonA = df.loc[df['first_A'] == 0, 'complaints_2008']
    t_stat, p_t = stats.ttest_ind(grp_A, grp_nonA, equal_var=False, nan_policy='omit')
    print("RESULT|t_test_complaints_by_first_A|t_stat|{:.6f}".format(t_stat))
    print("RESULT|t_test_complaints_by_first_A|p_value|{:.6g}".format(p_t))

if __name__ == "__main__":
    main()
