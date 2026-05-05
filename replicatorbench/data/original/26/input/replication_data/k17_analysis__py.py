import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from scipy import stats

PREFERRED_PATH = "/app/data/replication_data/k17_processed_data.csv"
FALLBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k17_processed_data.csv")

def load_data():
    for p in [PREFERRED_PATH, FALLBACK_PATH]:
        if os.path.exists(p):
            print(f"Loading processed data from: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"Processed data not found. Checked: {[PREFERRED_PATH, FALLBACK_PATH]}")


def zscore_series(s):
    return (s - s.mean()) / s.std(ddof=0)


def standardize_df(df, cols):
    out = df.copy()
    for c in cols:
        out[c + "_z"] = zscore_series(out[c].astype(float))
    return out


def main():
    df = load_data()
    # Ensure required cols exist
    required = ['T3_panas_joviality','age','children','work_hours','work_days','T1_panas_joviality','req_control','req_mastery','req_relax','req_detach','hassles','gender','birthyear','T3_panas_self_assurance','T3_panas_serenity','T3_panas_fear','T3_panas_sadness']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Warning: missing columns in data:", missing)

    # Create z-scored versions for scale() usage in R code
    zcols = ['T3_panas_joviality','age','children','work_hours','work_days','T1_panas_joviality','req_control','req_mastery','req_relax','req_detach','hassles']
    df = standardize_df(df, [c for c in zcols if c in df.columns])

    # Model 1
    formula1 = 'T3_panas_joviality_z ~ age_z + gender + children_z + work_hours_z + work_days_z'
    m1 = smf.ols(formula1, data=df).fit()
    print('\nModel 1 summary:')
    print(m1.summary())

    # Model 2
    formula2 = formula1 + ' + T1_panas_joviality_z'
    m2 = smf.ols(formula2, data=df).fit()
    print('\nModel 2 summary:')
    print(m2.summary())

    # Model 3 (focal)
    formula3 = formula2 + ' + req_control_z + req_mastery_z + req_relax_z + req_detach_z + hassles_z'
    m3 = smf.ols(formula3, data=df).fit()
    print('\nModel 3 (focal) summary:')
    print(m3.summary())

    # Model 3a (uncentered predictors)
    formula3a = 'T3_panas_joviality ~ age + gender + children + work_hours + work_days + T1_panas_joviality + req_control + req_mastery + req_relax + req_detach + hassles'
    m3a = smf.ols(formula3a, data=df).fit()
    print('\nModel 3a (uncentered) summary:')
    print(m3a.summary())

    # R-squared and incremental
    print('\nR-squared M1:', m1.rsquared)
    print('R-squared M2:', m2.rsquared)
    print('Incremental R2 (M2 - M1):', m2.rsquared - m1.rsquared)
    print('R-squared M3:', m3.rsquared)
    print('Incremental R2 (M3 - M2):', m3.rsquared - m2.rsquared)

    # ANOVA style comparison using F-test
    print('\nANOVA comparison M1 vs M2 (F-test):')
    try:
        print(m2.compare_f_test(m1))
    except Exception as e:
        print('Could not run compare_f_test:', e)

    print('\nANOVA comparison M2 vs M3 (F-test):')
    try:
        print(m3.compare_f_test(m2))
    except Exception as e:
        print('Could not run compare_f_test:', e)

    # Exploratory models (self-assurance, serenity, fear, sadness)
    if 'T3_panas_self_assurance' in df.columns:
        f4 = 'T3_panas_self_assurance ~ birthyear + gender + children + work_hours + work_days + T1_panas_self_assurance + req_control + req_mastery + req_relax + req_detach + hassles'
        m4 = smf.ols(f4, data=df).fit()
        print('\nModel 4 (self_assurance) summary:')
        print(m4.summary())

    if 'T3_panas_serenity' in df.columns:
        f5 = 'T3_panas_serenity ~ birthyear + gender + children + work_hours + work_days + T1_panas_serenity + req_control + req_mastery + req_relax + req_detach + hassles'
        m5 = smf.ols(f5, data=df).fit()
        print('\nModel 5 (serenity) summary:')
        print(m5.summary())

    if 'T3_panas_fear' in df.columns:
        f6 = 'T3_panas_fear ~ birthyear + gender + children + work_hours + work_days + T1_panas_fear + req_control + req_mastery + req_relax + req_detach + hassles'
        m6 = smf.ols(f6, data=df).fit()
        print('\nModel 6 (fear) summary:')
        print(m6.summary())

    if 'T3_panas_sadness' in df.columns:
        f7 = 'T3_panas_sadness ~ birthyear + gender + children + work_hours + work_days + T1_panas_sadness + req_control + req_mastery + req_relax + req_detach + hassles'
        m7 = smf.ols(f7, data=df).fit()
        print('\nModel 7 (sadness) summary:')
        print(m7.summary())

if __name__ == '__main__':
    main()
