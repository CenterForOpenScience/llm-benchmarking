import pandas as pd
import statsmodels.api as sm
import os

# Paths (use /app/data as base in execution environment)
DATA_PATH = '/app/data/original/14/0205_python_gpt5-mini/replication_data/Estimation Data - Pitts (126zz).csv'
OUTPUT_DIR = '/app/data/original/14/0205_python_gpt5-mini/replication_data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'replication_results.csv')

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    # Drop rows with any missing values (to mirror na.omit in the original R script)
    df = df.dropna()

    # Define outcome and predictors to match the R glm.cluster formula
    y = df['LeavingAgency']
    X_vars = ['JobSat','Over40','NonMinority','SatPay','SatAdvan','PerfCul','Empowerment','RelSup','Relcow','Over40xSatAdvan']
    X = df[X_vars]
    X = sm.add_constant(X)

    # Fit logistic regression (Logit)
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
    except Exception as e:
        # If perfect separation or other convergence issues, try a penalized fit
        result = model.fit_regularized(method='l1', disp=False)

    # Obtain clustered robust standard errors by Agency
    # statsmodels requires groups as an array
    try:
        clustered = result.get_robustcov_results(cov_type='cluster', groups=df['Agency'])
    except Exception:
        # If clustering fails (e.g., too many groups), fall back to robust (HC3)
        clustered = result.get_robustcov_results(cov_type='HC3')

    # Prepare output: coefficients, robust SE, z, p
    summary_df = pd.DataFrame({
        'coef': clustered.params,
        'std_err': clustered.bse,
        'z': clustered.tvalues,
        'p_value': clustered.pvalues
    })

    summary_df.to_csv(OUTPUT_FILE)
    print('Saved replication results to', OUTPUT_FILE)

if __name__ == '__main__':
    main()
