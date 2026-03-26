import pandas as pd, os, statsmodels.api as sm, sys, json


def main():
    # Path relative to /app/data, adjust accordingly
    data_path = os.path.join('/app/data', 'original/36/input/replication_data/Popper Replication Data Files/Popper Data for Correlations.csv')
    if not os.path.exists(data_path):
        # attempt to locate file anywhere under /app/data
        for root, dirs, files in os.walk('/app/data'):
            if 'Popper Data for Correlations.csv' in files:
                data_path = os.path.join(root, 'Popper Data for Correlations.csv')
                break
        if not os.path.exists(data_path):
            print(f"Data file not found. Searched starting at /app/data. Exiting.")
            sys.exit(1)
    df = pd.read_csv(data_path)
    # Rename columns for ease (strip whitespace)
    df = df.rename(columns=lambda x: x.strip())
    # select variables
    y = df['Leader_Exp_Total']
    X = df['Oppenness_Total']
    # Simple correlation
    corr = y.corr(X)
    # Simple linear regression
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, missing='drop').fit()

    results = {
        'pearson_r': corr,
        'regression_coef': model.params['Oppenness_Total'],
        'p_value': model.pvalues['Oppenness_Total'],
        'n_obs': int(model.nobs)
    }
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
