import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Paths
CANDIDATE_PATHS = [
    '/app/data/AlTammemi_Survey_deidentify.csv',
    'replication_data/AlTammemi_Survey_deidentify.csv',
    'replication_data/AlTammemi_Survey_example.csv'
]

def find_csv():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('Could not find AlTammemi_Survey_deidentify.csv in expected paths.')


def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


def main():
    csv_path = find_csv()
    print('Using dataset:', csv_path)
    df = pd.read_csv(csv_path)
    # Ensure Kessler items exist
    k_items = [f'Kessler_{i}' for i in range(1,11)]
    for k in k_items:
        if k not in df.columns:
            print(f'Missing expected column: {k}', file=sys.stderr)
            # try lowercase
            if k.lower() in df.columns:
                df[k] = df[k.lower()]
            else:
                raise KeyError(f'Missing K10 item column: {k}')
    # Cast to numeric
    for k in k_items:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    # Compute K10 total only for rows with all items
    df['K10_total'] = df[k_items].sum(axis=1, min_count=10)
    # Identify rows with any missing Kessler item
    missing_k = df[k_items].isnull().any(axis=1)
    print('Total rows:', len(df))
    print('Rows with any missing K10 item:', missing_k.sum())
    # Define categories
    def k10_category(score):
        if pd.isnull(score):
            return None
        if score <= 19:
            return 'none'
        elif 20 <= score <= 24:
            return 'mild'
        elif 25 <= score <= 29:
            return 'moderate'
        else:
            return 'severe'
    df['K10_category'] = df['K10_total'].apply(k10_category)
    df['severe'] = df['K10_total'] >= 30
    # Predictor: online_learning_1
    if 'online_learning_1' not in df.columns:
        raise KeyError('Missing primary predictor column: online_learning_1')
    df['online_learning_1'] = pd.to_numeric(df['online_learning_1'], errors='coerce')
    df['no_motivation'] = df['online_learning_1'] == 1
    # Covariates: demographic_1..4
    covs = []
    # Age from demographic_1: if looks like birthyear, convert to age using 2020
    if 'demographic_1' in df.columns:
        df['demographic_1'] = pd.to_numeric(df['demographic_1'], errors='coerce')
        # Heuristic: values > 1900 treated as birthyear
        byear_mask = df['demographic_1'].between(1900, 2025)
        if byear_mask.sum() > 0:
            df['age'] = np.where(byear_mask, 2020 - df['demographic_1'], df['demographic_1'])
        else:
            df['age'] = df['demographic_1']
        covs.append('age')
    # One-hot demographic_2..4
    for i in [2,3,4]:
        col = f'demographic_{i}'
        if col in df.columns:
            # treat as categorical
            df[col] = df[col].fillna('missing')
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
            # append dummies to df
            for c in dummies.columns:
                df[c] = dummies[c]
                covs.append(c)
    # Prepare regression dataframe: drop rows missing K10_total or online_learning_1
    reg_df = df[~df['K10_total'].isnull() & ~df['online_learning_1'].isnull()].copy()
    print('Rows after dropping missing K10_total or predictor:', len(reg_df))
    # Define outcome and predictors
    y = reg_df['severe'].astype(int)
    X_cols = ['no_motivation'] + covs
    X = reg_df[X_cols].copy()
    # Ensure no_motivation numeric
    X['no_motivation'] = X['no_motivation'].astype(int)

    # Add constant    # Add constant
    X = sm.add_constant(X)

    # Ensure all X columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Ensure y is numeric
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with any missing values in X or y (listwise deletion as planned)
    good_mask = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[good_mask]
    y = y.loc[good_mask]

    if len(y) == 0:
        raise ValueError('No observations available after dropping missing covariates.')

    results = None

    # Ensure X and y are float dtype for statsmodels
    try:
        X = X.astype(float)
        y = y.astype(float)
    except Exception as e:
        print('Casting X/y to float failed:', str(e), file=sys.stderr)
        print('X dtypes:', X.dtypes.to_dict(), file=sys.stderr)
        print('X sample head:\n', X.head().to_string(), file=sys.stderr)
        print('y sample head:\n', y.head().to_string(), file=sys.stderr)
        raise

    try:
        model = sm.Logit(y, X)
        results = model.fit(disp=False)
        method_used = 'Logit'
    except Exception as e:
        print('Logit failed, falling back to GLM Binomial. Error:', str(e))
        model = sm.GLM(y, X, family=sm.families.Binomial())
        results = model.fit()
        method_used = 'GLM-Binomial'

    coef = results.params.get('no_motivation', np.nan)
    se = results.bse.get('no_motivation', np.nan)
    pval = results.pvalues.get('no_motivation', np.nan) if hasattr(results, 'pvalues') else np.nan
    conf = results.conf_int().loc['no_motivation'].tolist() if 'no_motivation' in results.params.index else [np.nan, np.nan]
    or_val = float(np.exp(coef)) if not pd.isnull(coef) else np.nan
    or_ci = [float(np.exp(conf[0])), float(np.exp(conf[1]))] if not any(pd.isnull(conf)) else [np.nan, np.nan]

    out = {
        'n_total': int(len(df)),
        'n_used_in_regression': int(len(reg_df)),
        'k10_mean': float(df['K10_total'].mean(skipna=True)),
        'severe_prevalence': float(df['severe'].mean(skipna=True)),
        'model_method': method_used,
        'coefficient_no_motivation': float(coef) if not pd.isnull(coef) else None,
        'se_coefficient_no_motivation': float(se) if not pd.isnull(se) else None,
        'pvalue_no_motivation': float(pval) if not pd.isnull(pval) else None,
        'odds_ratio_no_motivation': or_val,
        'odds_ratio_ci95': or_ci,
        'confint_logodds': conf,
        'X_columns': X_cols
    }

    out_path = '/app/data/replication_results_AlTammemi.json'
    try:
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print('Wrote results to', out_path)
    except Exception as e:
        # fallback to local path
        with open('replication_results_AlTammemi.json', 'w') as f:
            json.dump(out, f, indent=2)
        print('Wrote results to local replication_results_AlTammemi.json')

    # Print a brief summary
    print('--- Regression summary (no_motivation) ---')
    print('Method used:', method_used)
    print('Coefficient (log-odds):', coef)
    print('SE:', se)
    print('p-value:', pval)
    print('Odds Ratio:', or_val)
    print('OR 95% CI:', or_ci)
    print('Rows used in regression:', len(reg_df))

if __name__ == '__main__':
    main()
