import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np

# Robust import for statsmodels with fallback installation and sys.path adjustments

def ensure_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
        return sm
    except Exception:
        pass

    # Try adding common site-packages to sys.path
    try:
        import site
        candidates = []
        try:
            candidates.append(site.getusersitepackages())
        except Exception:
            pass
        if hasattr(site, 'USER_SITE'):
            candidates.append(site.USER_SITE)
        candidates.extend([
            '/usr/local/lib/python3.10/site-packages',
            os.path.expanduser('~/.local/lib/python3.10/site-packages'),
        ])
        for p in candidates:
            if p and p not in sys.path and os.path.isdir(p):
                sys.path.append(p)
        try:
            import statsmodels.api as sm  # noqa: F401
            return sm
        except Exception:
            pass
    except Exception:
        pass

    # Attempt runtime installation to user site
    pkgs = [
        'statsmodels==0.14.0',
        'pandas==1.5.3',
        'numpy==1.24.4',
        'scipy==1.10.1',
    ]
    print('statsmodels not found; attempting runtime installation ...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user'] + pkgs)
    except Exception as e:
        raise ImportError(f'Failed to install required packages: {e}')

    # Re-add user site to sys.path and retry import
    try:
        import site
        for p in [getattr(site, 'USER_SITE', None), os.path.expanduser('~/.local/lib/python3.10/site-packages')]:
            if p and p not in sys.path and os.path.isdir(p):
                sys.path.append(p)
    except Exception:
        pass

    try:
        import statsmodels.api as sm  # noqa: F401
        print('Runtime installation succeeded.')
        return sm
    except Exception as e:
        raise ImportError(f'Failed to import or install required packages: {e}')


# IO paths to search (must include /app/data per policy but also allow local fallback)
SEARCH_INPUT_PATHS = [
    "/app/data/replication_data/k17_processed_data.csv",
    "/app/data/k17_processed_data.csv",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "k17_processed_data.csv"),
]
RESULTS_PATH = "/app/data/k17_results.json"


def load_processed_data():
    for path in SEARCH_INPUT_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, path
    raise FileNotFoundError(
        f"Processed dataset not found. Searched: {SEARCH_INPUT_PATHS}"
    )


def fit_ols(y, X, sm):
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    res = model.fit()
    return res


def standardize_series(s):
    s = pd.to_numeric(s, errors='coerce')
    return (s - s.mean()) / s.std(ddof=0)


def main():
    sm = ensure_statsmodels()

    df, used_path = load_processed_data()

    # Ensure required columns exist
    required_cols = [
        'T3_panas_joviality', 'age', 'gender', 'children', 'work_hours', 'work_days',
        'T1_panas_joviality', 'req_control', 'req_mastery', 'req_relax', 'req_detach', 'hassles'
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in processed data: {missing_cols}")

    # Build datasets for unstandardized and standardized models
    y_unstd = pd.to_numeric(df['T3_panas_joviality'], errors='coerce')
    X_unstd = df[['age', 'gender', 'children', 'work_hours', 'work_days',
                  'T1_panas_joviality', 'req_control', 'req_mastery', 'req_relax', 'req_detach', 'hassles']].apply(
        pd.to_numeric, errors='coerce'
    )

    # Listwise deletion for rows with missing on key vars
    data_unstd = pd.concat([y_unstd, X_unstd], axis=1).dropna()
    y_unstd = data_unstd['T3_panas_joviality']
    X_unstd = data_unstd.drop(columns=['T3_panas_joviality'])

    # Standardized model (z-score all predictors and outcome)
    y_std = standardize_series(df['T3_panas_joviality'])
    X_std = df[['age', 'gender', 'children', 'work_hours', 'work_days',
                'T1_panas_joviality', 'req_control', 'req_mastery', 'req_relax', 'req_detach', 'hassles']].apply(
        standardize_series
    )

    data_std = pd.concat([y_std, X_std], axis=1).dropna()
    y_std = data_std['T3_panas_joviality']
    X_std = data_std.drop(columns=['T3_panas_joviality'])

    # Fit models
    res_unstd = fit_ols(y_unstd, X_unstd, sm)
    res_std = fit_ols(y_std, X_std, sm)

    # Extract focal coefficient (req_relax)
    coef_unstd = res_unstd.params.get('req_relax', float('nan'))
    pval_unstd = res_unstd.pvalues.get('req_relax', float('nan'))
    coef_std = res_std.params.get('req_relax', float('nan'))
    pval_std = res_std.pvalues.get('req_relax', float('nan'))

    out = {
        'input_data_path': used_path,
        'n_obs_unstandardized': int(res_unstd.nobs),
        'n_obs_standardized': int(res_std.nobs),
        'model_unstandardized': {
            'dependent': 'T3_panas_joviality',
            'independent': ['age', 'gender', 'children', 'work_hours', 'work_days',
                            'T1_panas_joviality', 'req_control', 'req_mastery', 'req_relax', 'req_detach', 'hassles'],
            'r_squared': float(res_unstd.rsquared),
            'adj_r_squared': float(res_unstd.rsquared_adj),
            'coef_req_relax': float(coef_unstd) if pd.notnull(coef_unstd) else None,
            'pvalue_req_relax': float(pval_unstd) if pd.notnull(pval_unstd) else None
        },
        'model_standardized': {
            'dependent': 'z(T3_panas_joviality)',
            'independent': ['z(age)', 'z(gender)', 'z(children)', 'z(work_hours)', 'z(work_days)',
                            'z(T1_panas_joviality)', 'z(req_control)', 'z(req_mastery)', 'z(req_relax)', 'z(req_detach)', 'z(hassles)'],
            'r_squared': float(res_std.rsquared),
            'adj_r_squared': float(res_std.rsquared_adj),
            'coef_req_relax': float(coef_std) if pd.notnull(coef_std) else None,
            'pvalue_req_relax': float(pval_std) if pd.notnull(pval_std) else None
        }
    }

    # Ensure results dir exists
    try:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    except Exception:
        pass

    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    # Also print a short summary for logs
    print(json.dumps({
        'results_path': RESULTS_PATH,
        'unstandardized_coef_req_relax': out['model_unstandardized']['coef_req_relax'],
        'unstandardized_pvalue_req_relax': out['model_unstandardized']['pvalue_req_relax'],
        'standardized_coef_req_relax': out['model_standardized']['coef_req_relax'],
        'standardized_pvalue_req_relax': out['model_standardized']['pvalue_req_relax']
    }, indent=2))


if __name__ == '__main__':
    main()
