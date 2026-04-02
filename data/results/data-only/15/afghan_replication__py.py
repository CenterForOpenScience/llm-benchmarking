import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "/app/data/Afghanistan_Election_Violence_2014.csv"
OUTPUT_RESULTS_JSON = "/app/data/replication_results.json"
OUTPUT_RESULTS_CSV = "/app/data/replication_coefficients.csv"


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Fraud mapping: accept 0/1 or labeled strings
    if df['fraud'].dtype == object:
        # Map string labels to binary
        mapping = {
            'Fraud': 1,
            'No Fraud': 0,
            'fraud': 1,
            'no fraud': 0,
            'No fraud': 0,
        }
        df['fraud_bin'] = df['fraud'].map(mapping)
        # If any unmapped, try heuristic
        if df['fraud_bin'].isna().any():
            df['fraud_bin'] = df['fraud'].astype(str).str.lower().str.contains('fraud').astype(int)
    else:
        # Already numeric
        df['fraud_bin'] = df['fraud'].astype(int)

    # Election cycle categorical
    if df['elect'].dtype == object:
        # Expect labels like '1st Election', '2nd Election'
        df['elect_cat'] = df['elect'].astype(str)
    else:
        # numeric codes 1/2
        df['elect_cat'] = df['elect'].astype(str)

    # Violence rates (per 1,000 in 5-day and 60-day windows)
    for v in ['sigact_5r', 'sigact_60r']:
        if v in df.columns:
            df[f'{v}_sq'] = df[v] ** 2
        else:
            df[v] = np.nan
            df[f'{v}_sq'] = np.nan

    # Controls
    # Use electrification proportion (0-1), dev (pcexpend), elevation (km), distance to Kabul (km),
    # population (log), and percent closed polling centers (pcx)
    if 'pop_1314' in df.columns:
        df['log_pop'] = np.log1p(df['pop_1314'])
    else:
        df['log_pop'] = np.nan

    # Ensure numeric types where expected
    num_cols = ['sigact_5r', 'sigact_5r_sq', 'sigact_60r', 'sigact_60r_sq',
                'dist', 'elevationk', 'electric', 'pcexpend', 'log_pop', 'pcx']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Regional command for clustering. Keep as string labels.
    if 'regcom' in df.columns:
        df['regcom_str'] = df['regcom'].astype(str)
    else:
        df['regcom_str'] = 'NA'

    return df


def fit_models(df: pd.DataFrame) -> dict:
    # Define base formula components
    controls = ['dist', 'elevationk', 'electric', 'pcexpend', 'log_pop', 'pcx']
    control_terms = ' + '.join([f'C({c})' if df[c].dtype == object else c for c in controls if c in df.columns])
    # Ensure elect as categorical FE
    elect_term = 'C(elect_cat)'

    results = {}

    # Prepare data subset for Model A (5-day violence)
    model_a_vars = ['fraud_bin', 'sigact_5r', 'sigact_5r_sq', 'elect_cat', 'regcom_str'] + controls
    dfa = df[model_a_vars].dropna()

    # OLS (LPM) with cluster-robust SE by regcom
    formula_a = f"fraud_bin ~ sigact_5r + sigact_5r_sq + {elect_term}"
    if control_terms:
        formula_a += f" + {control_terms}"

    try:
        ols_a = smf.ols(formula=formula_a, data=dfa).fit(cov_type='cluster', cov_kwds={'groups': dfa['regcom_str']})
        results['OLS_sigact5'] = {
            'model': 'OLS (LPM)',
            'nobs': int(ols_a.nobs),
            'clusters': int(dfa['regcom_str'].nunique()),
            'coefficients': ols_a.params.to_dict(),
            'bse': ols_a.bse.to_dict(),
            'pvalues': ols_a.pvalues.to_dict(),
            'violence_term': 'sigact_5r_sq',
            'violence_sq_coef': float(ols_a.params.get('sigact_5r_sq', np.nan)),
            'violence_sq_pvalue': float(ols_a.pvalues.get('sigact_5r_sq', np.nan))
        }
    except Exception as e:
        results['OLS_sigact5_error'] = str(e)

    # Logit with cluster-robust SE
    try:
        logit_a = smf.logit(formula=formula_a, data=dfa).fit(disp=False)
        logit_a_rob = logit_a.get_robustcov_results(cov_type='cluster', groups=dfa['regcom_str'])
        results['Logit_sigact5'] = {
            'model': 'Logit',
            'nobs': int(logit_a_rob.nobs),
            'clusters': int(dfa['regcom_str'].nunique()),
            'coefficients': dict(zip(logit_a_rob.params.index.tolist(), logit_a_rob.params.values.tolist())),
            'bse': dict(zip(logit_a_rob.bse.index.tolist(), logit_a_rob.bse.values.tolist())),
            'pvalues': dict(zip(logit_a_rob.pvalues.index.tolist(), logit_a_rob.pvalues.values.tolist())),
            'violence_term': 'sigact_5r_sq',
            'violence_sq_coef': float(logit_a_rob.params.get('sigact_5r_sq', np.nan)),
            'violence_sq_pvalue': float(logit_a_rob.pvalues.get('sigact_5r_sq', np.nan))
        }
    except Exception as e:
        results['Logit_sigact5_error'] = str(e)

    # Model B (60-day violence) OLS only as robustness
    model_b_vars = ['fraud_bin', 'sigact_60r', 'sigact_60r_sq', 'elect_cat', 'regcom_str'] + controls
    dfb = df[model_b_vars].dropna()
    formula_b = f"fraud_bin ~ sigact_60r + sigact_60r_sq + {elect_term}"
    if control_terms:
        formula_b += f" + {control_terms}"

    try:
        ols_b = smf.ols(formula=formula_b, data=dfb).fit(cov_type='cluster', cov_kwds={'groups': dfb['regcom_str']})
        results['OLS_sigact60'] = {
            'model': 'OLS (LPM)',
            'nobs': int(ols_b.nobs),
            'clusters': int(dfb['regcom_str'].nunique()),
            'coefficients': ols_b.params.to_dict(),
            'bse': ols_b.bse.to_dict(),
            'pvalues': ols_b.pvalues.to_dict(),
            'violence_term': 'sigact_60r_sq',
            'violence_sq_coef': float(ols_b.params.get('sigact_60r_sq', np.nan)),
            'violence_sq_pvalue': float(ols_b.pvalues.get('sigact_60r_sq', np.nan))
        }
    except Exception as e:
        results['OLS_sigact60_error'] = str(e)

    return results


def write_outputs(results: dict):
    # Write JSON summary
    with open(OUTPUT_RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    # Flatten coefficients to CSV
    rows = []
    for name, res in results.items():
        if isinstance(res, dict) and 'coefficients' in res:
            for k, v in res['coefficients'].items():
                row = {
                    'model_name': name,
                    'term': k,
                    'coef': v,
                    'bse': res['bse'].get(k, np.nan),
                    'pvalue': res['pvalues'].get(k, np.nan),
                    'nobs': res.get('nobs', np.nan),
                    'clusters': res.get('clusters', np.nan)
                }
                rows.append(row)
        else:
            # error entry
            rows.append({'model_name': name, 'term': '', 'coef': np.nan, 'bse': np.nan, 'pvalue': np.nan, 'nobs': np.nan, 'clusters': np.nan})

    pd.DataFrame(rows).to_csv(OUTPUT_RESULTS_CSV, index=False)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}. Please place 'Afghanistan_Election_Violence_2014.csv' there.")

    df = load_and_prepare(DATA_PATH)
    results = fit_models(df)
    write_outputs(results)
    # Console hint
    print(f"Saved results to {OUTPUT_RESULTS_JSON} and {OUTPUT_RESULTS_CSV}")


if __name__ == "__main__":
    main()
