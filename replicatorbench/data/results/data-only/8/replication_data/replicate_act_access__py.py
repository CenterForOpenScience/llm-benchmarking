import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Paths
DATA_PATH = "/app/data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta"
OUT_JSON = "/app/data/act_access_regression_results.json"
OUT_SUMMARY_CSV = "/app/data/act_access_group_means.csv"


def main():
    # Load data
    try:
        df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read Stata file at {DATA_PATH}: {e}")

    # Validate required columns
    required_cols = [
        'drugs_taken_AL',  # outcome: ACT access/use during illness episode
        'group',           # treatment: Any ACT subsidy (assumed 0/1)
        'cu_code',         # cluster id (household code)
        'sub_county',      # location strata FE proxy
        'wave',            # wave/time FE
        'weight'           # sampling weight
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    # Prepare analysis sample: drop missing outcome or treatment
    work = df.copy()
    work = work.dropna(subset=['drugs_taken_AL', 'group', 'cu_code'])

    # Coerce types
    work['any_subsidy'] = work['group'].astype(int)
    work['act_access'] = work['drugs_taken_AL'].astype(int)

    # Descriptive: means by treatment group (unweighted and weighted)
    means_unw = work.groupby('any_subsidy')['act_access'].mean().rename('mean_unweighted')
    # Weighted mean function
    def wmean(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        mask = np.isfinite(x) & np.isfinite(w)
        if mask.sum() == 0:
            return np.nan
        return np.average(x[mask], weights=w[mask])

    means_w = work.groupby('any_subsidy').apply(lambda g: wmean(g['act_access'], g['weight'])).rename('mean_weighted')

    means_df = pd.concat([means_unw, means_w], axis=1).reset_index().rename(columns={'any_subsidy': 'treatment_any_subsidy'})
    try:
        means_df.to_csv(OUT_SUMMARY_CSV, index=False)
    except Exception as e:
        # Non-fatal
        print(f"Warning: failed to write group means CSV: {e}")

    # Regression models: Linear Probability Model (OLS) with cluster-robust SE at household level
    results_payload = {
        'data': {
            'n_total_rows': int(len(df)),
            'n_analysis_rows': int(len(work)),
            'n_treated': int((work['any_subsidy'] == 1).sum()),
            'n_control': int((work['any_subsidy'] == 0).sum())
        },
        'by_group_means': means_df.to_dict(orient='records'),
        'models': {}
    }

    # Unweighted with FE
    try:
        model_unw_fe = smf.ols('act_access ~ any_subsidy + C(sub_county) + C(wave)', data=work)
        res_unw_fe = model_unw_fe.fit(cov_type='cluster', cov_kwds={'groups': work['cu_code']})
        coef = res_unw_fe.params.get('any_subsidy', np.nan)
        se = res_unw_fe.bse.get('any_subsidy', np.nan)
        pval = res_unw_fe.pvalues.get('any_subsidy', np.nan)
        results_payload['models']['unweighted_fe'] = {
            'spec': 'act_access ~ any_subsidy + C(sub_county) + C(wave)',
            'coef_any_subsidy': float(coef) if np.isfinite(coef) else None,
            'se_any_subsidy': float(se) if np.isfinite(se) else None,
            'p_any_subsidy': float(pval) if np.isfinite(pval) else None,
            'r2': float(res_unw_fe.rsquared) if res_unw_fe.rsquared is not None else None
        }
    except Exception as e:
        results_payload['models']['unweighted_fe'] = {'error': f'{e}'}

    # Unweighted without FE (simple diff-in-means via OLS)
    try:
        model_unw = smf.ols('act_access ~ any_subsidy', data=work)
        res_unw = model_unw.fit(cov_type='cluster', cov_kwds={'groups': work['cu_code']})
        coef = res_unw.params.get('any_subsidy', np.nan)
        se = res_unw.bse.get('any_subsidy', np.nan)
        pval = res_unw.pvalues.get('any_subsidy', np.nan)
        results_payload['models']['unweighted'] = {
            'spec': 'act_access ~ any_subsidy',
            'coef_any_subsidy': float(coef) if np.isfinite(coef) else None,
            'se_any_subsidy': float(se) if np.isfinite(se) else None,
            'p_any_subsidy': float(pval) if np.isfinite(pval) else None,
            'r2': float(res_unw.rsquared) if res_unw.rsquared is not None else None
        }
    except Exception as e:
        results_payload['models']['unweighted'] = {'error': f'{e}'}

    # Weighted with FE (WLS). Note: cluster-robust covariance with weights.
    try:
        # statsmodels WLS requires weights as 1/var; we use sampling weights via freq_weights in GLM or use WLS with sqrt weights.
        # Here we use WLS with weights normalized to mean 1.
        w = work['weight'].astype(float)
        w = w / w.mean()
        model_w_fe = smf.wls('act_access ~ any_subsidy + C(sub_county) + C(wave)', data=work, weights=w)
        res_w_fe = model_w_fe.fit(cov_type='cluster', cov_kwds={'groups': work['cu_code']})
        coef = res_w_fe.params.get('any_subsidy', np.nan)
        se = res_w_fe.bse.get('any_subsidy', np.nan)
        pval = res_w_fe.pvalues.get('any_subsidy', np.nan)
        results_payload['models']['weighted_fe'] = {
            'spec': 'act_access ~ any_subsidy + C(sub_county) + C(wave)',
            'coef_any_subsidy': float(coef) if np.isfinite(coef) else None,
            'se_any_subsidy': float(se) if np.isfinite(se) else None,
            'p_any_subsidy': float(pval) if np.isfinite(pval) else None,
            'r2': float(res_w_fe.rsquared) if res_w_fe.rsquared is not None else None
        }
    except Exception as e:
        results_payload['models']['weighted_fe'] = {'error': f'{e}'}

    # Weighted without FE
    try:
        w = work['weight'].astype(float)
        w = w / w.mean()
        model_w = smf.wls('act_access ~ any_subsidy', data=work, weights=w)
        res_w = model_w.fit(cov_type='cluster', cov_kwds={'groups': work['cu_code']})
        coef = res_w.params.get('any_subsidy', np.nan)
        se = res_w.bse.get('any_subsidy', np.nan)
        pval = res_w.pvalues.get('any_subsidy', np.nan)
        results_payload['models']['weighted'] = {
            'spec': 'act_access ~ any_subsidy',
            'coef_any_subsidy': float(coef) if np.isfinite(coef) else None,
            'se_any_subsidy': float(se) if np.isfinite(se) else None,
            'p_any_subsidy': float(pval) if np.isfinite(pval) else None,
            'r2': float(res_w.rsquared) if res_w.rsquared is not None else None
        }
    except Exception as e:
        results_payload['models']['weighted'] = {'error': f'{e}'}

    # Write results JSON
    try:
        with open(OUT_JSON, 'w') as f:
            json.dump(results_payload, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write results JSON: {e}")

    # Also print a short summary to stdout
    print(json.dumps(results_payload['models'].get('unweighted', {})))


if __name__ == '__main__':
    main()
