import os
import json
import sys
import traceback
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_DIR = '/app/data/'
OUT_DIR = '/app/data'

def try_load():
    pkl_path = os.path.join(DATA_DIR, 'PISA2012.replication.pkl')
    rds_path = os.path.join(DATA_DIR, 'PISA2012.replication.RDS')
    r_path = os.path.join(DATA_DIR, '!!CORRECTION!!_Seaton_AmEduResJourn_2010_Blxd_final.R')
    if os.path.exists(pkl_path):
        try:
            df = pd.read_pickle(pkl_path)
            return df, 'pkl'
        except Exception as e:
            raise
    # try pyreadr for RDS
    try:
        import pyreadr
        if os.path.exists(rds_path):
            result = pyreadr.read_r(rds_path)
            # take first object
            for k,v in result.items():
                df = v
                return df, 'rds'
    except Exception:
        pass
    # fallback: try to parse any file in directory with pandas
    for f in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, f)
        if f.lower().endswith('.csv'):
            try:
                df = pd.read_csv(fp)
                return df, 'csv'
            except Exception:
                continue
    raise FileNotFoundError('No supported data file found in ' + DATA_DIR)


def find_column(df, keywords, require_all=False):
    cols = list(df.columns)
    keywords = [k.lower() for k in keywords]
    matches = []
    for c in cols:
        cl = c.lower()
        if require_all:
            if all(k in cl for k in keywords):
                matches.append(c)
        else:
            for k in keywords:
                if k in cl:
                    matches.append(c)
                    break
    return matches


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {'status': 'started'}
    try:
        df, src = try_load()
        out['data_source'] = src
        # basic info
        out['n_rows'] = int(df.shape[0])
        # attempt to find math self-concept
        sc_candidates = find_column(df, ['selfconcept', 'self_concept', 'self concept', 'mathself', 'math_self', 'math self', 'self'])
        math_candidates = find_column(df, ['math', 'achievement', 'pv', 'score'])
        memor_candidates = find_column(df, ['memor', 'memorisation', 'memorization'])
        schoolid_candidates = find_column(df, ['schoolid', 'school_id', 'schoolid', 'school'])

        out['candidates'] = {
            'selfconcept': sc_candidates[:5],
            'math': math_candidates[:5],
            'memorization': memor_candidates[:5],
            'school_id': schoolid_candidates[:5]
        }

        # choose plausible columns
        # Choose self-concept: prefer columns containing 'self' and 'math' together
        sc_col = None
        for c in sc_candidates:
            if 'math' in c.lower() or 'math' in ''.join(sc_candidates).lower():
                sc_col = c
                break
        if sc_col is None and len(sc_candidates)>0:
            sc_col = sc_candidates[0]
        # Choose memorization
        mem_col = memor_candidates[0] if len(memor_candidates)>0 else None
        # Choose math achievement column
        math_col = None
        for c in math_candidates:
            if 'pv' in c.lower() or 'score' in c.lower() or 'achievement' in c.lower():
                math_col = c
                break
        if math_col is None and len(math_candidates)>0:
            math_col = math_candidates[0]
        # school id
        school_col = None
        for c in schoolid_candidates:
            if 'id' in c.lower() or 'school'==c.lower():
                school_col = c
                break
        if school_col is None and len(schoolid_candidates)>0:
            school_col = schoolid_candidates[0]

        out['selected'] = {'selfconcept': sc_col, 'memorization': mem_col, 'math': math_col, 'school_id': school_col}

        # check requirements
        if sc_col is None:
            raise ValueError('Could not find a column for mathematics self-concept in data. Candidates: ' + str(sc_candidates))
        if mem_col is None:
            raise ValueError('Could not find a column for memorization in data. Candidates: ' + str(memor_candidates))
        if math_col is None and school_col is None:
            raise ValueError('Could not find math achievement or school id to compute school-average ability. Candidates math: '+str(math_candidates)+" school: "+str(schoolid_candidates))

        df2 = df.copy()
        # convert columns to numeric if possible
        for c in [sc_col, mem_col, math_col, school_col]:
            if c and c in df2.columns:
                df2[c] = pd.to_numeric(df2[c], errors='coerce')

        # compute school average ability
        if math_col is not None and school_col is not None and school_col in df2.columns and math_col in df2.columns:
            df2['school_avg_math'] = df2.groupby(school_col)[math_col].transform('mean')
        elif math_col is not None:
            # if no school id, approximate school average as global mean (fallback)
            df2['school_avg_math'] = df2[math_col].astype(float).mean()
        else:
            raise ValueError('Insufficient data to compute school average math ability')

        # drop rows missing outcome or key vars
        key_vars = [sc_col, mem_col, 'school_avg_math']
        df_est = df2[key_vars].dropna()
        df_est = df_est.rename(columns={sc_col: 'math_self_concept', mem_col: 'memorization'})

        # standardize/scale variables
        df_est['math_self_concept_z'] = (df_est['math_self_concept'] - df_est['math_self_concept'].mean())/df_est['math_self_concept'].std(ddof=0)
        df_est['memorization_z'] = (df_est['memorization'] - df_est['memorization'].mean())/df_est['memorization'].std(ddof=0)
        df_est['school_avg_math_z'] = (df2['school_avg_math'] - df2['school_avg_math'].mean())/df2['school_avg_math'].std(ddof=0)
        # align index
        df_est['school_avg_math_z'] = df_est.index.to_series().map(df2['school_avg_math'].loc)
        df_est['school_avg_math_z'] = (df_est['school_avg_math_z'] - df_est['school_avg_math_z'].mean())/df_est['school_avg_math_z'].std(ddof=0)

        # ensure no infinite or NaN
        df_est = df_est.replace([np.inf, -np.inf], np.nan).dropna()
        out['n_estimation'] = int(df_est.shape[0])

        # fit OLS: math_self_concept_z ~ memorization_z + school_avg_math_z + memorization_z:school_avg_math_z
        df_est = df_est.reset_index(drop=True)
        formula = 'math_self_concept_z ~ memorization_z + school_avg_math_z + memorization_z:school_avg_math_z'
        mod = smf.ols(formula=formula, data=df_est).fit(cov_type='HC1')

        coef = mod.params.to_dict()
        se = mod.bse.to_dict()
        pvalues = mod.pvalues.to_dict()
        conf = mod.conf_int().to_dict()

        results = {
            'formula': formula,
            'n_obs': int(mod.nobs),
            'params': coef,
            'std_errors': se,
            'pvalues': pvalues,
            'conf_int': conf,
            'rsquared': float(mod.rsquared)
        }

        # save outputs
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
            json.dump({'status': 'success', 'results': results, 'meta': out}, f, indent=2)
        with open(os.path.join(OUT_DIR, 'regression_summary.txt'), 'w') as f:
            f.write(mod.summary().as_text())

        print('Estimation completed. Results saved to /app/data/results.json and /app/data/regression_summary.txt')

    except Exception as e:
        tb = traceback.format_exc()
        err = {'status': 'error', 'message': str(e), 'traceback': tb}
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
            json.dump(err, f, indent=2)
        print('Error during execution:', str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
