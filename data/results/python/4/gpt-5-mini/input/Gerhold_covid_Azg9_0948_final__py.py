import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# Try several candidate paths for the data file depending on how volumes are mounted
CANDIDATE_PATHS = [
    '/app/data/original/4/python/replication_data/data_gerhold.csv',
    '/app/data/./original/4/python/replication_data/data_gerhold.csv',
    '/workspace/replication_data/data_gerhold.csv',
    'replication_data/data_gerhold.csv',
    './replication_data/data_gerhold.csv'
]
DATA_PATH = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break
OUTPUT_PATH = '/app/data/original/4/python/replication_data/replication_results.json' if os.path.exists('/app/data') else 'replication_data/replication_results.json'

def var_test(x, y):
    # F-test for equality of variances, analogous to R's var.test
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = x.size
    n2 = y.size
    s1 = np.var(x, ddof=1)
    s2 = np.var(y, ddof=1)
    if s2 == 0:
        return {'f_stat': None, 'p_value': None, 'df1': n1-1, 'df2': n2-1}
    f = s1 / s2
    df1 = n1 - 1
    df2 = n2 - 1
    # two-sided p-value
    p = stats.f.cdf(f, df1, df2)
    p_two = 2 * min(p, 1 - p)
    return {'f_stat': float(f), 'p_value': float(p_two), 'df1': df1, 'df2': df2}


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = x.size
    n2 = y.size
    m1 = np.nanmean(x)
    m2 = np.nanmean(y)
    s1 = np.nanstd(x, ddof=1)
    s2 = np.nanstd(y, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return None
    d = (m1 - m2) / pooled_sd
    return float(d)


def analyze_variable(df, var):
    # drop NA in var
    subset = df[[var, 'female']].dropna()
    female = subset[subset['female'] == 1][var].astype(float)
    male = subset[subset['female'] == 0][var].astype(float)
    n_f = female.size
    n_m = male.size
    res = {
        'n_female': int(n_f),
        'n_male': int(n_m),
        'mean_female': None,
        'mean_male': None,
        'sd_female': None,
        'sd_male': None,
        'f_test': {},
        't_test': {},
        'cohens_d': None
    }
    if n_f > 0:
        res['mean_female'] = float(np.nanmean(female))
        res['sd_female'] = float(np.nanstd(female, ddof=1)) if n_f > 1 else None
    if n_m > 0:
        res['mean_male'] = float(np.nanmean(male))
        res['sd_male'] = float(np.nanstd(male, ddof=1)) if n_m > 1 else None

    # F-test
    if n_f > 1 and n_m > 1:
        res['f_test'] = var_test(female, male)
    else:
        res['f_test'] = {'f_stat': None, 'p_value': None}

    # t-test assuming equal variances
    try:
        t_res = stats.ttest_ind(female, male, equal_var=True, nan_policy='omit')
        res['t_test'] = {'t_stat': float(t_res.statistic) if t_res.statistic is not None else None,
                         'p_value': float(t_res.pvalue) if t_res.pvalue is not None else None}
    except Exception as e:
        res['t_test'] = {'error': str(e)}

    # Cohen's d
    try:
        d = cohen_d(female, male)
        res['cohens_d'] = d
    except Exception:
        res['cohens_d'] = None

    return res


def main():
    if DATA_PATH is None or not os.path.exists(DATA_PATH):
        print('Data file not found at', DATA_PATH)
        return 1
    df = pd.read_csv(DATA_PATH)

    # Ensure gender filter: remove gender == 3
    # gender column might be string or numeric
    if 'gender' in df.columns:
        # coerce to numeric where possible
        try:
            df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
        except Exception:
            pass
        df = df[df['gender'] != 3]

    # ensure female column exists and is numeric 1/0
    if 'female' in df.columns:
        try:
            df['female'] = pd.to_numeric(df['female'], errors='coerce')
        except Exception:
            # try mapping
            df['female'] = df['female'].map({'1': 1, '0': 0}).astype(float)
    else:
        # try to construct from gender if present (gender 1 female?)
        df['female'] = None

    # report sample sizes
    n_total = int(df.shape[0])
    n_female = int(df[df['female'] == 1].shape[0]) if 'female' in df.columns else None
    n_male = int(df[df['female'] == 0].shape[0]) if 'female' in df.columns else None

    results = {
        'n_total': n_total,
        'n_female': n_female,
        'n_male': n_male,
        'variables_available': [],
        'by_variable': {}
    }

    for var in ['mh_anxiety_1', 'mh_anxiety_3']:
        if var in df.columns:
            results['variables_available'].append(var)
            results['by_variable'][var] = analyze_variable(df, var)

    # write results
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print('Analysis complete. Results written to', OUTPUT_PATH)
    return 0

if __name__ == '__main__':
    main()
