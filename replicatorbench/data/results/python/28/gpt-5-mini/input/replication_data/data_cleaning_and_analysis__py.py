"""
Data cleaning and analysis translated from the provided Stata .do file.
Reads: /app/data/original/28/input/replication_data/FinalData.csv
Writes: /app/data/replication_results.json and /app/data/replication_results.csv

Performs:
 - Basic variable type cleaning
 - Descriptive statistics by treatment
 - Mann-Whitney U test (ranksum) on effort_t1 by treatment
 - Two-sample t-test (for reference)

This script is intentionally simple and self-contained.
"""
import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# Paths (try multiple likely mount locations inside the container)
INPUT_PATHS = [
    '/app/data/original/28/input/replication_data/FinalData.csv',
    '/workspace/replication_data/replication_data/FinalData.csv',
    '/workspace/replication_data/FinalData.csv',
    './replication_data/FinalData.csv',
    './FinalData.csv'
]
INPUT_PATH = next((p for p in INPUT_PATHS if os.path.exists(p)), None)
# Choose output locations: prefer /app/data if available (host-mounted), otherwise /workspace
if os.path.exists('/app/data'):
    OUTPUT_JSON = '/app/data/replication_results.json'
    OUTPUT_CSV = '/app/data/replication_results.csv'
else:
    OUTPUT_JSON = '/workspace/replication_results.json'
    OUTPUT_CSV = '/workspace/replication_results.csv'

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input data not found at {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Ensure relevant variables are numeric
    for col in ['treat', 'effort_t1', 'effort_t2', 'age', 'subject']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic sample sizes
    total_n = len(df)
    n_by_treat = df['treat'].value_counts(dropna=True).to_dict()

    # Summary stats for effort_t1 by treatment
    summary = df.groupby('treat')['effort_t1'].agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index()
    summary_dict = summary.to_dict(orient='records')

    # Prepare samples for tests: treat==1 (OS) vs treat==2 (TS)
    os_sample = df.loc[df['treat'] == 1, 'effort_t1'].dropna()
    ts_sample = df.loc[df['treat'] == 2, 'effort_t1'].dropna()

    # Mann-Whitney U test (two-sided)
    try:
        mw_stat, mw_p = stats.mannwhitneyu(os_sample, ts_sample, alternative='two-sided')
    except Exception as e:
        mw_stat, mw_p = None, None

    # Two-sample t-test (Welch)
    try:
        t_stat, t_p = stats.ttest_ind(os_sample, ts_sample, equal_var=False, nan_policy='omit')
    except Exception as e:
        t_stat, t_p = None, None

    # Effect size: difference in means and Cohen's d (pooled std not assumed)
    mean_diff = ts_sample.mean() - os_sample.mean() if (len(ts_sample) and len(os_sample)) else None
    # Cohen's d (using pooled std)
    try:
        pooled_sd = np.sqrt(((len(ts_sample)-1)*ts_sample.std(ddof=1)**2 + (len(os_sample)-1)*os_sample.std(ddof=1)**2) / (len(ts_sample)+len(os_sample)-2))
        cohen_d = (ts_sample.mean() - os_sample.mean()) / pooled_sd
    except Exception:
        cohen_d = None

    results = {
        'total_n': int(total_n),
        'n_by_treat': {int(k): int(v) for k,v in n_by_treat.items()},
        'effort_t1_summary_by_treat': summary_dict,
        'mann_whitney_u': {'statistic': None if mw_stat is None else float(mw_stat), 'p_value': None if mw_p is None else float(mw_p)},
        't_test_welch': {'statistic': None if t_stat is None else float(t_stat), 'p_value': None if t_p is None else float(t_p)},
        'mean_difference_ts_minus_os': None if mean_diff is None else float(mean_diff),
        'cohens_d_ts_vs_os': None if cohen_d is None else float(cohen_d)
    }

    # Save JSON and CSV summary
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    summary.to_csv(OUTPUT_CSV, index=False)

    # Print concise output
    print('Replication analysis complete. Results saved to:')
    print(' -', OUTPUT_JSON)
    print(' -', OUTPUT_CSV)
    print('\nSummary by treatment:')
    print(summary)
    print('\nMann-Whitney U test (two-sided): stat =', mw_stat, ', p =', mw_p)
    print('Welch t-test: stat =', t_stat, ', p =', t_p)
    print('Mean difference (TS - OS):', mean_diff)
    print("Cohen's d (TS vs OS):", cohen_d)

if __name__ == '__main__':
    main()
