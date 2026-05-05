#!/usr/bin/env python3
"""
Replication analysis for Rich & Gureckis (2018) focal claim:
Participants in the contingent feedback condition adopt an inferior one-dimension categorization strategy more often than those in the full-information condition.

The script reproduces the analysis performed in the original RMarkdown (`analysis.Rmd`) using Python.
Outputs descriptive statistics, independent-samples t-test, and Cohen's d effect size.
"""
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_dataset():
    """Attempt to locate the analysis-data.csv file inside /app/data no matter the nested folder structure."""
    search_root = Path('/app/data')
    for p in search_root.rglob('analysis-data.csv'):
        return p
    raise FileNotFoundError("analysis-data.csv not found inside /app/data")


def compute_test_phase_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the transformation steps used in analysis.Rmd for the test phase."""
    test_df = df[df['phase'] == 'Test'].copy()

    # mean accuracy across test trials per subject for different rules
    grouped = test_df.groupby(['subject', 'condition']).agg(
        acc_2d=('correct', 'mean'),
        acc_1d_a=('correct_1d_a', 'mean'),
        acc_1d_b=('correct_1d_b', 'mean')
    ).reset_index()

    # Choose best of the two 1D accuracies as in the original analysis
    grouped['acc_1d'] = grouped[['acc_1d_a', 'acc_1d_b']].max(axis=1)
    return grouped


def independent_ttest(df: pd.DataFrame):
    """Perform independent samples t-test comparing acc_1d by condition (contingent vs full-information)."""
    contingent = df[df['condition'] == 'contingent']['acc_1d']
    full_info = df[df['condition'] == 'full-information']['acc_1d']

    # ensure equal sample sizes not required; use equal_var=True to match original assumption
    t_stat, p_val = stats.ttest_ind(contingent, full_info, equal_var=True)

    # Cohen's d (pooled SD)
    n1, n2 = contingent.size, full_info.size
    s1, s2 = contingent.std(ddof=1), full_info.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohens_d = (contingent.mean() - full_info.mean()) / pooled_sd

    return {
        'n_contingent': int(n1),
        'n_full_information': int(n2),
        'mean_contingent': contingent.mean(),
        'sd_contingent': s1,
        'mean_full_information': full_info.mean(),
        'sd_full_information': s2,
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d
    }


def main():
    csv_path = load_dataset()
    print(f"Found dataset at: {csv_path}")
    df = pd.read_csv(csv_path)

    # Basic sanity check
    required_cols = {'phase', 'correct', 'correct_1d_a', 'correct_1d_b', 'subject', 'condition'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    scores_df = compute_test_phase_scores(df)
    results = independent_ttest(scores_df)

    # Save results
    out_path = Path('/app/data/replication_results.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)

    print("Replication analysis complete. Results saved to", out_path)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
