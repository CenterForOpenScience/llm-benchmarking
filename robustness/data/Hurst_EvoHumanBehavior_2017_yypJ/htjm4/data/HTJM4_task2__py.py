#!/usr/bin/env python3
import argparse
import json
import os
import math
import numpy as np
import pandas as pd
from scipy import stats


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    elif ext in ('.xls', '.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported data file extension: {ext}")


def pairwise_nonmissing(df: pd.DataFrame, cols):
    return df[cols].dropna()


def kendall_corr(x: pd.Series, y: pd.Series):
    r, p = stats.kendalltau(x, y, nan_policy='omit')
    return float(r), float(p)


def spearman_corr(x: pd.Series, y: pd.Series):
    r, p = stats.spearmanr(x, y, nan_policy='omit')
    return float(r), float(p)


def main():
    parser = argparse.ArgumentParser(description='HTJM4 Task2 Python analysis: Kendall tau (primary) and Spearman (supplemental).')
    parser.add_argument('--data', required=False, default='/app/data/1-s2.0-S1090513816301118-mmc1.csv', help='Path to input dataset (.csv or .xlsx) under /app/data')
    parser.add_argument('--out', required=False, default='/app/data/htjm4_task2_results.json', help='Path to output JSON under /app/data')
    args = parser.parse_args()

    df = load_data(args.data)

    required_totals = ['DSM5_Total', 'MiniK_Total']
    missing_cols = [c for c in required_totals if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in dataset: {missing_cols}")

    results = {
        'task': 'Task2',
        'alpha': 0.005,
        'pairs': [],
        'notes': 'Kendall tau between DSM5_Total and MiniK_Total; Spearman reported as supplemental.'
    }

    x, y = 'DSM5_Total', 'MiniK_Total'
    sub = pairwise_nonmissing(df, [x, y])
    n = int(sub.shape[0])
    if n >= 3:
        rk, pk = kendall_corr(sub[x], sub[y])
        rs, ps = spearman_corr(sub[x], sub[y])
    else:
        rk = pk = rs = ps = math.nan
    results['pairs'].append({'x': x, 'y': y, 'method': 'kendall', 'correlation': rk, 'p_value': pk, 'n': n})
    results['pairs'].append({'x': x, 'y': y, 'method': 'spearman', 'correlation': rs, 'p_value': ps, 'n': n})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps({'status': 'ok', 'out': args.out}))


if __name__ == '__main__':
    main()
