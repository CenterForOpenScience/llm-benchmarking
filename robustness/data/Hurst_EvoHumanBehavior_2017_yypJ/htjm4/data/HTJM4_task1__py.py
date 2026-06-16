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


def to_numeric_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        pass
    codes, _ = pd.factorize(s.astype(str), sort=True)
    return pd.Series(codes, index=s.index, dtype=float)


def pairwise_nonmissing(df: pd.DataFrame, cols):
    return df[cols].dropna()


def spearman_corr(x: pd.Series, y: pd.Series):
    r, p = stats.spearmanr(x, y, nan_policy='omit')
    return float(r), float(p)


def kendall_corr(x: pd.Series, y: pd.Series):
    r, p = stats.kendalltau(x, y, nan_policy='omit')
    return float(r), float(p)


def partial_spearman(df: pd.DataFrame, x_col: str, y_col: str, covar_cols):
    cols = [x_col, y_col] + covar_cols
    d = df[cols].dropna().copy()
    if d.shape[0] < 5:
        return math.nan, math.nan, int(d.shape[0])
    for c in cols:
        d[c] = d[c].rank(method='average')
    X = d[covar_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    yx = d[x_col].to_numpy(dtype=float)
    yy = d[y_col].to_numpy(dtype=float)
    bx, *_ = np.linalg.lstsq(X, yx, rcond=None)
    by, *_ = np.linalg.lstsq(X, yy, rcond=None)
    rx = yx - X @ bx
    ry = yy - X @ by
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = d.shape[0]
    k = len(covar_cols)
    dfree = max(n - k - 2, 1)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt(dfree / max(1e-12, 1 - r * r))
        p = 2 * stats.t.sf(abs(t), dfree)
    return r, float(p), int(n)


def main():
    parser = argparse.ArgumentParser(description='HTJM4 Task1 Python analysis: Spearman and Kendall correlations, plus partial Spearman.')
    parser.add_argument('--data', required=False, default='/app/data/1-s2.0-S1090513816301118-mmc1.csv', help='Path to input dataset (.csv or .xlsx) under /app/data')
    parser.add_argument('--out', required=False, default='/app/data/htjm4_task1_results.json', help='Path to output JSON under /app/data')
    args = parser.parse_args()

    df = load_data(args.data)

    required_totals = ['DSM5_Total', 'MiniK_Total', 'HKSS_Total']
    missing_cols = [c for c in required_totals if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in dataset: {missing_cols}")

    covars = []
    if 'Age' in df.columns:
        covars.append('Age')
    if 'Sex' in df.columns:
        if not np.issubdtype(df['Sex'].dtype, np.number):
            df['Sex'] = to_numeric_series(df['Sex'])
        covars.append('Sex')

    results = {
        'task': 'Task1',
        'alpha': 0.005,
        'pairs': [],
        'partial_spearman': [],
        'notes': 'Non-parametric correlations between DSM5_Total and life history totals (MiniK_Total, HKSS_Total). Partial Spearman controls for Age and Sex where available.'
    }

    pairs = [
        ('DSM5_Total', 'MiniK_Total'),
        ('DSM5_Total', 'HKSS_Total')
    ]

    for x, y in pairs:
        sub = pairwise_nonmissing(df, [x, y])
        n = int(sub.shape[0])
        if n >= 3:
            rs, ps = spearman_corr(sub[x], sub[y])
            rk, pk = kendall_corr(sub[x], sub[y])
        else:
            rs = ps = rk = pk = math.nan
        results['pairs'].append({'x': x, 'y': y, 'method': 'spearman', 'correlation': rs, 'p_value': ps, 'n': n})
        results['pairs'].append({'x': x, 'y': y, 'method': 'kendall', 'correlation': rk, 'p_value': pk, 'n': n})

    covar_use = [c for c in ['Age', 'Sex'] if c in covars]
    if len(covar_use) > 0:
        for x, y in pairs:
            rpart, ppart, n_eff = partial_spearman(df, x, y, covar_use)
            results['partial_spearman'].append({
                'x': x, 'y': y, 'covariates': covar_use, 'correlation': rpart, 'p_value': ppart, 'n': n_eff
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps({'status': 'ok', 'out': args.out}))


if __name__ == '__main__':
    main()
