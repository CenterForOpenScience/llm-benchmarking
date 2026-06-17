import os
import sys
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def resolve_data_path():
    """Resolve the dataset path across different container mount layouts."""
    candidates = [
        "/app/data/original/16/0108_python_gpt5/replication_data/replicationDataset_Malik2020_with.year.csv",
        os.path.join("/workspace", "replication_data", "replicationDataset_Malik2020_with.year.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "replication_data", "replicationDataset_Malik2020_with.year.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback: search under /workspace
    search_root = "/workspace"
    if os.path.exists(search_root):
        for dirpath, dirnames, filenames in os.walk(search_root):
            if "replicationDataset_Malik2020_with.year.csv" in filenames:
                return os.path.join(dirpath, "replicationDataset_Malik2020_with.year.csv")
    raise FileNotFoundError("Could not locate dataset 'replicationDataset_Malik2020_with.year.csv' in expected locations. Tried: " + ", ".join(candidates))


def load_and_prepare(data_path):
    df = pd.read_csv(data_path)
    # Parse date
    df['date2'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    # Drop rows with invalid dates
    df = df.dropna(subset=['date2'])
    # Construct numeric time: days since min date
    df['time'] = (df['date2'] - df['date2'].min()).dt.days.astype(int)
    # Normalize lockdown to binary 0/1
    if 'lockdown' not in df.columns:
        raise ValueError("Required column 'lockdown' not found in dataset")
    df['lockdown'] = (df['lockdown'].astype(float) != 0).astype(int)
    # Ensure city is a string identifier
    if 'city' not in df.columns:
        raise ValueError("Required column 'city' not found in dataset")
    df['city'] = df['city'].astype(str)
    return df


def fit_mixedlm(df, yvar, artifacts_dir):
    # Keep only relevant complete cases
    vars_needed = ['city', 'time', 'lockdown', yvar]
    d = df.dropna(subset=vars_needed).copy()
    if d.empty:
        raise ValueError(f"No data available after dropping missing values for {yvar}")

    formula = f"{yvar} ~ time + lockdown"
    # Fit model with random intercepts by city
    md = sm.MixedLM.from_formula(formula, groups='city', data=d, re_formula=None)
    result = None
    fit_logs = []
    try:
        result = md.fit(method='lbfgs', maxiter=2000, disp=False)
        fit_logs.append('lbfgs converged')
    except Exception as e:
        fit_logs.append(f'lbfgs failed: {e}')
        try:
            result = md.fit(method='bfgs', maxiter=2000, disp=False)
            fit_logs.append('bfgs converged')
        except Exception as e2:
            fit_logs.append(f'bfgs failed: {e2}')
            # Final fallback: Nelder-Mead via optimize kw is not supported; try powell-like by changing start params
            start_params = np.asarray(md.fit(method='nm', maxiter=100, disp=False).params)
            result = md.fit(start_params=start_params, method='lbfgs', maxiter=2000, disp=False)
            fit_logs.append('fallback with nm start + lbfgs converged')

    # Save summary
    summary_txt = result.summary().as_text()
    with open(os.path.join(artifacts_dir, f"{yvar}_mixedlm_summary.txt"), 'w') as f:
        f.write(summary_txt)
        f.write("\n\n")
        f.write("\n".join(fit_logs))

    # Collect key stats
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    ci = result.conf_int()
    ci.columns = ['ci_lower', 'ci_upper']

    def row_for(param_name):
        return {
            'outcome': yvar,
            'parameter': param_name,
            'value': float(params.get(param_name, np.nan)),
            'standard_error': float(bse.get(param_name, np.nan)),
            'p_value': float(pvalues.get(param_name, np.nan)),
            'ci_lower': float(ci.loc[param_name, 'ci_lower']) if param_name in ci.index else np.nan,
            'ci_upper': float(ci.loc[param_name, 'ci_upper']) if param_name in ci.index else np.nan,
            'nobs': float(result.nobs),
            'aic': float(getattr(result, 'aic', np.nan)),
            'bic': float(getattr(result, 'bic', np.nan)),
            'converged': bool(getattr(result, 'converged', False))
        }

    rows = [row_for('Intercept'), row_for('time'), row_for('lockdown')]
    return rows


def main():
    # Paths inside the container
    data_path = "/app/data/original/16/0108_python_gpt5/replication_data/replicationDataset_Malik2020_with.year.csv"
    # Resolve actual dataset path inside container mounts
    data_path = resolve_data_path()
    artifacts_dir = "/app/artifacts"
    ensure_dirs([artifacts_dir])

    print("Loading and preparing data...", flush=True)
    df = load_and_prepare(data_path)
    print(f"Data shape after preparation: {df.shape}", flush=True)

    outcomes = []
    # Primary outcome: CMI
    if 'CMI' in df.columns:
        outcomes.append('CMI')
    else:
        print("Warning: CMI column not found; primary model will be skipped.", flush=True)
    # Additional robustness: CMRT_residential
    if 'CMRT_residential' in df.columns:
        outcomes.append('CMRT_residential')
    else:
        print("Note: CMRT_residential not found; robustness model will be skipped.", flush=True)

    all_rows = []
    for y in outcomes:
        print(f"Fitting MixedLM for outcome: {y} ...", flush=True)
        try:
            rows = fit_mixedlm(df, y, artifacts_dir)
            all_rows.extend(rows)
            print(f"Completed model for {y}", flush=True)
        except Exception as e:
            print(f"Error fitting model for {y}: {e}", flush=True)

    if all_rows:
        results_df = pd.DataFrame(all_rows)
        results_csv = os.path.join(artifacts_dir, 'model_results.csv')
        results_df.to_csv(results_csv, index=False)
        # Also write a brief human-readable summary
        with open(os.path.join(artifacts_dir, 'analysis_summary.txt'), 'w') as f:
            f.write('MixedLM Replication Results\n')
            f.write('='*32 + '\n')
            for y in sorted(set(r['outcome'] for r in all_rows)):
                f.write(f"\nOutcome: {y}\n")
                sub = results_df[results_df['outcome'] == y]
                for _, r in sub.iterrows():
                    f.write(f"  {r['parameter']}: coef={r['value']:.4f}, se={r['standard_error']:.4f}, p={r['p_value']:.4g}, CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]\n")
        print(f"Saved results to {results_csv}", flush=True)
    else:
        print("No models were fitted; please check dataset columns and preprocessing steps.", flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
