import json
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# All IO must use /app/data
DATA_PATH = "/app/data/replicationDataset_Malik2020_with.year.csv"
RESULTS_JSON = "/app/data/replication_results.json"
RESULTS_TXT = "/app/data/replication_model_summary.txt"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Parse dates and create time index (days since first date in sample)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'city']).reset_index(drop=True)
    first_date = df['date'].min()
    df['day_index'] = (df['date'] - first_date).dt.days

    # Ensure types for predictors
    for col in ['lockdown', 'completeLockdown']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Mixed effects model with random intercept by city
    # Model 1: focal claim - effect of lockdown controlling for time trend
    model1 = smf.mixedlm("CMI ~ day_index + lockdown", df, groups=df["city"])  # random intercept per city
    res1 = model1.fit(method='lbfgs', reml=False)

    # Model 2 (optional): add complete lockdown intensity as an incremental effect
    has_complete = 'completeLockdown' in df.columns
    res2 = None
    if has_complete:
        try:
            model2 = smf.mixedlm("CMI ~ day_index + lockdown + completeLockdown", df, groups=df["city"])  # random intercept per city
            res2 = model2.fit(method='lbfgs', reml=False)
        except Exception as e:
            res2 = None

    # Collect results
    out = {
        'data_info': {
            'n_obs': int(df.shape[0]),
            'n_cities': int(df['city'].nunique()),
            'date_range': {
                'start': str(df['date'].min().date()),
                'end': str(df['date'].max().date())
            }
        },
        'model_1': {
            'formula': 'CMI ~ day_index + lockdown',
            'params': {k: float(v) for k, v in res1.params.items()},
            'pvalues': {k: float(v) for k, v in res1.pvalues.items()},
            'conf_int_95': {k: [float(v[0]), float(v[1])] for k, v in res1.conf_int().iterrows()},
            'notes': 'Random intercept for city; ML estimation (reml=False). The coefficient on lockdown is the focal estimate.'
        }
    }

    if res2 is not None:
        out['model_2'] = {
            'formula': 'CMI ~ day_index + lockdown + completeLockdown',
            'params': {k: float(v) for k, v in res2.params.items()},
            'pvalues': {k: float(v) for k, v in res2.pvalues.items()},
            'conf_int_95': {k: [float(v[0]), float(v[1])] for k, v in res2.conf_int().iterrows()},
            'notes': 'Includes intensity (completeLockdown) as incremental effect relative to lockdown.'
        }

    # Save JSON results
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    # Save text summary for human inspection
    with open(RESULTS_TXT, 'w') as f:
        f.write('MODEL 1 SUMMARY (CMI ~ day_index + lockdown)\n')
        f.write(str(res1.summary()))
        f.write('\n\n')
        if res2 is not None:
            f.write('MODEL 2 SUMMARY (CMI ~ day_index + lockdown + completeLockdown)\n')
            f.write(str(res2.summary()))
            f.write('\n')


if __name__ == '__main__':
    main()
