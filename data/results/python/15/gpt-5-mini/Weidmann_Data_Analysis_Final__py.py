#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import warnings
warnings.filterwarnings('ignore')

# Paths# Paths
# Determine data path robustly: prefer relative to script location, fallback to /app/data mount
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(SCRIPT_DIR, 'Afghanistan_Election_Violence_2014.csv'),
    os.path.join('/app/data', 'original', '15', '0205_gpt5-mini', 'replication_data', 'Afghanistan_Election_Violence_2014.csv'),
    os.path.join('/workspace', 'replication_data', 'Afghanistan_Election_Violence_2014.csv')
]
DATA_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError('Could not find Afghanistan_Election_Violence_2014.csv in expected locations: ' + ','.join(possible_paths))

ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = SCRIPT_DIR

os.makedirs(OUT_DIR, exist_ok=True)os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
# Save dataset info
dataset_info = {
    'shape': df.shape,
    'columns': df.columns.tolist(),
}
with open(os.path.join(OUT_DIR, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f)

# Ensure key variables exist
needed = ['fraud', 'sigact_5r', 'sigact_60r', 'pcx', 'electric', 'pcexpend', 'dist', 'elevation']
for v in needed:
    if v not in df.columns:
        raise ValueError(f'Missing required column: {v}')

# Coerce fraud to binary if needed
if df['fraud'].dropna().isin([0,1]).all():
    df['fraud_bin'] = df['fraud']
else:
    df['fraud_bin'] = (df['fraud'] > 0).astype(int)

# Create squared terms
df['sigact_5r_sq'] = df['sigact_5r'] ** 2
df['sigact_60r_sq'] = df['sigact_60r'] ** 2

# Helper to run logit with cluster-robust SEs
def run_logit(formula, data, cluster_col=None):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    if cluster_col and cluster_col in data.columns:
        try:
            clusters = data[cluster_col]
            cov = sm.stats.sandwich_covariance.cov_cluster(model, clusters)
            se = np.sqrt(np.diag(cov))
            res = {'model': model, 'bse_cluster': se, 'cov_cluster': cov}
        except Exception as e:
            res = {'model': model, 'error': str(e)}
    else:
        res = {'model': model}
    return res

# Model specifications
controls = 'pcx + electric + pcexpend + dist + elevation'
form_5 = 'fraud_bin ~ sigact_5r + sigact_5r_sq + ' + controls
form_60 = 'fraud_bin ~ sigact_60r + sigact_60r_sq + ' + controls

# Prepare data: drop NA for each model separately and fit
res_outputs = {}
for name, form, sq in [('model_5', form_5, 'sigact_5r'), ('model_60', form_60, 'sigact_60r')]:
    mod_df = df[[ 'fraud_bin', 'sigact_5r', 'sigact_5r_sq', 'sigact_60r', 'sigact_60r_sq', 'pcx', 'electric', 'pcexpend', 'dist', 'elevation', 'regcom' ]].copy()
    mod_df = mod_df.dropna()
    res = run_logit(form, mod_df, cluster_col='regcom')
    model = res['model']
    # Summary text
    summ = model.summary2().as_text()
    with open(os.path.join(OUT_DIR, f'{name}_summary.txt'), 'w') as f:
        f.write(summ)
    # Coefs
    coefs = model.params
    bse = model.bse
    # If cluster robust se available, replace bse
    if 'bse_cluster' in res:
        bse = pd.Series(res['bse_cluster'], index=coefs.index)
    coefs_df = pd.DataFrame({'coef': coefs, 'se': bse, 'z': coefs / bse, 'p': model.pvalues})
    coefs_df.to_csv(os.path.join(OUT_DIR, f'{name}_coefs.csv'))
    res_outputs[name] = {'nobs': int(model.nobs), 'params': coefs_df.to_dict(orient='index')}

    # Margins: predicted probabilities across range of violence
    violence_col = 'sigact_5r' if name=='model_5' else 'sigact_60r'
    vmin = df[violence_col].min()
    vmax = df[violence_col].max()
    grid = np.linspace(vmin, vmax, 41)
    preds = []
    for v in grid:
        tmp = mod_df.copy()
        tmp[violence_col] = v
        tmp[f'{violence_col}_sq'] = v**2
        # set other covariates to their mean
        for c in ['pcx', 'electric', 'pcexpend', 'dist', 'elevation']:
            tmp[c] = tmp[c].mean()
        pred = model.predict(tmp)
        preds.append({'violence': v, 'pred_mean': float(pred.mean())})
    pd.DataFrame(preds).to_csv(os.path.join(OUT_DIR, f'margins_{violence_col}.csv'), index=False)

# Save a summary JSON
with open(os.path.join(OUT_DIR, 'execution_result.json'), 'w') as f:
    json.dump({'results': res_outputs}, f)

print('Analysis complete. Outputs saved to', OUT_DIR)
