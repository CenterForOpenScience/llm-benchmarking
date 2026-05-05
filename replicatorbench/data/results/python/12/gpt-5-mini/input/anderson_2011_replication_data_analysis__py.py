# Python translation of anderson_2011_replication_data_analysis.do
# Uses statsmodels to run regressions similar to Stata's regress with cluster-robust SEs
import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Candidate data paths
CANDIDATES = [
    '/app/data/replication_data/analysis_data.dta',
    '/app/data/data/original/12/0205_gpt5-mini/replication_data/analysis_data.dta',
    '/workspace/replication_data/analysis_data.dta'
]
OUTPUT_DIR = '/app/data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = None
for p in CANDIDATES:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError('Could not find analysis_data.dta. Checked: ' + ','.join(CANDIDATES))

print('Using data file at', DATA_PATH)

# Load data
df = pd.read_stata(DATA_PATH)

# Helper to fit OLS and then compute cluster-robust results with aligned groups
def ols_clustered(formula, data, cluster_col):
    # Fit OLS (this will drop NA rows)
    res = smf.ols(formula, data=data).fit()
    # Align groups to the rows used in the model
    try:
        row_idx = res.model.data.row_labels
    except Exception:
        # fallback: use index
        row_idx = res.model.data.orig_endog.index
    groups = data.loc[row_idx, cluster_col]
    # Compute clustered standard errors
    robust = res.get_robustcov_results(cov_type='cluster', groups=groups)
    return robust

# Regression 1: final analysis (raw_inc_per_acre)
formula1 = 'raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode):caste + C(stcode) + caste'
model1 = ols_clustered(formula1, df, 'vill_id')

# Save summary to file
with open(os.path.join(OUTPUT_DIR, 'anderson_analysis_raw_inc_per_acre.txt'), 'w') as f:
    f.write(model1.summary().as_text())

# Regression 2: net_inc_per_acre
formula2 = 'net_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode):caste + C(stcode) + caste'
model2 = ols_clustered(formula2, df, 'vill_id')
with open(os.path.join(OUTPUT_DIR, 'anderson_analysis_net_inc_per_acre.txt'), 'w') as f:
    f.write(model2.summary().as_text())

# Regression 3: subset for stcode in (2,15)
subset = df[df['stcode'].isin([2,15])]
formula3 = 'raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode):caste + C(stcode) + caste'
model3 = ols_clustered(formula3, subset, 'vill_id')
with open(os.path.join(OUTPUT_DIR, 'anderson_analysis_raw_inc_per_acre_UPB_subset.txt'), 'w') as f:
    f.write(model3.summary().as_text())

print('Analyses complete. Outputs saved to', OUTPUT_DIR)
