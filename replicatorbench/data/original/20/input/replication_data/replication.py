import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Read deidentified dataset
DF = pd.read_csv("replication_data/Kachanoff_Survey_deidentify.csv")

# Adjust scoring as in R script where necessary
# BAI 1-9 were 1-4 in example; deidentified appears float but already in 1-4; R script subtracted 1
bai_cols = [c for c in DF.columns if c.startswith('bai') or c=='attention_check1']
for c in bai_cols:
    DF[c] = DF[c] - 1

# Create composites
realistic_cols = [c for c in DF.columns if c.startswith('covid_real')]
symbolic_cols = [c for c in DF.columns if c.startswith('covid_symbolic')]
DF['Realistic'] = DF[realistic_cols].mean(axis=1)
DF['Symbolic'] = DF[symbolic_cols].mean(axis=1)

intrusion_cols = [c for c in DF.columns if c.startswith('intrusion')]
avoid_cols = [c for c in DF.columns if c.startswith('avoid')]
DF['Intrusion'] = DF[intrusion_cols].sum(axis=1)
DF['Avoidance'] = DF[avoid_cols].sum(axis=1)

swls_cols = [c for c in DF.columns if c.startswith('swls')]
DF['SWLS'] = DF[swls_cols].mean(axis=1)

positive_cols = [c for c in DF.columns if c.startswith('positive')]
negative_cols = [c for c in DF.columns if c.startswith('negative')]
DF['Positive'] = DF[positive_cols].sum(axis=1)
DF['Negative'] = DF[negative_cols].sum(axis=1)

social_cols = [c for c in DF.columns if c.startswith('social')]
DF['Social'] = DF[social_cols].sum(axis=1)

sds_cols = [c for c in DF.columns if c.startswith('sds')]
# reverse sds1 and sds2 as in R script
if 'sds1' in DF.columns:
    DF['sds1'] = 8 - DF['sds1']
if 'sds2' in DF.columns:
    DF['sds2'] = 8 - DF['sds2']
DF['SDS'] = DF[sds_cols].mean(axis=1)

behave_norm_cols = [c for c in DF.columns if c.startswith('behave_norm')]
behave_american_cols = [c for c in DF.columns if c.startswith('behave_american')]
if behave_norm_cols:
    DF['Norms'] = DF[behave_norm_cols].mean(axis=1)
if behave_american_cols:
    DF['American'] = DF[behave_american_cols].mean(axis=1)

# identify timepoints: use participant_id (deidentified file) similar to PROLIFIC_PID in example
if 'participant_id' in DF.columns:
    pid = 'participant_id'
else:
    pid = 'PROLIFIC_PID'

# sort by created to ensure time1 first
DF = DF.sort_values(by='created')

# create time2 indicator: duplicated pid marks second occurrence
DF['time2'] = DF.duplicated(subset=pid)

# select relevant variables and split
variables = [pid, 'created', 'BAI_total' if 'BAI_total' in DF.columns else None, 'Realistic', 'Symbolic',
             'Intrusion', 'Avoidance', 'SWLS', 'Positive', 'Negative', 'Social', 'SDS', 'Norms', 'American', 'handwashing']
variables = [v for v in variables if v is not None and v in DF.columns]
DFreduced = DF[variables + ['created', pid]] if pid not in variables else DF[variables]

# Ensure we have necessary columns
if pid not in DFreduced.columns:
    DFreduced[pid] = DF[pid]

# create time1 and time2 subsets
DFreduced = DFreduced.copy()
DFreduced['time2'] = DF.duplicated(subset=pid)
DFtime1 = DFreduced[DFreduced['time2'] == False].copy()
DFtime2 = DFreduced[DFreduced['time2'] == True].copy()

# Merge by pid
# If participant_id is object, ensure types match
DFtime1 = DFtime1.rename(columns={col: col + '.x' for col in DFreduced.columns if col not in [pid, 'time2']})
DFtime2 = DFtime2.rename(columns={col: col + '.y' for col in DFreduced.columns if col not in [pid, 'time2']})

# merge on pid
merge_key = pid
try:
    DFwide = pd.merge(DFtime1, DFtime2, left_on=merge_key, right_on=merge_key)
except Exception as e:
    # If merge fails due to types, coerce to str
    DFtime1[merge_key] = DFtime1[merge_key].astype(str)
    DFtime2[merge_key] = DFtime2[merge_key].astype(str)
    DFwide = pd.merge(DFtime1, DFtime2, left_on=merge_key, right_on=merge_key)

# Now run regression: Negative.y ~ Realistic.x + Symbolic.x
if 'Negative.y' not in DFwide.columns:
    # try to find negative columns in .y
    neg_y = [c for c in DFwide.columns if c.startswith('Negative') and c.endswith('.y')]
    if neg_y:
        neg_col = neg_y[0]
    else:
        # fallback: use 'Negative' from time2 if exists
        neg_col = 'Negative.y' if 'Negative.y' in DFwide.columns else 'Negative'
else:
    neg_col = 'Negative.y'

real_col = None
sym_col = None
for c in DFwide.columns:
    if c.startswith('Realistic') and c.endswith('.x'):
        real_col = c
    if c.startswith('Symbolic') and c.endswith('.x'):
        sym_col = c

# fallback if not found
if real_col is None and 'Realistic.x' in DFwide.columns:
    real_col = 'Realistic.x'
if sym_col is None and 'Symbolic.x' in DFwide.columns:
    sym_col = 'Symbolic.x'

# If still None, try without suffix
if real_col is None and 'Realistic' in DFwide.columns:
    real_col = 'Realistic'
if sym_col is None and 'Symbolic' in DFwide.columns:
    sym_col = 'Symbolic'

results = {
    'success': False,
    'notes': ''
}

if real_col is None or sym_col is None or neg_col is None:
    results['notes'] = f'Columns not found. real={real_col}, sym={sym_col}, neg={neg_col}.'
else:
    # prepare data for regression
    reg_df = DFwide[[neg_col, real_col, sym_col]].dropna()
    Y = reg_df[neg_col]
    X = reg_df[[real_col, sym_col]]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    coef = model.params.get(real_col, None)
    se = model.bse.get(real_col, None)
    pval = model.pvalues.get(real_col, None)
    conf_int = model.conf_int().loc[real_col].tolist() if real_col in model.params.index else [None, None]
    results['success'] = True
    results['model_summary'] = model.summary().as_text()
    results['coef_realistic'] = float(coef)
    results['se_realistic'] = float(se)
    results['pval_realistic'] = float(pval)
    results['confint_realistic'] = conf_int

# Save execution result to JSON# Save execution result to JSON
with open('execution_result.json', 'w') as f:
    json.dump(results, f)
with open('replication_output.json', 'w') as f:
    json.dump(results, f)

print('Done. Results saved to execution_result.json')
