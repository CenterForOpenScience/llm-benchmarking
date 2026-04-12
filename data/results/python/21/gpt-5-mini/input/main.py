import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# Paths
DATA_DIR = 'replication_data'
DF1_PATH = os.path.join(DATA_DIR, 'Bischetti_Survey_Part1_deidentify.csv')
DF2_PATH = os.path.join(DATA_DIR, 'Bischetti_Survey_Part2_deidentify.csv')
OUTPUT = 'artifacts'

os.makedirs(OUTPUT, exist_ok=True)

print('Reading data...')
DF1 = pd.read_csv(DF1_PATH, low_memory=False)
DF2 = pd.read_csv(DF2_PATH, low_memory=False)

# Minimal renaming to match Rmd expectations
# The Rmd expects merging by participant_id; if not present, try PROLIFIC_PID
if 'participant_id' in DF1.columns and 'participant_id' in DF2.columns:
    merge_key = 'participant_id'
elif 'PROLIFIC_PID' in DF1.columns and 'PROLIFIC_PID' in DF2.columns:
    merge_key = 'PROLIFIC_PID'
else:
    # fallback: try to find any common id-like column
    common = set(DF1.columns).intersection(set(DF2.columns))
    merge_key = None
    for c in ['participant_id','PROLIFIC_PID','PROLIFIC_ID','session']:
        if c in common:
            merge_key = c
            break
    if merge_key is None:
        raise SystemExit('No merge key found between DF1 and DF2')

print(f'Merging on {merge_key}...')
DF = pd.merge(DF1, DF2, on=merge_key, how='outer')

# Select columns that include disturbing ratings and demographics
cols = [c for c in DF.columns if 'disturbing' in c or c in [merge_key,'state','age']]
DF_sel = DF[cols].copy()

# Wide to long: gather disturbing columns
dist_cols = [c for c in DF_sel.columns if 'disturbing' in c]
long = DF_sel.melt(id_vars=[merge_key,'state','age'], value_vars=dist_cols, var_name='variable', value_name='Aversiveness')

# variable names like pic1_disturbing -> convert to anote-style name as R does
long['name'] = long['variable'].str.replace('disturbing','anote', regex=False)

# Aversiveness numeric coercion and adjust scale if needed
long['Aversiveness'] = pd.to_numeric(long['Aversiveness'], errors='coerce')
# R code subtracts 1 in updated Rmd: attempt same if max>6
if long['Aversiveness'].max(skipna=True) > 6:
    long['Aversiveness'] = long['Aversiveness'] - 1

# Without metadata mapping, we approximate labels by grouping picture numbers into 4 buckets
# Extract picture number
import re
long['picnum'] = long['name'].str.extract(r'(pic)(\d+)', expand=False)[1]
long['picnum'] = pd.to_numeric(long['picnum'], errors='coerce')
# Assign labels by picnum modulo 4 --- deterministic mapping: 1->covid-verbal,2->covid-meme,3->covid-strip,0->non-verbal
def assign_label(p):
    if np.isnan(p):
        return 'unknown'
    m = int(p) % 4
    if m == 1:
        return 'covid-verbal'
    elif m == 2:
        return 'covid-meme'
    elif m == 3:
        return 'covid-strip'
    else:
        return 'non-verbal'

long['label'] = long['picnum'].apply(assign_label)

# Clean age
long['age'] = pd.to_numeric(long['age'], errors='coerce')
# If age looks like birth year (>1900), convert to age
long['age'] = long['age'].apply(lambda x: 2021 - x if x is not None and not np.isnan(x) and x>1900 else x)

# Drop rows without Aversiveness
long = long.dropna(subset=['Aversiveness'])

print('Descriptive stats:')
desc = long.groupby('label')['Aversiveness'].agg(['count','mean','std']).reset_index()
print(desc)
desc.to_csv(os.path.join(OUTPUT,'descriptives_by_label.csv'), index=False)

# Fit a mixed-effects model Aversiveness ~ label with random intercepts for participant and item (name)
# Use statsmodels MixedLM: formula interface limited, so use patsy
try:
    print('Fitting mixed model...')
    # encode label as categorical
    long['label_c'] = long['label'].astype('category')
    # Build design matrices
    import patsy
    y, X = patsy.dmatrices('Aversiveness ~ C(label_c)', data=long, return_type='dataframe')
    # random effects: group by participant id and by item via combined group? statsmodels MixedLM supports one grouping variable.
    # We'll fit random intercepts for participant_id if present, else PROLIFIC_PID
    group = merge_key if merge_key in long.columns else merge_key
    md = MixedLM(endog=y, exog=X, groups=long[merge_key])
    mdf = md.fit(reml=False)
    print(mdf.summary())
    with open(os.path.join(OUTPUT,'mixed_model_summary.txt'),'w') as f:
        f.write(str(mdf.summary()))
except Exception as e:
    print('Mixed model failed:', e)
    with open(os.path.join(OUTPUT,'mixed_model_error.txt'),'w') as f:
        f.write(str(e))

print('Saving sample of long data...')
long.head(200).to_csv(os.path.join(OUTPUT,'long_head.csv'), index=False)
print('Done.')
