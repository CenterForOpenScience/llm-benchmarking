# Replication analysis script translated from SPSS syntax
# Saves outputs to /app/data to comply with IO requirements

import os
import sys
import pandas as pd
import numpy as np
import pyreadstat
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

# Attempt to locate the .sav file in several plausible locations
possible_paths = [
    '/app/data/SCORE_all data.sav',
    '/app/data/SCORE_all_data.sav',
    '/app/data/data/original/39/input/replication_data/SCORE_all data.sav',
    'data/original/39/input/replication_data/SCORE_all data.sav'
]

sav_path = None
for p in possible_paths:
    if os.path.exists(p):
        sav_path = p
        break

if sav_path is None:
    print('ERROR: Could not find SCORE_all data.sav. Please place it at /app/data/SCORE_all data.sav or ensure path exists.')
    sys.exit(1)

print('Loading data from:', sav_path)
df, meta = pyreadstat.read_sav(sav_path)

# Work on a copy
data = df.copy()

# Helper to safely compute mean across columns ignoring NaNs
def safe_mean(row, cols):
    return row[cols].mean()

# Recode operations and compute scales according to SPSS syntax
# Example: NFCog recoding of specific items
recoded_map = { -4:4, -3:3, -2:2, -1:1, 0:0, 1:-1, 2:-2, 3:-3, 4:-4 }

# List of Cog items to recode
cog_rec_items = ['Cog5','Cog6','Cog8','Cog9','Cog10','Cog11','Cog13','Cog14','Cog15','Cog16','Cog17','Cog18','Cog20','Cog21','Cog22','Cog24','Cog25','Cog26','Cog31','Cog32','Cog33']
for it in cog_rec_items:
    out = f"{it}_rec"
    if it in data.columns:
        data[out] = data[it].map(recoded_map)
    else:
        data[out] = np.nan

# Compute NFCog mean
cog_items_all = ['Cog1','Cog2','Cog3','Cog4','Cog7','Cog12','Cog19','Cog23','Cog27','Cog28','Cog29','Cog30','Cog34'] + [f"{it}_rec" for it in cog_rec_items]
cog_present = [c for c in cog_items_all if c in data.columns]
data['NFCog'] = data[cog_present].mean(axis=1)

# IRI recode
iri_rec = ['IRI3','IRI4','IRI7','IRI12','IRI13','IRI14','IRI15','IRI18','IRI19']
iri_map = {5:1,4:2,3:3,2:4,1:5}
for it in iri_rec:
    out = f"{it}_rec"
    if it in data.columns:
        data[out] = data[it].map(iri_map)
    else:
        data[out] = np.nan

# Compute IRI total and subscales per syntax
iri_all_items = ['IRI1','IRI2','IRI5','IRI6','IRI8','IRI9','IRI10','IRI11','IRI16','IRI17','IRI20','IRI21','IRI22','IRI23','IRI24','IRI25','IRI26','IRI27','IRI28'] + [f"{it}_rec" for it in iri_rec]
iri_present = [c for c in iri_all_items if c in data.columns]
if len(iri_present)>0:
    data['IRI_ALL'] = data[iri_present].mean(axis=1)

# Compute IRI subscales if items present (EC, FS, PT, PD) - using lists from syntax
iri_ec = ['IRI2','IRI4_rec','IRI9','IRI14_rec','IRI18_rec','IRI20','IRI22']
iri_fs = ['IRI1','IRI5','IRI7_rec','IRI12_rec','IRI16','IRI23','IRI26']
iri_pt = ['IRI3_rec','IRI8','IRI11','IRI15_rec','IRI21','IRI25','IRI28']
iri_pd = ['IRI6','IRI10','IRI13_rec','IRI17','IRI19_rec','IRI24','IRI27']
for name, items in [('IRI_EC',iri_ec),('IRI_FS',iri_fs),('IRI_PT',iri_pt),('IRI_PD',iri_pd)]:
    present = [c for c in items if c in data.columns]
    if present:
        data[name] = data[present].mean(axis=1)

# Fear recoding and composites
fear_rec = ['Fear2','Fear3','Fear6','Fear10','Fear15','Fear17','Fear23']
fear_map = {5:1,4:2,3:3,2:4,1:5}
for it in fear_rec:
    out = f"{it}_rec"
    if it in data.columns:
        data[out] = data[it].map(fear_map)

# Fear scales
fear_all_items = ['Fear1','Fear4','Fear5','Fear7','Fear8','Fear9','Fear11','Fear12','Fear13','Fear14','Fear16','Fear18','Fear19','Fear20','Fear21','Fear22','Fear24','Fear25','Fear26','Fear27'] + [f"{it}_rec" for it in fear_rec]
fear_present = [c for c in fear_all_items if c in data.columns]
if fear_present:
    data['Fear_ALL'] = data[fear_present].mean(axis=1)

fear_ia = ['Fear1','Fear4','Fear5','Fear7','Fear8','Fear9','Fear11','Fear12','Fear13','Fear14','Fear2_rec','Fear3_rec','Fear6_rec','Fear10_rec','Fear15_rec']
fear_aa = ['Fear16','Fear18','Fear19','Fear20','Fear21','Fear22','Fear24','Fear25','Fear26','Fear27','Fear17_rec','Fear23_rec']
for name, items in [('Fear_IA',fear_ia),('Fear_AA',fear_aa)]:
    present = [c for c in items if c in data.columns]
    if present:
        data[name] = data[present].mean(axis=1)

# QfS_all
qfs_items = ['QfS1','QfS2','QfS3','QfS4','QfS5','QfS6']
qfs_present = [c for c in qfs_items if c in data.columns]
if qfs_present:
    data['QfS_ALL'] = data[qfs_present].mean(axis=1)

# Moral dilemmas training (pretest) recode: APP1..APP20 (1->0,2->1)
app_items = [f'APP{i}' for i in range(1,21)]
for it in app_items:
    if it in data.columns:
        data[it] = data[it].replace({1:0,2:1})

# Compute APP_incong and cong and RH_incong, RH_cong, U, D
incong = [f'APP{i}' for i in range(1,11)]
cong = [f'APP{i}' for i in range(11,21)]
present_incong = [c for c in incong if c in data.columns]
present_cong = [c for c in cong if c in data.columns]
if present_incong:
    data['APP_incong'] = data[present_incong].sum(axis=1)
else:
    data['APP_incong'] = np.nan
if present_cong:
    data['APP_cong'] = data[present_cong].sum(axis=1)
else:
    data['APP_cong'] = np.nan

if 'APP_incong' in data.columns and 'APP_cong' in data.columns:
    data['RH_incong'] = data['APP_incong'] / 10
    data['RH_cong'] = data['APP_cong'] / 10
    data['U'] = data['RH_cong'] - data['RH_incong']
    # Avoid division by zero
    data['D'] = data['RH_incong'] / (1 - data['U']).replace(0, np.nan)

# Main task recode APP_main items 1..50
app_main = [f'APP{i}_main' for i in range(1,51)]
for it in app_main:
    if it in data.columns:
        data[it] = data[it].replace({1:0,2:1})

# Compute APP_PMD (1-20), APP_NMD (21-35), APP_CMD (36-50)
app_pmd = [f'APP{i}_main' for i in range(1,21)]
app_nmd = [f'APP{i}_main' for i in range(21,36)]
app_cmd = [f'APP{i}_main' for i in range(36,51)]
for name, items in [('APP_PMD',app_pmd),('APP_NMD',app_nmd),('APP_CMD',app_cmd)]:
    present = [c for c in items if c in data.columns]
    if present:
        data[name] = data[present].sum(axis=1)

# APP_ALL and RHs
if 'APP_PMD' in data.columns and 'APP_NMD' in data.columns and 'APP_CMD' in data.columns:
    data['APP_ALL'] = data['APP_PMD'] + data['APP_NMD'] + data['APP_CMD']
    data['RH_PMD'] = data['APP_PMD'] / 20
    data['RH_NMD'] = data['APP_NMD'] / 15
    data['RH_CMD'] = data['APP_CMD'] / 15
    data['RH_ALL'] = data['APP_ALL'] / 50

# Confidence recode for CONF_main items and compute CONF_PMD, CONF_NMD, CONF_CMD as means
conf_items = [f'CONF{i}_main' for i in range(1,51)]
# The SPSS syntax maps (1=2) (4=2) (2=1) (3=1) INTO REC; that's odd but implement mapping
conf_map = {1:2,4:2,2:1,3:1}
for it in conf_items:
    if it in data.columns:
        data[f"{it}_REC"] = data[it].replace(conf_map)

# Compute CONF_PMD (means of first 20 REC), CONF_NMD next 15, CONF_CMD next 15
conf_pmd = [f'CONF{i}_main_REC' for i in range(1,21)]
conf_nmd = [f'CONF{i}_main_REC' for i in range(21,36)]
conf_cmd = [f'CONF{i}_main_REC' for i in range(36,51)]
if any(c in data.columns for c in conf_pmd):
    present = [c for c in conf_pmd if c in data.columns]
    data['CONF_PMD'] = data[present].mean(axis=1)
if any(c in data.columns for c in conf_nmd):
    present = [c for c in conf_nmd if c in data.columns]
    data['CONF_NMD'] = data[present].mean(axis=1)
if any(c in data.columns for c in conf_cmd):
    present = [c for c in conf_cmd if c in data.columns]
    data['CONF_CMD'] = data[present].mean(axis=1)

# Reaction times: RT1_corr ... RT50_corr
rt_items = [f'RT{i}_corr' for i in range(1,51)]
rt_pmd = [f'RT{i}_corr' for i in range(1,21)]
rt_nmd = [f'RT{i}_corr' for i in range(21,36)]
rt_cmd = [f'RT{i}_corr' for i in range(36,51)]
if any(c in data.columns for c in rt_pmd):
    present = [c for c in rt_pmd if c in data.columns]
    data['RT_PMD'] = data[present].mean(axis=1)
if any(c in data.columns for c in rt_nmd):
    present = [c for c in rt_nmd if c in data.columns]
    data['RT_NMD'] = data[present].mean(axis=1)
if any(c in data.columns for c in rt_cmd):
    present = [c for c in rt_cmd if c in data.columns]
    data['RT_CMD'] = data[present].mean(axis=1)

# RT_ALL
present_rt_all = [c for c in rt_items if c in data.columns]
if present_rt_all:
    data['RT_ALL'] = data[present_rt_all].mean(axis=1)

# Now conduct primary analyses: compare RH_PMD across Condition groups
# Ensure Condition coding: check unique values
if 'Condition' not in data.columns:
    print('Warning: Condition variable not found. Many analyses require Condition.')
else:
    # Inspect coding
    print('Condition unique values:', data['Condition'].unique())

# Prepare outputs
out_summary = {}

# Group means for RH_PMD, RH_NMD, RH_CMD
for var in ['RH_PMD','RH_NMD','RH_CMD','CONF_PMD','CONF_NMD','CONF_CMD','RT_PMD','RT_NMD','RT_CMD','LDT_RT_Warm','LDT_RT_Comp']:
    if var in data.columns:
        grp = data.groupby('Condition')[var].agg(['mean','std','count']).to_dict()
        out_summary[var] = grp

# T-tests: OBS vs CON for RH_PMD if present
if 'RH_PMD' in data.columns and 'Condition' in data.columns:
    # assume Condition coded 1 and 2 as in SPSS
    cond_vals = data['Condition'].unique()
    # filter missing
    g1 = data[data['Condition']==1]['RH_PMD'].dropna()
    g2 = data[data['Condition']==2]['RH_PMD'].dropna()
    if len(g1)>1 and len(g2)>1:
        tstat, pval = stats.ttest_ind(g1, g2, equal_var=False)
        out_summary['t_RH_PMD'] = {'t': float(tstat), 'p': float(pval), 'n1': int(len(g1)), 'n2': int(len(g2)), 'mean1': float(g1.mean()), 'mean2': float(g2.mean())}

# Mixed ANOVA (Condition x Dilemma Type) using pingouin
if all(x in data.columns for x in ['RH_PMD','RH_NMD','RH_CMD','Condition']):
    # Build long dataframe for rm ANOVA with one within factor 'dilemma_type' with 3 levels
    long = pd.melt(data.reset_index(), id_vars=['index','Condition'], value_vars=['RH_PMD','RH_NMD','RH_CMD'], var_name='dilemma_type', value_name='RH')
    # Rename index to subject
    long = long.rename(columns={'index':'Subject'})
    # Ensure Condition is categorical
    long['Condition'] = long['Condition'].astype('category')
    try:
        aov = pg.mixed_anova(dv='RH', within='dilemma_type', between='Condition', subject='Subject', data=long)
        out_summary['mixed_anova_RH'] = aov.to_dict()
    except Exception as e:
        out_summary['mixed_anova_RH_error'] = str(e)

# Save summary to /app/data
out_path = '/app/data/replication_summary.csv'
# Convert summary to a DataFrame-friendly form
rows = []
for k,v in out_summary.items():
    rows.append({'metric':k, 'value': str(v)})
summary_df = pd.DataFrame(rows)
# Ensure output dir exists
os.makedirs('/app/data', exist_ok=True)
summary_df.to_csv(out_path, index=False)
print('Saved replication summary to', out_path)

# Also save the processed data for inspection
processed_path = '/app/data/replication_processed_data.csv'
data.to_csv(processed_path, index=False)
print('Saved processed data to', processed_path)

print('Analysis completed.')
