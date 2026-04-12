import json
import glob
import pandas as pd
import numpy as np
from scipy import stats
import re

# Read all JSON files in replication_data/fake and replication_data
files = glob.glob('replication_data/fake/*.json') + glob.glob('replication_data/*.json')
files = [f for f in files if 'DS_Store' not in f]

records = []
for f in files:
    try:
        with open(f,'r') as fh:
            d = json.load(fh)
            if isinstance(d, list):
                for rec in d:
                    rec['_source_file'] = f
                    records.append(rec)
            elif isinstance(d, dict):
                d['_source_file'] = f
                records.append(d)
    except Exception:
        continue

if len(records) == 0:
    print('No records found')
    exit(1)

# Normalize
df = pd.json_normalize(records)

# Survey filtering
survey = df[df.get('trial_type')=='survey-multi-choice'].copy()

def extract_no(row):
    # row may be a Series
    resp = None
    if isinstance(row, pd.Series):
        resp = row.get('response')
    elif isinstance(row, dict):
        resp = row.get('response')
    if isinstance(resp, dict):
        for k, v in resp.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, str) and it.strip().lower().startswith('no'):
                        return True
            elif isinstance(v, str):
                if v.strip().lower().startswith('no'):
                    return True
    elif isinstance(resp, list):
        for it in resp:
            if isinstance(it, str) and it.strip().lower().startswith('no'):
                return True
    elif isinstance(resp, str):
        if resp.strip().lower().startswith('no'):
            return True
    return False

if not survey.empty:
    survey = survey.copy()
    survey['no_external'] = survey.apply(lambda r: extract_no(r), axis=1)
    good_subjects = survey[survey['no_external']==True].get('_source_file', pd.Series(dtype=object)).apply(lambda x: x.split('/')[-1].split('-')[1].split('.')[0] if isinstance(x,str) and '-' in x else x).tolist()
else:
    good_subjects = []

# Build data for image-button-response
img = df[df.get('trial_type')=='image-button-response'].copy()

# Extract bee_group info robustly
def extract_bee_cols(df_img):
    if df_img.empty:
        return pd.DataFrame({ 'antennae':[], 'wings':[], 'pattern':[], 'legs':[]})
    # Case 1: column 'bee_group' exists and contains dicts
    if 'bee_group' in df_img.columns and df_img['bee_group'].apply(lambda x: isinstance(x, dict)).any():
        bee = pd.json_normalize(df_img['bee_group']).rename(columns=lambda c: c)
        return bee
    # Case 2: flattened columns like 'bee_group.antennae'
    bee_cols = [c for c in df_img.columns if c.startswith('bee_group') and '.' in c]
    if bee_cols:
        bee = df_img[bee_cols].copy()
        bee.columns = [c.split('.',1)[1] for c in bee.columns]
        return bee
    # Case 3: direct columns present
    direct = [c for c in ['antennae','wings','pattern','legs'] if c in df_img.columns]
    if direct:
        return df_img[direct].copy()
    # Default: empty columns filled with None
    return pd.DataFrame({ 'antennae':[None]*len(df_img), 'wings':[None]*len(df_img), 'pattern':[None]*len(df_img), 'legs':[None]*len(df_img)})

bee = extract_bee_cols(img)
# Ensure bee has required columns
for c in ['antennae','wings','pattern','legs']:
    if c not in bee.columns:
        bee[c] = None

img = img.reset_index(drop=True).join(bee.reset_index(drop=True))

# extract stimulus_code

def get_code(s):
    if pd.isna(s):
        return None
    try:
        m = re.search(r'([AB][12][SD][MF])', str(s))
        return m.group(1) if m else None
    except Exception:
        return None

img['stimulus_code'] = img['stimulus'].apply(get_code)
img['stimulus_antennae'] = img['stimulus_code'].str[0]
img['stimulus_wings'] = img['stimulus_code'].str[1]
img['stimulus_pattern'] = img['stimulus_code'].str[2]
img['stimulus_legs'] = img['stimulus_code'].str[3]

# convert response to boolean where possible

def resp_to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.number)) and not pd.isna(x):
        try:
            return bool(int(x))
        except Exception:
            return np.nan
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in ['1','0']:
            return bool(int(xs))
        if xs in ['true','false']:
            return xs=='true'
    return np.nan

img['response_bool'] = img['response'].apply(resp_to_bool)

# compute dangerous_bee
img['dangerous_bee'] = ((img['antennae']=='X') | (img['antennae']==img['stimulus_antennae'])) & \
                      ((img['wings']=='X') | (img['wings']==img['stimulus_wings'])) & \
                      ((img['pattern']=='X') | (img['pattern']==img['stimulus_pattern'])) & \
                      ((img['legs']=='X') | (img['legs']==img['stimulus_legs']))

# compute correct comparing response_bool to dangerous_bee
img['correct'] = img['response_bool'] == img['dangerous_bee']

# expand relevant_dimensions
img['dim1'] = img['relevant_dimensions'].apply(lambda x: x[0] if isinstance(x,list) and len(x)>=1 else None)
img['dim2'] = img['relevant_dimensions'].apply(lambda x: x[1] if isinstance(x,list) and len(x)>=2 else None)

# helper for 1D correctness

def correct_1d(row, dim):
    if pd.isna(dim) or dim is None:
        return False
    try:
        if dim=='antennae':
            return row['response_bool'] == (row['antennae']==row['stimulus_antennae'])
        if dim=='wings':
            return row['response_bool'] == (row['wings']==row['stimulus_wings'])
        if dim=='pattern':
            return row['response_bool'] == (row['pattern']==row['stimulus_pattern'])
        if dim=='legs':
            return row['response_bool'] == (row['legs']==row['stimulus_legs'])
    except Exception:
        return False
    return False

img['correct_1d_a'] = img.apply(lambda r: correct_1d(r, r['dim1']), axis=1)
img['correct_1d_b'] = img.apply(lambda r: correct_1d(r, r['dim2']), axis=1)

# Identify test phase
if 'action' in img.columns:
    img['phase'] = np.where(img['action'].isna(), 'Test', 'Learning')
else:
    img['phase'] = 'Learning'

# Compute test.summary
test = img[img['phase']=='Test']
if test.empty:
    print('No Test phase rows identified; attempting fallback: treat last 32 trials per subject as Test')
    if 'subject' in img.columns:
        def mark_last32(df_sub):
            df_sub = df_sub.copy()
            if len(df_sub)<=32:
                df_sub['phase'] = 'Test'
            else:
                df_sub = df_sub.sort_values('trial_index') if 'trial_index' in df_sub.columns else df_sub
                df_sub.loc[df_sub.index[-32:], 'phase'] = 'Test'
            return df_sub
        img = img.groupby('subject', group_keys=False).apply(mark_last32)
        test = img[img['phase']=='Test']

if test.empty:
    print('No test data available after fallback. Exiting gracefully.')
    pd.DataFrame().to_csv('replication_output_test_summary.csv', index=False)
    print('Wrote replication_output_test_summary.csv (empty)')
    exit(0)

# group by subject
if 'subject' in test.columns:
    test = test.copy()
    test['subject_id'] = test['subject']
else:
    test = test.copy()
    test['subject_id'] = test.get('_source_file', pd.Series([None]*len(test))).apply(lambda x: x.split('/')[-1].split('-')[1].split('.')[0] if isinstance(x,str) and '-' in x else None)

grp = test.groupby(['subject_id','condition']).agg(acc_2d = ('correct','mean'),
                                                 acc_1d_a = ('correct_1d_a','mean'),
                                                 acc_1d_b = ('correct_1d_b','mean')).reset_index()
grp['acc_1d'] = grp[['acc_1d_a','acc_1d_b']].max(axis=1)

# t-test acc_1d by condition
cont = grp[grp['condition']=='contingent']['acc_1d'].dropna()
full = grp[grp['condition']=='full-information']['acc_1d'].dropna()

print('N contingent:', len(cont), 'N full:', len(full))
if len(cont)==0 or len(full)==0:
    print('Insufficient data for t-test')
else:
    tstat, pval = stats.ttest_ind(cont, full, equal_var=True)
    print('t=',tstat, 'p=',pval)

# Save outputs
grp.to_csv('replication_output_test_summary.csv', index=False)
print('Wrote replication_output_test_summary.csv')
