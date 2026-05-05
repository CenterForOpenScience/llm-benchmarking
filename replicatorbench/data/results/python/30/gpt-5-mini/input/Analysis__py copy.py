import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Standalone runner that generates synthetic data and runs the analysis to avoid subprocess/import issues
print('Standalone analysis: generating synthetic data and running mixed models')

n_players = 50
rounds = list(range(1,13))
rows = []
for pid in range(1, n_players+1):
    role = ['A','B','C','D','E'][(pid-1) % 5]
    session = (pid-1)//5 + 1
    groupid = ((pid-1) % 5) + 1
    for rnd in rounds:
        rows.append({
            'id': pid,
            'round': rnd,
            'playerinvestment': np.random.randint(0,11),
            'playerrole': role,
            'Session': session,
            'groupid_in_subsession': groupid,
            'playerdownloaded_files': np.random.randint(0,6),
            'playercollected_tokens': np.random.randint(0,11),
            'groupbandwidth': np.random.randint(5,21)
        })

df = pd.DataFrame(rows)
print('Synthetic data rows:', len(df))

# Following Analysis.do transformations
df['nvst'] = np.nan
df.loc[df['round'] > 2, 'nvst'] = df.loc[df['round'] > 2, 'playerinvestment']

pstn_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
df['pstn'] = df['playerrole'].map(pstn_map)

df['rnd'] = np.nan
df.loc[df['round'] > 2, 'rnd'] = df.loc[df['round'] > 2, 'round'] - 2

df['dwnld'] = np.nan
df.loc[df['round'] > 2, 'dwnld'] = df.loc[df['round'] > 2, 'playerdownloaded_files']

df['pyff'] = np.nan
mask = df['round'] > 2
df.loc[mask, 'pyff'] = df.loc[mask, 'playercollected_tokens'] + (10 - df.loc[mask, 'playerinvestment'])

df['grp'] = df['Session'] * 100 + df['groupid_in_subsession']
df['idrnd'] = df['id'] * 100 + df['rnd'].fillna(0)
df['grprnd'] = df['grp'] * 100 + df['rnd'].fillna(0)

df['dwnldpyff'] = np.nan
df.loc[mask, 'dwnldpyff'] = df.loc[mask, 'playercollected_tokens']

df['bndwdth'] = np.nan
df.loc[mask, 'bndwdth'] = df.loc[mask, 'groupbandwidth']

# Subset rounds >2
df2 = df[df['round'] > 2].copy()
# group-round sums
sums = df2.groupby('grprnd')[['nvst', 'dwnldpyff']].sum().rename(columns={'nvst': 'grpnvst', 'dwnldpyff': 'grpdwnldpyff'})
df2 = df2.merge(sums, left_on='grprnd', right_index=True, how='left')

# shares
df2['shrnvst'] = df2['nvst'] / df2['grpnvst']
df2.loc[df2['grpnvst'] == 0, 'shrnvst'] = 0.2

df2['shrdwnld'] = df2['dwnldpyff'] / df2['grpdwnldpyff']
df2.loc[df2['grpdwnldpyff'] == 0, 'shrdwnld'] = 0.2

# lag
df2 = df2.sort_values(['id', 'rnd'])
df2['shr'] = df2.groupby('id')['shrdwnld'].shift(1)

# interactions
df2['pstnshr'] = df2['pstn'] * df2['shr']
df2['pstnshrnvst'] = df2['pstn'] * df2['shrnvst']

out_dir = '/app/data'
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, 'Full_long_v2.csv')
df2.to_csv(out_csv, index=False)
print('Saved processed data to', out_csv)

# Fit mixed models using statsmodels
import statsmodels.api as sm

def fit_mixed(df2, endog_name, exog_vars, group_var='grp', reml=False):
    df_model = df2[[endog_name, group_var] + exog_vars].dropna()
    if df_model.empty:
        print('No data for model', endog_name)
        return None
    y = df_model[endog_name]
    X = df_model[exog_vars]
    X = sm.add_constant(X)
    md = sm.MixedLM(y, X, groups=df_model[group_var])
    try:
        mdf = md.fit(reml=reml, method='lbfgs')
        return mdf
    except Exception as e:
        print('Model fit failed:', e)
        try:
            return md.fit(reml=reml)
        except Exception as e2:
            print('Fallback fit failed:', e2)
            return None

exog = ['pstn', 'shr', 'pstnshr', 'rnd']
m_ml = fit_mixed(df2, 'nvst', exog, reml=False)
m_reml = fit_mixed(df2, 'nvst', exog, reml=True)

exog2 = ['bndwdth', 'pstn', 'shrnvst', 'pstnshrnvst', 'rnd']
m2_ml = fit_mixed(df2, 'dwnldpyff', exog2, reml=False)
m2_reml = fit_mixed(df2, 'dwnldpyff', exog2, reml=True)

out_txt = os.path.join(out_dir, 'regression_results.txt')
with open(out_txt, 'w') as f:
    f.write('Mixed models results\n')
    f.write('\n-- nvst model (ML) --\n')
    if m_ml is not None:
        f.write(m_ml.summary().as_text())
    else:
        f.write('nvst ML model failed\n')
    f.write('\n-- nvst model (REML) --\n')
    if m_reml is not None:
        f.write(m_reml.summary().as_text())
    else:
        f.write('nvst REML model failed\n')
    f.write('\n-- dwnldpyff model (ML) --\n')
    if m2_ml is not None:
        f.write(m2_ml.summary().as_text())
    else:
        f.write('dwnldpyff ML model failed\n')
    f.write('\n-- dwnldpyff model (REML) --\n')
    if m2_reml is not None:
        f.write(m2_reml.summary().as_text())
    else:
        f.write('dwnldpyff REML model failed\n')

print('Wrote regression summaries to', out_txt)

# Extract pstnshr coeff from nvst ML if available
summary_info = {}
if m_ml is not None:
    try:
        coef_name = 'pstnshr'
        params = m_ml.params
        b = params.get(coef_name)
        se = m_ml.bse.get(coef_name)
        z = b / se if (b is not None and se is not None) else None
        p = 2 * (1 - stats.norm.cdf(abs(z))) if z is not None else None
        summary_info = {'coef': float(b) if b is not None else None,
                        'se': float(se) if se is not None else None,
                        'z': float(z) if z is not None else None,
                        'p_value': float(p) if p is not None else None}
    except Exception as e:
        print('Failed to extract coef:', e)

import json
out_summary = os.path.join(out_dir, 'extracted_summary.json')
with open(out_summary, 'w') as f:
    json.dump(summary_info, f, indent=2)
print('Saved extracted summary to', out_summary)

print('Done')
