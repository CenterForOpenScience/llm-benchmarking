#!/usr/bin/env python3
"""
Replication analysis for Kachanoff et al. focal claim:
"Realistic threat (T1) is positively associated with negative affect experienced over the subsequent week (T2)."

This script replicates the focal test using the de-identified replication dataset.

Steps:
1. Load de-identified dataset (CSV).
2. Score composite variables:
   - Realistic threat: mean of covid_real1-covid_real5 per row.
   - Negative affect: sum of negative1-negative10 per row.
3. Identify each participant's first (T1) and second (T2) survey response based on the `created` time stamp.
4. Build a wide data set with one row per participant containing Realistic_T1 and Negative_T2.
5. Estimate linear regression: Negative_T2 ~ Realistic_T1 (optionally include Symbolic_T1 as control).
6. Output coefficient, standard error, t-value, p-value.

All paths are relative to /app/data so that the dockerised execution works.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Constants
# Determine data location relative to this script to avoid path issues inside container
DATA_PATH = Path(__file__).resolve().parent / 'Kachanoff_Survey_deidentify.csv'

REALISTIC_COLS = [f'covid_real{i}' for i in range(1, 6)]
SYMBOLIC_COLS = [f'covid_symbolic{i}' for i in range(1, 6)]
NEGATIVE_COLS = [f'negative{i}' for i in range(1, 11)]
ID_COL = 'participant_id'
TIME_COL = 'created'

# 1. Load data
print('\nLoading data ...')
df = pd.read_csv(DATA_PATH)
print(f'Original rows: {len(df)}')

# Ensure timestamp parsing
print('Parsing date column ...')
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')

# 2. Score composites
print('Scoring composite variables ...')
# Realistic threat (mean)
df['Realistic'] = df[REALISTIC_COLS].mean(axis=1, skipna=True)
# Symbolic threat (mean) – might be useful as control
if set(SYMBOLIC_COLS).issubset(df.columns):
    df['Symbolic'] = df[SYMBOLIC_COLS].mean(axis=1, skipna=True)
else:
    df['Symbolic'] = np.nan
# Negative affect (sum)
df['Negative'] = df[NEGATIVE_COLS].sum(axis=1, skipna=True)

# 3. Sort by time to identify waves
print('Sorting within participant ...')
df = df.sort_values([ID_COL, TIME_COL])

def mark_wave(g):
    """Mark first occurrence as 1 (time1) and second as 2 (time2)."""
    g = g.copy()
    g['wave'] = np.arange(1, len(g) + 1)
    return g

df = df.groupby(ID_COL, group_keys=False).apply(mark_wave)

# Keep only participants with at least 2 waves
wide_records = []
for pid, g in df.groupby(ID_COL):
    if g.shape[0] < 2:
        continue  # not enough waves
    # extract time1 (wave==1) and time2 (wave==2)
    t1 = g.loc[g['wave'] == 1].iloc[0]
    t2 = g.loc[g['wave'] == 2].iloc[0]
    wide_records.append({
        'participant_id': pid,
        'Realistic_T1': t1['Realistic'],
        'Symbolic_T1': t1['Symbolic'],
        'Negative_T2': t2['Negative']
    })

wide = pd.DataFrame(wide_records)
print(f'Participants with 2 waves: {len(wide)}')

# Drop rows with missing key values
wide = wide.dropna(subset=['Realistic_T1', 'Negative_T2'])
print(f'After dropping missing: {len(wide)}')

# 4. Regression
print('\nRunning OLS regression ...')
X = sm.add_constant(wide['Realistic_T1'])
model = sm.OLS(wide['Negative_T2'], X).fit()
print(model.summary())
# Print key coefficient only
coef = model.params['Realistic_T1']
se = model.bse['Realistic_T1']
pval = model.pvalues['Realistic_T1']
print(f"\nKey coefficient Realistic_T1: {coef:.3f} (SE={se:.3f}, p={pval:.3g})")


# 5. Save results to /app/data
out_path = Path('/app/data/replication_results.txt')
with open(out_path, 'w') as f:
    f.write(model.summary().as_text())
print(f'\nResults written to {out_path}\n')