import pandas as pd
import numpy as np
from scipy import stats
import os

# Paths (assume /app/data will be mounted to contain the CSV at runtime)
input_path = '/app/data/ohtsubo data.csv'
output_path = '/app/data/analysis_ohtsubo_results.txt'

# Read data
df = pd.read_csv(input_path)

# Recode a4 to a4r (reverse 1-7 -> 7->1 etc)
if 'a4' in df.columns:
    df['a4r'] = 8 - df['a4']
else:
    raise KeyError('a4 not found in dataset')

# Compute atot as sum of a1..a6 (use recomputed a4r)
items = ['a1','a2','a3','a4r','a5','a6']
for col in items:
    if col not in df.columns:
        raise KeyError(f"Required column {col} not found in data")

df['atot_recalc'] = df[items].sum(axis=1)

# Compare to existing atot if present
if 'atot' in df.columns:
    diff = (df['atot'] - df['atot_recalc']).abs().sum()
else:
    df['atot'] = df['atot_recalc']
    diff = 0

# Prepare groups based on condition (1 vs 0)
if 'condition' not in df.columns:
    raise KeyError('condition column not found')

g1 = df.loc[df['condition'] == 1, 'atot_recalc'].dropna()
g0 = df.loc[df['condition'] == 0, 'atot_recalc'].dropna()

n1 = len(g1)
n0 = len(g0)
mean1 = g1.mean()
mean0 = g0.mean()
sd1 = g1.std(ddof=1)
sd0 = g0.std(ddof=1)

# Perform t-test (Welch's t-test)
t_result = stats.ttest_ind(g1, g0, equal_var=False)

# Compute Welch-Satterthwaite df
var1 = g1.var(ddof=1)
var0 = g0.var(ddof=1)
se = np.sqrt(var1/n1 + var0/n0)
df_welch = (var1/n1 + var0/n0)**2 / ((var1**2)/((n1**2)*(n1-1)) + (var0**2)/((n0**2)*(n0-1)))

# Compute Cohen's d using pooled SD
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1 + n0 - 2))
cohens_d = (mean1 - mean0) / pooled_sd

# Write results
with open(output_path, 'w') as f:
    f.write('Recalculated atot vs existing atot absolute diff sum: {:.4f}\n'.format(diff))
    f.write('Group 1 (condition=1): n={} mean={:.3f} sd={:.3f}\n'.format(n1, mean1, sd1))
    f.write('Group 0 (condition=0): n={} mean={:.3f} sd={:.3f}\n'.format(n0, mean0, sd0))
    f.write('\n')
    f.write('Welch t-test results:\n')
    f.write('t = {:.4f}, p = {:.6f}, df_approx = {:.2f}\n'.format(t_result.statistic, t_result.pvalue, df_welch))
    f.write('Cohen\'s d (pooled sd) = {:.3f}\n'.format(cohens_d))

print('Analysis complete. Results written to', output_path)
