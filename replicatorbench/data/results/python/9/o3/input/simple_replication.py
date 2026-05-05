"""Simplified replication of Andrews & Money economic dispersion model.

The original algorithm involves complex consecutive seat-share rules and PCA.
This script implements a lightweight approximation that should be sufficient
for testing the hypothesis while avoiding heavy pandas operations that crash
under qemu emulation in the sandbox.

Steps:
1. Load CPDS_final.dta (country-year electoral-system data) and CMP_final.dta
   (party manifestos with seat counts & policy categories) using pyreadstat.
   Both files are small (<5000 rows) so full in-memory load is fine.
2. Merge on country/year.
3. Identify parties with seat share > 1% in the CURRENT election only (ignores
   the consecutive-election rule for simplicity).
4. Compute an economic left–right score as the difference between selected CMP
   right categories (free market, etc.) and left categories (welfare, etc.).
5. For each country-year, calculate dispersion = max(econ) - min(econ).
6. Create ln_parties = log(number of qualifying parties), ln_dispersion = log
   dispersion, control single_member (prop==0) and lagged_dispersion.
7. Run OLS with country-clustered SEs using statsmodels.
8. Save the coefficient table to replication_results.csv.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Limit threads to minimise segfault risk under qemu
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[simple_replication] DATA_DIR = {DATA_DIR}")

# Helper to read dta via pyreadstat or fallback
def read_dta(path):
    try:
        import pyreadstat
        df, _ = pyreadstat.read_dta(path, apply_value_formats=False)
        return df
    except Exception as e:
        print(f"pyreadstat failed ({e}), falling back to pandas.read_stata", flush=True)
        return pd.read_stata(path, convert_categoricals=False)

# Load datasets
cpds = read_dta(os.path.join(DATA_DIR, 'CPDS_final.dta'))[['country','year','prop']]
print('CPDS rows:', len(cpds))

cmp_needed = ['countryname','edate','party','absseat','totseats'] + [f'per{c}' for c in [303,401,402,403,404,407,412,413,414,504,505,701]]
cmp = read_dta(os.path.join(DATA_DIR, 'CMP_final.dta'))[cmp_needed]
print('CMP rows:', len(cmp))

# Harmonise names
cmp = cmp.rename(columns={'countryname':'country'})
cmp['year'] = pd.to_datetime(cmp['edate']).dt.year.astype(str)

# Compute seat share within election
cmp['seat_share'] = cmp['absseat'] / cmp['totseats']

# Keep parties with >1% seat share in THIS election
cmp_keep = cmp[cmp['seat_share'] > 0.01].copy()

# Economic score (simple index)
right_vars = [401,402,403,404,303]
left_vars = [407,412,413,504,701]
for v in right_vars + left_vars:
    col = f'per{v}'
    cmp_keep[col] = cmp_keep[col].fillna(0)

cmp_keep['econ_score'] = cmp_keep[[f'per{v}' for v in right_vars]].sum(axis=1) - cmp_keep[[f'per{v}' for v in left_vars]].sum(axis=1)

# Aggregate to country-year level
agg = cmp_keep.groupby(['country','year']).agg(
    disp = ('econ_score', lambda x: x.max() - x.min()),
    n_parties = ('party','nunique')
).reset_index()

# Log transforms
agg['ln_dispersion'] = np.log(agg['disp'].replace(0, np.nan))
agg['ln_parties'] = np.log(agg['n_parties'])

# Merge with CPDS to get single_member control
panel = pd.merge(agg, cpds, on=['country','year'], how='left')
panel['single_member'] = (panel['prop'] == 0).astype(int)

# Sort for lag
panel = panel.sort_values(['country','year'])
panel['lagged_dispersion'] = panel.groupby('country')['ln_dispersion'].shift(1)

# Drop missing rows
panel_clean = panel.dropna(subset=['ln_dispersion','ln_parties','lagged_dispersion'])
print('Panel rows for regression:', len(panel_clean))

# Regression
model = smf.ols('ln_dispersion ~ ln_parties + single_member + lagged_dispersion', data=panel_clean).fit(cov_type='cluster', cov_kwds={'groups':panel_clean['country']})
print(model.summary())

# Save results
out_path = os.path.join(DATA_DIR, 'replication_results.csv')
res_df = pd.DataFrame({'coef': model.params, 'se': model.bse, 'pval': model.pvalues})
res_df.to_csv(out_path, index_label='variable')
print('Saved results to', out_path)
