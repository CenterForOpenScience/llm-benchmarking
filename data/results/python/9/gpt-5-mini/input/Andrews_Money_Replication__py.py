# Andrews & Money (2009) replication translated from Stata .do to Python
# This script mirrors the steps in Andrews-Money_Replication.do
# Input files (assumed to be available under /app/data):
#  - /app/data/original/9/python/replication_data/CMP_final.dta
#  - /app/data/original/9/python/replication_data/CPDS_final.dta
# Outputs saved to /app/data:
#  - Andrews_Money_replication_data.csv
#  - Andrews_Money_replication_results.txt

import os
import numpy as np
import pandas as pd
# Using numpy SVD for PCA to avoid dependency on scikit-learn
import statsmodels.api as sm

# Paths (use /app/data as required by run policy; fallback to workspace if not mounted)
default_cmp = '/app/data/original/9/python/replication_data/CMP_final.dta'
default_cpds = '/app/data/original/9/python/replication_data/CPDS_final.dta'
workspace_cmp = '/workspace/replication_data/CMP_final.dta'
workspace_cpds = '/workspace/replication_data/CPDS_final.dta'
cmp_path = default_cmp if os.path.exists(default_cmp) else workspace_cmp
cpds_path = default_cpds if os.path.exists(default_cpds) else workspace_cpds
output_data_csv = '/app/data/Andrews_Money_replication_data.csv'
output_results = '/app/data/Andrews_Money_replication_results.txt'

# Read datasets
cmp = pd.read_stata(cmp_path)
cpds = pd.read_stata(cpds_path)

# Drop duplicates as in the .do file
cmp = cmp.drop_duplicates()
cpds = cpds.drop_duplicates()

# Rename countryname -> country to match do-file
if 'countryname' in cmp.columns:
    cmp = cmp.rename(columns={'countryname': 'country'})

# Generate year from edate (Stata date); pandas likely read edate as datetime
if 'edate' in cmp.columns:
    try:
        cmp['year'] = pd.DatetimeIndex(cmp['edate']).year
    except Exception:
        # if edate is numeric year already
        cmp['year'] = cmp['edate']
else:
    raise ValueError('edate not found in CMP dataset')

# Ensure cpds has a numeric "year" column (it does as loaded)
# Prepare election identifier (unique country-edate pairs)
election_identifier = cmp[['country', 'edate']].drop_duplicates().copy()
election_identifier = election_identifier.sort_values(['country', 'edate'])
election_identifier['election'] = election_identifier.groupby('country').cumcount() + 1

# Merge election identifier back into cmp
cmp = cmp.merge(election_identifier, on=['country', 'edate'], how='left')

# Merge CMP and CPDS on country and year (m:1 country year using CPDS_temp.dta)
# Before merging, ensure cpds has 'year' column numeric
if 'year' not in cpds.columns:
    raise ValueError('year not found in CPDS dataset')

# Merge on country and year
# In the Stata do-file, they convert year to string then merge; here we'll merge on numeric year
cmp['year'] = cmp['year'].astype(int)
cpds['year'] = cpds['year'].astype(int)
merged = cmp.merge(cpds, on=['country', 'year'], how='inner', suffixes=('_cmp', '_cpds'))

# The do-file then generates relative_seat = absseat/totseats and proceeds with party-level filtering
merged['relative_seat'] = merged['absseat'] / merged['totseats']

# We need to sort by party and election to compute leads
# Ensure 'party' and 'election' exist
if 'party' not in merged.columns or 'election' not in merged.columns:
    raise ValueError('Required variables party or election missing after merge')

merged = merged.sort_values(['party', 'election']).reset_index(drop=True)

# Create leads (relative_seat_t_1, relative_seat_t_2)
merged['relative_seat_t_1'] = merged.groupby('party')['relative_seat'].shift(-1)
merged['relative_seat_t_2'] = merged.groupby('party')['relative_seat'].shift(-2)

# First filter: include parties that have >=1% in at least two consecutive elections
merged['consecutive_temp1'] = (merged['relative_seat'] > 0.01) & (merged['relative_seat_t_1'] > 0.01)
# For each party, if any observation satisfies consecutive_temp1, keep all those party-election observations? In Stata they did by party: egen consecutive = max(consecutive_temp) then keep if consecutive==1 (keeps all rows for parties that ever satisfied)
party_consec1 = merged.groupby('party')['consecutive_temp1'].transform('max')
merged = merged[party_consec1 == True].copy()

# Second filter: drop a party if it fails to gain 1% in three subsequent elections
merged = merged.sort_values(['party', 'election']).reset_index(drop=True)
merged['relative_seat_t_1'] = merged.groupby('party')['relative_seat'].shift(-1)
merged['relative_seat_t_2'] = merged.groupby('party')['relative_seat'].shift(-2)
merged['consecutive_temp2'] = (merged['relative_seat'] > 0.01) & (merged['relative_seat_t_1'] > 0.01) & (merged['relative_seat_t_2'] > 0.01)
party_consec2 = merged.groupby('party')['consecutive_temp2'].transform('max')
merged = merged[party_consec2 == True].copy()

# Now compute number of parties in system by country-election (bysort country election: gen count_parties = _N)
count_parties = merged.groupby(['country', 'election']).size().reset_index(name='count_parties')
merged = merged.merge(count_parties, on=['country', 'election'], how='left')

# Log-transform count_parties (Stata: replace count_parties = log(count_parties))
merged['count_parties_log'] = np.log(merged['count_parties'])

# PCA on economic indicators: per303 per401 per402 per403 per404 per407 per412 per413 per414 per504 per505 per701
pca_vars = ['per303','per401','per402','per403','per404','per407','per412','per413','per414','per504','per505','per701']
# Drop rows missing any of the pca variables (Stata PCA excludes missing)
pca_df = merged.dropna(subset=pca_vars).copy()

# Standardize variables (mean zero) before PCA to better match Stata's PCA on correlation
X = pca_df[pca_vars].astype(float)
X_centered = (X - X.mean()) / X.std(ddof=0)

# Using numpy SVD for PCA (no scikit-learn)# Compute first principal component using numpy SVD
# X_centered is standardized; compute SVD
U, S, Vt = np.linalg.svd(X_centered.fillna(0).values, full_matrices=False)
# First principal component scores are the first column of U * S[0]
pca_scores = (U[:, 0] * S[0]).reshape(-1,1)

# Put scores back into merged frame (for rows used in PCA)
pca_df['economic_policy'] = pca_scores[:,0]
merged = merged.merge(pca_df[['country','election','party','economic_policy']], on=['country','election','party'], how='left')

# Identify most extreme parties per country-election and compute dispersion = max - min
dispersion_df = merged.groupby(['country','election']).agg(min_policy=('economic_policy','min'), max_policy=('economic_policy','max')).reset_index()
dispersion_df['dispersion'] = dispersion_df['max_policy'] - dispersion_df['min_policy']

# Log-transform dispersion (drop non-positive dispersions before log)
dispersion_df = dispersion_df[dispersion_df['dispersion'] > 0].copy()
dispersion_df['dispersion_log'] = np.log(dispersion_df['dispersion'])

# Merge dispersion into a country-election-level dataset
# Keep one unique observation per country-election with required vars as in do-file
# The do-file keeps: dispersion count_parties country edate single_member year election data_sample
# For single_member: gen single_member = (prop == 0); replace single_member = . if prop == .
# We'll take prop from merged/CPDS data (should be present in merged)
# Get a representative row per country-election to pull edate, year, prop, data_sample
rep_rows = merged.sort_values(['country','election']).groupby(['country','election']).first().reset_index()
rep = rep_rows[['country','election','edate','year','data_sample']].copy()
# Pull prop (from merged)
if 'prop' in merged.columns:
    prop_df = merged.groupby(['country','election'])['prop'].first().reset_index()
    rep = rep.merge(prop_df, on=['country','election'], how='left')
else:
    rep['prop'] = np.nan

# Merge count_parties
rep = rep.merge(count_parties, on=['country','election'], how='left')
# Merge dispersion
rep = rep.merge(dispersion_df[['country','election','dispersion','dispersion_log']], on=['country','election'], how='left')

# single_member coding
rep['single_member'] = np.where(rep['prop'] == 0, 1, np.nan)
rep.loc[rep['prop'].notna(), 'single_member'] = np.where(rep.loc[rep['prop'].notna(), 'prop'] == 0, 1, 0)

# Numeric id for country
rep['id_country'] = rep['country'].astype('category').cat.codes + 1

# Sort and generate lagged log dependent variable (by id_country: gen lagged_dispersion = L.dispersion)
rep = rep.sort_values(['id_country','election']).reset_index(drop=True)
rep['lagged_dispersion'] = rep.groupby('id_country')['dispersion_log'].shift(1)

# Prepare final dataset for regression: keep dispersion_log (log of dispersion), count_parties_log, single_member, lagged_dispersion, id_country
rep['count_parties_log'] = np.log(rep['count_parties'])
final = rep[['country','election','dispersion_log','count_parties_log','single_member','lagged_dispersion','id_country']].copy()
final = final.dropna(subset=['dispersion_log','count_parties_log'])

# Save final data for inspection
final.to_csv(output_data_csv, index=False)

# Regression: reg dispersion count_parties single_member lagged_dispersion, cluster(id_country)
# Stata reg uses OLS; dispersion variable already logged
# Drop rows with missing predictors (single_member or lagged_dispersion may be missing)
reg_df = final.dropna(subset=['count_parties_log'])
# Build model matrix
y = reg_df['dispersion_log']
X = reg_df[['count_parties_log','single_member','lagged_dispersion']]
X = sm.add_constant(X)

# Fit OLS with clustered SE by id_country
model = sm.OLS(y, X, missing='drop')
# Fit and request cluster robust cov
try:
    res = model.fit(cov_type='cluster', cov_kwds={'groups': reg_df['id_country']})
except Exception:
    # fallback to default if clustering fails
    res = model.fit()

# Save results summary
with open(output_results, 'w') as f:
    f.write(res.summary().as_text())

print('Replication script finished. Outputs:')
print(' -', output_data_csv)
print(' -', output_results)
