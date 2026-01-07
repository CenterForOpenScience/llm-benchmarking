# Python translation of the Stata do-file for replicating the research claim from Andrews & Money (2009)

import pyreadstat
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
import pandas as pd

# Load datasets
cmp_df, _ = pyreadstat.read_dta('/app/data/CMP_final.dta')
cpds_df, _ = pyreadstat.read_dta('/app/data/CPDS_final.dta')

print('CMP_final.dta columns:', cmp_df.columns.tolist())
print('CPDS_final.dta columns:', cpds_df.columns.tolist())
cmp_df.rename(columns={'countryname': 'country'}, inplace=True)
# Drop duplicates
cpds_df.drop_duplicates(inplace=True)

cmp_df['edate'] = pd.to_datetime(cmp_df['edate'], errors='coerce')
cmp_df['year'] = cmp_df['edate'].dt.year.astype('int32')
print('Unique values in cpds_df year column:', cpds_df['year'].unique())
cpds_df = cpds_df[pd.to_numeric(cpds_df['year'], errors='coerce').notnull()]
cpds_df['year'] = cpds_df['year'].astype('int32')
# Merge datasets
cmp_df['edate'] = pd.to_datetime(cmp_df['edate'], errors='coerce')
cmp_df['year'] = cmp_df['edate'].dt.year
merged_df = pd.merge(cmp_df, cpds_df, on=['country', 'year'], how='inner')

# Generate an id for each country-election pair
merged_df['election'] = merged_df.groupby(['country', 'edate']).ngroup()

# Focal Independent Variable: Count of parties in system
merged_df['relative_seat'] = merged_df['absseat'] / merged_df['totseats']
merged_df['relative_seat_t_1'] = merged_df.groupby('party')['relative_seat'].shift(-1)
merged_df['consecutive'] = (merged_df['relative_seat'] > 0.01) & (merged_df['relative_seat_t_1'] > 0.01)
merged_df = merged_df[merged_df.groupby('party')['consecutive'].transform('max') == 1]

# Number of parties in the system
merged_df['count_parties'] = merged_df.groupby(['country', 'election'])['party'].transform('size')
merged_df['count_parties'] = np.log(merged_df['count_parties'])

merged_df.dropna(subset=['per303', 'per401', 'per402', 'per403', 'per404', 'per407', 'per412', 'per413', 'per414', 'per504', 'per505', 'per701'], inplace=True)
# Dependent Variable: Distance between Extreme Parties (Dispersion)
pca = PCA(n_components=1)
merged_df['economic_policy'] = pca.fit_transform(merged_df[['per303', 'per401', 'per402', 'per403', 'per404', 'per407', 'per412', 'per413', 'per414', 'per504', 'per505', 'per701']])
merged_df['min'] = merged_df.groupby(['country', 'election'])['economic_policy'].transform('min')
merged_df['max'] = merged_df.groupby(['country', 'election'])['economic_policy'].transform('max')
merged_df['dispersion'] = merged_df['max'] - merged_df['min']
merged_df['dispersion'] = np.log(merged_df['dispersion'])

# Other controls
merged_df['single_member'] = (merged_df['prop'] == 0).astype(float)

# Keep unique observation for country-election
unique_df = merged_df.drop_duplicates(subset=['country', 'election'])
unique_df['id_country'] = unique_df['country'].astype('category').cat.codes

# Declare panel
unique_df = unique_df.set_index(['id_country', 'election'])

# Generate lagged log dependent variable
unique_df['lagged_dispersion'] = unique_df.groupby(level=0)['dispersion'].shift(1)

unique_df.dropna(subset=['count_parties', 'single_member', 'lagged_dispersion', 'dispersion'], inplace=True)
# Focal replication analysis
X = unique_df[['count_parties', 'single_member', 'lagged_dispersion']]
X = sm.add_constant(X)
y = unique_df['dispersion']
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': unique_df.index.get_level_values('id_country')})

# Output results
print(model.summary())
