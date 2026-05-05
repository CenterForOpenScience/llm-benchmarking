# Python translation of the Stata do-file

import pyreadstat
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_ind

# Load the dataset using pyreadstat
data, meta = pyreadstat.read_dta('/app/data/GSSreplication.dta')

data.columns = data.columns.str.lower()

# Recode binary variables
binary_vars = ['hotcore', 'radioact', 'boyorgrl', 'lasers', 'electron', 'viruses', 'earthsun', 'condrift', 'bigbang', 'evolved', 'expdesgn', 'odds1', 'odds2']
for var in binary_vars:
    data[var + '_clean'] = data[var].replace({1: np.nan, 5: np.nan, 8: 0, 9: np.nan})

# Recode true/false variables
true_vars = ['hotcore', 'boyorgrl', 'electron', 'earthsun', 'condrift', 'bigbang', 'evolved', 'odds2']
false_vars = ['radioact', 'lasers', 'viruses', 'expdesgn', 'odds1']
for var in true_vars:
    data[var + '_clean'] = data[var + '_clean'].replace({2: 1, 3: 0, 4: 0})
for var in false_vars:
    data[var + '_clean'] = data[var + '_clean'].replace({3: 1, 2: 0, 4: 0})

# Recode scale variables
scale_vars = ['nextgen', 'toofast', 'advfront', 'scibnfts']
for var in scale_vars:
    data[var + '_clean'] = data[var].replace({1: np.nan, 2: 1, 3: 2, 4: 3, 5: 4, 6: np.nan, 7: np.nan})

# Reverse code 'toofast'
data['toofast_clean'] = data['toofast_clean'].replace({4: 1, 3: 2, 2: 3, 1: 4})

# Recode 'bible'
data['bible_clean'] = data['bible'].replace({1: np.nan, 2: 1, 3: 2, 4: 3, 5: np.nan, 6: np.nan, 7: np.nan})

# Recode 'reliten'
data['reliten_clean'] = data['reliten'].replace({1: np.nan, 2: 4, 3: 3, 4: 2, 5: 1, 6: np.nan, 7: np.nan})

# Latent Class Analysis
# Note: Python's sklearn.mixture.GaussianMixture is used as an approximation for latent class analysis
lca_vars = [var + '_clean' for var in binary_vars + scale_vars + ['bible', 'reliten']]
X = data[lca_vars]
X_non_na = X.dropna()
gmm = GaussianMixture(n_components=3, random_state=12345)
gmm.fit(X_non_na)
predictions = gmm.predict(X_non_na) + 1

# Create a new column with NaNs and fill in predictions
predclass = pd.Series(np.nan, index=data.index)
predclass[X_non_na.index] = predictions
data['predclass'] = predclass

# T-test for 'evolved_clean' by 'PostsecVsTrad'
data['PostsecVsTrad'] = np.where(data['predclass'] == 3, 1, np.where(data['predclass'] == 1, 0, np.nan))
postsec = data[data['PostsecVsTrad'] == 1]['evolved_clean'].dropna()
trad = data[data['PostsecVsTrad'] == 0]['evolved_clean'].dropna()
t_stat, p_value = ttest_ind(postsec, trad)

# Save results
with open('/app/data/OBrienReplication_Axxe_20201012_results.txt', 'w') as f:
    f.write(f'T-test statistic: {t_stat}, p-value: {p_value}')
