import pandas as pd
import numpy as np
from scipy import stats

# Read data# Read data
data = pd.read_csv('/app/data/Data_Cleaned_22102020.csv', encoding='latin1')

# Clean numeric WTP columns: remove $ and commas and coerce
for col in ['Lot_WTPc', 'Gift_WTPc']:
    if col in data.columns:
        data[col] = data[col].astype(str).str.replace('[\$,]', '', regex=True)
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Create Cond: 0=lottery if Lot_check>0, 1=gift if Gift_check>0
data['Cond'] = np.nan
if 'Lot_check' in data.columns:
    data.loc[data['Lot_check'].notna() & (data['Lot_check']!='NA') & (data['Lot_check']!=''), 'Cond'] = 0
if 'Gift_check' in data.columns:
    data.loc[data['Gift_check'].notna() & (data['Gift_check']!='NA') & (data['Gift_check']!=''), 'Cond'] = 1

# Map Cond to labels
data['Cond_label'] = data['Cond'].map({0: 'lottery', 1: 'gift'})

# Create WTP
data['WTP'] = np.where(data['Cond']==0, data['Lot_WTPc'], np.where(data['Cond']==1, data['Gift_WTPc'], np.nan))

# Drop missing WTP
data = data[~data['WTP'].isna()].copy()

# Create check variable
data['check'] = np.where(data['Cond']==0, data.get('Lot_check', np.nan), np.where(data['Cond']==1, data.get('Gift_check', np.nan), np.nan))

# Filter comprehension == 3
dataS = data[data['check']==3].copy()

dataS.to_csv('/app/data/dataS.csv', index=False)

# Function to compute Cohen's d (two groups, pooled SD)
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / dof)
    return (x.mean() - y.mean()) / pooled_sd

# Group stats and Welch t-test for dataS
groups = dataS.groupby('Cond_label')['WTP']

results = []
for name, grp in groups:
    results.append((name, len(grp), grp.mean(), grp.std(ddof=1)))

# Prepare and run Welch t-test
lot = dataS[dataS['Cond_label']=='lottery']['WTP']
gift = dataS[dataS['Cond_label']=='gift']['WTP']

# handle cases with insufficient data
if len(lot)>0 and len(gift)>0:
    tstat, pval = stats.ttest_ind(lot, gift, equal_var=False, nan_policy='omit')
    d = cohens_d(lot, gift)
else:
    tstat, pval, d = [np.nan, np.nan, np.nan]

# Save results text
with open('/app/data/replication_results.txt', 'w') as f:
    f.write('Comprehension-filtered sample (check==3)\n')
    for name, n, mean, sd in results:
        f.write(f'{name}: N={n}, mean={mean}, sd={sd}\n')
    f.write(f'Welch t-test: t={tstat}, p={pval}\n')
    f.write(f"Cohen's d: {d}\n")

# WTP<=20 subset
dataSC = dataS[dataS['WTP']<=20].copy()
dataSC.to_csv('/app/data/dataSC.csv', index=False)

lotc = dataSC[dataSC['Cond_label']=='lottery']['WTP']
giftc = dataSC[dataSC['Cond_label']=='gift']['WTP']
if len(lotc)>0 and len(giftc)>0:
    tstatc, pvalc = stats.ttest_ind(lotc, giftc, equal_var=False, nan_policy='omit')
    dc = cohens_d(lotc, giftc)
else:
    tstatc, pvalc, dc = [np.nan, np.nan, np.nan]

with open('/app/data/replication_results.txt', 'a') as f:
    f.write('\nSubset WTP<=20\n')
    for name, n, mean, sd in dataSC.groupby('Cond_label')['WTP'].agg(['count','mean','std']).itertuples():
        pass
    f.write(f'Subset Welch t-test: t={tstatc}, p={pvalc}\n')
    f.write(f"Subset Cohen's d: {dc}\n")

print('Analysis complete. Results written to /app/data/replication_results.txt')