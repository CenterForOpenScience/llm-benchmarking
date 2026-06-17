import pandas as pd
from scipy.stats import ttest_ind

# Load the dataset
data = pd.read_csv('/app/data/ohtsubo data.csv')

# Recode a4
recode_map = {7: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6, 1: 7}
data['a4r'] = data['a4'].map(recode_map)

# Compute atot
data['atot'] = data['a1'] + data['a2'] + data['a3'] + data['a4r'] + data['a5'] + data['a6']

# Perform t-test
group1 = data[data['condition'] == 1]['atot']
group2 = data[data['condition'] == 0]['atot']
t_stat, p_value = ttest_ind(group1, group2)

# Output results
print(f'T-statistic: {t_stat}, P-value: {p_value}')
