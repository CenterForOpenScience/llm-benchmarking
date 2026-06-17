import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.anova import AnovaRM

# Load datasets
items = pd.read_csv('/app/data/ItemsList_Final.csv')
data = pd.read_csv('/app/data/LeBeouf_replication_data.csv')

# Data cleaning and preparation
# Exclude unnecessary columns and filter data
columns_to_exclude = ['StartDate', 'Q1']
data = data.drop(columns=columns_to_exclude)
data = data[data['Status'] == 0]  # Exclude 'Preview' sessions

data = data[(data['Attention1'] == 7) & (data['Attention2'] == 1) & (data['screenQ'] == 2)]

# Create separate data frames for the three conditions
control_data = data[data['Cond'].isin([1, 2])]
brandA_data = data[data['Cond'] == 3]
brandB_data = data[data['Cond'] == 4]

# Reshape data from wide to long format
control_long = pd.wide_to_long(control_data, stubnames=['Util', 'Symbol'],
                               i='ID', j='ItemCatID', sep='_', suffix='\d+')
brandA_long = pd.wide_to_long(brandA_data, stubnames=['Util', 'Symbol'],
                              i='ID', j='ItemCatID', sep='_', suffix='\d+')
brandB_long = pd.wide_to_long(brandB_data, stubnames=['Util', 'Symbol'],
                              i='ID', j='ItemCatID', sep='_', suffix='\d+')

# Combine data
combined = pd.concat([control_long, brandA_long, brandB_long])
combined = combined.merge(items, on=['ItemCatID', 'Cond'])
combined['ScoreDiff'] = combined['Util'] - combined['Symbol']

# Collapse data by participant, condition, and product type
collapsed_data = combined.groupby(['ID', 'ProductType', 'Condition']).agg({'ScoreDiff': 'mean'}).reset_index()

# Perform ANOVA
anova = AnovaRM(collapsed_data, 'ScoreDiff', 'ID', within=['ProductType', 'Condition']).fit()
print(anova.summary())

# Save results
collapsed_data.to_csv('/app/data/collapsed_data_results.csv', index=False)
