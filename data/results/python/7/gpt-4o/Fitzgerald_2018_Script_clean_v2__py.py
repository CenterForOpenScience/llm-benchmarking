# Python script for data preprocessing and model fitting
import pyreadstat
import pandas as pd
import numpy as np
from linearmodels import PanelOLS

# Load datasets
compiled, _ = pyreadstat.read_dta('/app/data/compiled.dta')
epa, _ = pyreadstat.read_dta('/app/data/epa.dta')
hhsize, _ = pyreadstat.read_dta('/app/data/hhsize.dta')

# Merge datasets
merged_data = compiled.merge(epa, on='State').merge(hhsize, on='State')

# Calculate additional variables
# Assuming necessary calculations are done here

# Log-transform variables
for col in ['epa', 'wrkhrs', 'emppop', 'laborprod', 'pop', 'manuf', 'energy', 'hhsize07', ]:
    merged_data['log_' + col] = np.log(merged_data[col])
merged_data.rename(columns={'log_epa': 'log_carbon_emissions'}, inplace=True)
merged_data = merged_data.set_index(['State', 'year_x'])
print('Columns in merged_data:', merged_data.columns)
print(merged_data.head())

# Fit panel regression models
model = PanelOLS.from_formula('log_carbon_emissions ~ log_wrkhrs + log_emppop + log_laborprod + log_pop + log_manuf + log_energy + log_hhsize07', data=merged_data)
results = model.fit()

# Output results
print(results.summary)
