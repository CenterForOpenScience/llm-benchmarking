import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM

# Load the dataset
file_path = '/app/data/finaldata_noNA.csv'
data = pd.read_csv(file_path)

# Create a non-string value for countries
country_codes = {country: idx for idx, country in enumerate(data['country'].unique())}
data['countrynum'] = data['country'].map(country_codes)

# Define National Affluence as in the paper
data['NAff'] = data['gdp'] / data['pop']

# Define Imports from South as in the paper
data['IMS'] = data['totalimport'] / (data['gdp'] * 10000)

# Define Exports to South as in the paper
data['EXS'] = data['totalexport'] / (data['gdp'] * 10000)

# Detect outliers using Hadi outlier detection (placeholder for actual implementation)
data['bad'] = 0  # Placeholder: Implement Hadi outlier detection if available

# Drop observations tagged as outliers
data = data[data['bad'] != 1]

# Retain only the columns necessary for estimation
data = data.drop(columns=['country', 'countryyear', 'gdp', 'pop', 'totalimport', 'totalexport', 'bad'])

# Generate 5-year time dummies
years = [(1970, 1974), (1975, 1979), (1980, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2018)]
for start, end in years:
    col_name = f'DUM{start % 100}to{end % 100}'
    data[col_name] = ((data['year'] >= start) & (data['year'] <= end)).astype(int)

# Sort data by countrynum and year
data = data.sort_values(by=['countrynum', 'year'])

# Fit the model (placeholder for actual model fitting)
# model = MixedLM.from_formula('NAff ~ IMS + EXS + unemp + C(countrynum) + DUM70to74 + DUM75to79 + DUM80to84 + DUM85to89 + DUM90to94 + DUM95to99 + DUM00to04 + DUM05to09 + DUM10to14 + DUM15to18', data)
# result = model.fit()
# print(result.summary())

# Note: Implement the actual model fitting and diagnostics as needed.
