# Python translation of KMYR.do
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLS

# Load the dataset
file_path = '/app/data/finaldata_noNA.csv'
data = pd.read_csv(file_path)

# Create a non-string value for countries
country_codes = {country: idx for idx, country in enumerate(data['country'].unique())}
data['countrynum'] = data['country'].map(country_codes)

# Define National Affluence
data['NAff'] = data['gdp'] / data['pop']

# Define Imports from South
data['IMS'] = data['totalimport'] / (data['gdp'] * 10000)

# Define Exports to South
data['EXS'] = data['totalexport'] / (data['gdp'] * 10000)

# Detect outliers (placeholder for Hadi outlier detection)
data['bad'] = 0  # Placeholder, implement Hadi outlier detection if needed
data = data[data['bad'] == 0]

# Drop unnecessary columns
data.drop(columns=['country', 'countryyear', 'gdp', 'pop', 'totalimport', 'totalexport', 'bad'], inplace=True)

# Generate 5-year time dummies
years = [(1970, 1974), (1975, 1979), (1980, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2018)]
for start, end in years:
    col_name = f'DUM{start%100}to{end%100}'
    data[col_name] = ((data['year'] >= start) & (data['year'] <= end)).astype(int)

# Sort data
data.sort_values(by=['countrynum', 'year'], inplace=True)

# Fit GLS model
X = data[['IMS', 'EXS', 'unemp'] + [f'DUM{start%100}to{end%100}' for start, end in years]]
X = sm.add_constant(X)  # Add constant term
y = data['NAff']
model = GLS(y, X).fit()

# Save the processed data and model summary
processed_file_path = '/app/data/processed_data.csv'
data.to_csv(processed_file_path, index=False)

model_summary_path = '/app/data/model_summary.txt'
with open(model_summary_path, 'w') as f:
    f.write(model.summary().as_text())
