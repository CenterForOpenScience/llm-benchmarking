import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = '/app/data/SCORE_ALL_DATA.csv'
data = pd.read_csv(file_path)

# Prepare the data for regression
data['S1_price'] = pd.to_numeric(data['S1_price'].replace('[\$,]', '', regex=True), errors='coerce')
data = data.dropna(subset=['S1_quality', 'S1_price', 'S1_purchase', 'S1_likely'])
X = data[['S1_quality', 'S1_price', 'S1_purchase']]
y = data['S1_likely']
y = data['S1_likely']

# Initialize and fit the regression model
model = LinearRegression()
model.fit(X, y)

# Output the coefficients
coefficients = model.coef_
intercept = model.intercept_

print('Coefficients:', coefficients)
print('Intercept:', intercept)

# Save the results
results_path = '/app/data/regression_results.txt'
with open(results_path, 'w') as f:
    f.write(f'Coefficients: {coefficients}\n')
    f.write(f'Intercept: {intercept}\n')
