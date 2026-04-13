import pyreadstat
import pandas as pd
import statsmodels.api as sm

# Load the dataset
# Assuming the dataset is in the /app/data directory
file_path = '/app/data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta'
data, meta = pyreadstat.read_dta(file_path)

# Prepare variables
independent_vars = ['cu_code', 'ses_no_sheep', 'ses_toilet_type', 'ses_wall_material']
X = data[independent_vars]
X = sm.add_constant(X)  # Adds a constant term to the predictor

y = data['take_action']

# Convert data to numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing values
X = X.dropna()
y = y[X.index]

# Ensure X and y are aligned and convert to numpy arrays
X, y = X.align(y, join='inner', axis=0)
# Conduct regression analysis using OLS
model = sm.OLS(y, X).fit()

# Convert to numpy arrays after fitting the model
X = X.to_numpy()
y = y.to_numpy()

# Print the summary of the regression
print(model.summary())

# Save the results
with open('/app/data/regression_results.txt', 'w') as f:
    f.write(model.summary().as_text())

# Convert data to numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing values
X = X.dropna()
y = y[X.index]

# Ensure X and y are aligned and convert to numpy arrays
X, y = X.align(y, join='inner')
X = X.to_numpy()
y = y.to_numpy()
