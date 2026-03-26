import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the data
fulldata = pd.read_csv('/app/data/AlTammemi_Survey_deidentify.csv')

# Prepare the data for logistic regression
# Assuming 'online_learning_1' is the target variable and 'Kessler_1' to 'Kessler_10' are features
X = fulldata[['Kessler_1', 'Kessler_2', 'Kessler_3', 'Kessler_4', 'Kessler_5',
              'Kessler_6', 'Kessler_7', 'Kessler_8', 'Kessler_9', 'Kessler_10']]
y = fulldata['online_learning_1']

# Convert target variable to binary if necessary
# Here we assume 'online_learning_1' needs to be binary for logistic regression
# This step may need adjustment based on actual data values
# y = np.where(y == 'NoMotivation', 0, 1)

# Handle missing values by imputing with the mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Handle missing values in y by imputing with the most frequent value
imputer_y = SimpleImputer(strategy='most_frequent')
y = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Output the model summary
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Predict and calculate accuracy or other metrics if needed
# predictions = model.predict(X)
# accuracy = np.mean(predictions == y)
# print('Accuracy:', accuracy)