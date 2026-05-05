import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset
file_path = '/app/data/sandra_replicate.csv'
data = pd.read_csv(file_path)

# Filter data for accuracy
filtered_data = data[data['accuracy'] == 1]
filtered_data = filtered_data.reset_index(drop=True)
filtered_data.loc[:, 'SubjectID'] = filtered_data['SubjectID'].astype('category')filtered_data.loc[:, 'SubjectID'] = filtered_data['SubjectID'].astype('category')
formula = 'logRT ~ NFC * trial * rewardlevel + blocknumber'

# Fit the mixed-effects model
model = smf.mixedlm(formula, filtered_data, groups=filtered_data['SubjectID'])
result = model.fit()

# Print the summary of the model
print(result.summary())

# Additional analyses can be added similarly by defining new formulas and fitting models
# Example for accuracy model
# formula_accuracy = 'accuracy ~ NFC * trial * rewardlevel + blocknumber + (1|SubjectID)'
# model_accuracy = smf.mixedlm(formula_accuracy, filtered_data, groups=filtered_data['SubjectID'], family=sm.families.Binomial())
# result_accuracy = model_accuracy.fit()
# print(result_accuracy.summary())