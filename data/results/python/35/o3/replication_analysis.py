"""
Replication analysis for Tremoliere mortality‐salience moral utilitarian judgment claim.
This script reproduces the key test using the replication dataset.
All input/output is restricted to /app/data as required.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

# Paths
DATA_PATH = Path('/app/data/Tremoliere_generalizability_score.csv')

# Load dataset

df = pd.read_csv(DATA_PATH)

# Select relevant columns and reshape to long format
cols_id = [
    'ResponseId', 'death_salience', 'pain_salience', 'satisfactory_manipulation_response1',
    'age', 'gender', 'gender_3_TEXT', 'politic_1', 'politic_2', 'politic_3', 'race',
    'race_7_TEXT', 'race_8_TEXT', 'income', 'education', 'open', 'cond', 'salience',
    'participant_uid'
]
cols_value = ['moral_accept', 'moral_accept1']

long_df = (
    df[cols_id + cols_value]
    .melt(id_vars=cols_id, value_vars=cols_value, var_name='variable', value_name='moral_acceptability')
)

# Keep rows that passed manipulation check (1 indicates correct)
long_df = long_df[long_df['satisfactory_manipulation_response1'] == 1].copy()

# Recoding outcome to binary 0 = not moral, 1 = moral (match R code logic)
long_df['moral_acceptability_01'] = long_df['moral_acceptability'].map({1: 0, 2: 1})

# Scenario label
scenario_map = {
    'moral_accept': 'impartial_beneficience',
    'moral_accept1': 'partial_beneficience'
}
long_df['moral_scenario'] = long_df['variable'].map(scenario_map)

# Categorical variables
long_df['salience_fact'] = long_df['salience'].astype('category')
long_df['moral_scenario_fact'] = long_df['moral_scenario'].astype('category')

# Check that categories are as expected
print('Salience levels:', long_df['salience_fact'].cat.categories)
print('Scenario levels:', long_df['moral_scenario_fact'].cat.categories)

# Logistic regression with cluster‐robust SE at participant level
model = smf.glm(
    formula='moral_acceptability_01 ~ salience_fact * moral_scenario_fact',
    data=long_df,
    family=sm.families.Binomial()
)
res = model.fit(cov_type='cluster', cov_kwds={'groups': long_df['participant_uid']})
print(res.summary())

# Save coefficients table to csv inside /app/data
coefs = res.summary2().tables[1]
coefs.to_csv('/app/data/replication_coefficients.csv')

print('\nSaved coefficient table to /app/data/replication_coefficients.csv')
