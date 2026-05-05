import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Load the dataset
file_path = '/app/data/Tremoliere_generalizability_score.csv'
df = pd.read_csv(file_path)

# Filter the data
filtered_df = df[df['satisfactory_manipulation_response1'] == 1].copy()

# Create moral acceptability measure
filtered_df['moral_acceptability_01'] = filtered_df['moral_accept'].apply(lambda x: 0 if x == 1 else 1)

# Create a variable for the different moral dilemma outcomes
filtered_df['moral_scenario'] = filtered_df['moral_accept'].apply(lambda x: 'impartial_beneficience' if x == 'moral_accept' else 'partial_beneficience')

# Set variables to their type
filtered_df['salience_fact'] = filtered_df['salience'].astype('category')
filtered_df['moral_scenario_fact'] = filtered_df['moral_scenario'].astype('category')

# Descriptive Visualization
sns.catplot(x='moral_acceptability_01', hue='salience', data=filtered_df, kind='count', height=5, aspect=2)
plt.title('Moral acceptability by salience condition and outcome')
plt.show()

# Encode categorical variables
filtered_df['salience_fact'] = filtered_df['salience_fact'].cat.codes
filtered_df['moral_scenario_fact'] = filtered_df['moral_scenario_fact'].cat.codes

# Check for multicollinearity
print(filtered_df[['salience_fact', 'moral_scenario_fact']].corr())

# Logistic Regression Model
logit_model = sm.Logit(filtered_df['moral_acceptability_01'], filtered_df[['salience_fact']])
result = logit_model.fit()
print(result.summary())

# Simplified ANOVA
anova_model = ols('moral_acceptability_01 ~ C(salience_fact)', data=filtered_df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)