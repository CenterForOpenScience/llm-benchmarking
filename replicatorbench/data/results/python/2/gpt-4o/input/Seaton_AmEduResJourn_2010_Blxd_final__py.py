import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyreadr

# Load the dataset
result = pyreadr.read_r('/app/data/PISA2012.replication.RDS')
df = result[None]  # extract the pandas dataframe

# Create unique IDs for schools and students
df['school_id'] = df['SCHOOLID'].astype('category').cat.codes
df['student_id'] = df['STIDSTD'].astype('category').cat.codes

# Filter and clean the data
df = df.dropna(subset=['MEMOR', 'SCMAT', 'PV1MATH'])

# Standardize variables
df['MEMOR_z'] = (df['MEMOR'] - df['MEMOR'].mean()) / df['MEMOR'].std()
df['SCMAT_z'] = (df['SCMAT'] - df['SCMAT'].mean()) / df['SCMAT'].std()
df['school_PV1MATH_z'] = (df['PV1MATH'] - df['PV1MATH'].mean()) / df['PV1MATH'].std()

# Estimate school average ability
df['school_avg_ability'] = df.groupby('school_id')['school_PV1MATH_z'].transform('mean')

# Create cross products
df['CROSS1'] = df['MEMOR_z'] * df['school_avg_ability']

# Fit multilevel models
model = sm.MixedLM.from_formula('SCMAT_z ~ MEMOR_z * school_avg_ability', groups='school_id', data=df)
result = model.fit()

# Print the results
print(result.summary())
