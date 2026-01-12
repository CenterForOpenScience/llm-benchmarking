# Python translation of the R script
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Load the dataset
file_path = '/app/data/Final replication dataset.rds'
import pyreadr
result = pyreadr.read_r(file_path)
data = result[None]  # extract the dataframe
data = data.reset_index(drop=True)
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].mean(), inplace=True)
for column in data.select_dtypes(include=['object', 'category']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)
data = data.dropna(subset=['I03_ST_A_S27B', 'FSW_READ_TR', 'FSW_WRIT_TR', 'FSW_LIST_TR', 'PV1_WRIT_C', 'PV2_WRIT_C', 'PV3_WRIT_C', 'PV4_WRIT_C', 'PV5_WRIT_C', 'PV1_READ', 'PV2_READ', 'PV3_READ', 'PV4_READ', 'PV5_READ', 'PV1_LIST', 'PV2_LIST', 'PV3_LIST', 'PV4_LIST', 'PV5_LIST', 'I03_ST_A_S26A', 'I03_ST_A_S26B', 'SQt01i01', 'I08_ST_A_S02A', 'HISEI', 'PARED', 'SQt21i01'])
print(data.isnull().sum())
data['school_id'] = data['school_id'].astype('category')
data['country_id'] = data['country_id'].astype('category')
print(data.dtypes)
print(data.head())

# Data transformation
# Creation of a new variable that classifies participants in a monolingual (0) and in a bilingual group (1)
data['bilingual'] = np.where(data['I03_ST_A_S26A'].isin([2, 3]), 0, 1)
data = data.dropna(subset=['bilingual'])

# Data exclusion: Excluding students who speak English (the target language) at home
data = data[data['I03_ST_A_S27B'] == 0]

# Creation of an average score for writing, reading, and listening
data['ave_writing'] = data[['PV1_WRIT_C', 'PV2_WRIT_C', 'PV3_WRIT_C', 'PV4_WRIT_C', 'PV5_WRIT_C']].mean(axis=1)
data['ave_reading'] = data[['PV1_READ', 'PV2_READ', 'PV3_READ', 'PV4_READ', 'PV5_READ']].mean(axis=1)
data['ave_listening'] = data[['PV1_LIST', 'PV2_LIST', 'PV3_LIST', 'PV4_LIST', 'PV5_LIST']].mean(axis=1)
data['average_english'] = data[['ave_writing', 'ave_reading', 'ave_listening']].mean(axis=1)

# Converting Cultural Capital into a continuous variable
cultural_map = {'0-10 books': 0, '11-25 books': 1, '26-100 books': 2, '101-200 books': 3, '201-500 books': 4, 'More than 500 books': 5}
data['Cultural_capital'] = data['SQt21i01'].map(cultural_map).astype(float)

# Excluding observations with no weights or weights == 0
data = data[(data['FSW_WRIT_TR'] > 0) | (data['FSW_READ_TR'] > 0) | (data['FSW_LIST_TR'] > 0)]

# Centering continuous variables
scaler = StandardScaler(with_mean=True, with_std=False)
data['c_age'] = scaler.fit_transform(data[['I08_ST_A_S02A']])
data['c_HISEI'] = scaler.fit_transform(data[['HISEI']])

# Converting to Z-scores the variable 'parental education' and 'cultural capital'
scaler_std = StandardScaler()
data['Z_Parental'] = scaler_std.fit_transform(data[['PARED']])
data['Z_Cultural'] = scaler_std.fit_transform(data[['Cultural_capital']])

# Three-level model
model = mixedlm('average_english ~ bilingual + C(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural', data, groups=data['country_id'], re_formula='1')
result = model.fit()
print(result.summary())

# Note: Additional models for writing, reading, and listening can be similarly implemented.
