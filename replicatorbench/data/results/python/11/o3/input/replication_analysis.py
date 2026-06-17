import pyreadr, pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, os, sys

# Set data path
DATA_PATH = os.path.join('/app/data', 'Final replication dataset.rds')

# Load RDS using pyreadr
result = pyreadr.read_r(DATA_PATH)
# RDS may contain multiple objects; assume first
key = list(result.keys())[0]
df = result[key]

# Reproduce data transformations analogous to original R script

df = df.copy()

# Create bilingual indicator: if I03_ST_A_S26A equals 2 or 3 then 0 else 1 (reverse of R? follow R code logic)
# R code: ifelse(dat$I03_ST_A_S26A == (2 | 3), 0, 1)
# In R, (2 | 3) evaluates to  TRUE (logical) equal 1, so the condition is I03 == TRUE which is always FALSE, not correct.
# The intent was likely: value 2 or 3 indicates monolingual; else bilingual. We'll replicate logically: monolingual= values 2 or 3, bilingual=others (1)

df['bilingual'] = np.where(df['I03_ST_A_S26A'].isin([2,3]), 0, 1)

# Exclude rows with missing bilingual
df = df[~df['bilingual'].isna()]

# Exclude students who speak English at home (I03_ST_A_S27B == 0 keep?) R code keeps rows where I03_ST_A_S27B == 0
if 'I03_ST_A_S27B' in df.columns:
    df = df[df['I03_ST_A_S27B'] == 0]

# Average English scores
ENG_WRITE = ['PV1_WRIT_C','PV2_WRIT_C','PV3_WRIT_C','PV4_WRIT_C','PV5_WRIT_C']
ENG_READ = ['PV1_READ','PV2_READ','PV3_READ','PV4_READ','PV5_READ']
ENG_LIST = ['PV1_LIST','PV2_LIST','PV3_LIST','PV4_LIST','PV5_LIST']

df['ave_writing'] = df[ENG_WRITE].mean(axis=1, skipna=True)
df['ave_reading'] = df[ENG_READ].mean(axis=1, skipna=True)
df['ave_listening'] = df[ENG_LIST].mean(axis=1, skipna=True)

df['average_english'] = df[['ave_writing','ave_reading','ave_listening']].mean(axis=1, skipna=True)

# Cultural capital numeric mapping
cap_map = {
    '0-10 books':0,
    '11-25 books':1,
    '26-100 books':2,
    '101-200 books':3,
    '201-500 books':4,
    'More than 500 books':5
}
df['Cultural_capital'] = df['SQt21i01'].map(cap_map)
# Ensure numeric dtype for Cultural_capital for arithmetic operations
if pd.api.types.is_categorical_dtype(df['Cultural_capital']):
    df['Cultural_capital'] = df['Cultural_capital'].astype(float)


# Keep rows with positive any of FSW_WRIT_TR/READ/LIST weight
df = df[(df['FSW_WRIT_TR']>0) | (df['FSW_READ_TR']>0) | (df['FSW_LIST_TR']>0)]

# Center or standardize variables

df['c_age'] = df['I08_ST_A_S02A'] - df['I08_ST_A_S02A'].mean()

df['c_HISEI'] = df['HISEI'] - df['HISEI'].mean()

df['Z_Parental'] = (df['PARED'] - df['PARED'].mean()) / df['PARED'].std()

df['Z_Cultural'] = (df['Cultural_capital'] - df['Cultural_capital'].mean()) / df['Cultural_capital'].std()

# Remove rows with missing dependent or key independent variables
model_vars = ['average_english','bilingual','SQt01i01','c_age','c_HISEI','Z_Parental','Z_Cultural','country_id','school_id']
df_model = df[model_vars].dropna()

# Mixed effects model: random intercepts for school nested in country
# statsmodels MixedLM doesn't support nested random terms directly with string formula; create group variable combined

df_model['country_school'] = df_model['country_id'].astype(str) + ':' + df_model['school_id'].astype(str)

# Fit mixed model
md = smf.mixedlm('average_english ~ bilingual + C(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural',
                 df_model, groups=df_model['country_school'], re_formula='1')

mdf = md.fit(method='lbfgs')
print(mdf.summary())

# Save summary to text file
with open('/app/data/model_summary.txt','w') as f:
    f.write(mdf.summary().as_text())
