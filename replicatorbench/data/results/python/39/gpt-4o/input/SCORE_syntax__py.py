import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
file_path = '/app/data/SCORE_all data.sav'
data = pd.read_spss(file_path)

# Recode variables
cog_vars = ['Cog5', 'Cog6', 'Cog8', 'Cog9', 'Cog10', 'Cog11', 'Cog13', 'Cog14', 'Cog15', 'Cog16', 'Cog17', 'Cog18', 'Cog20', 'Cog21', 'Cog22', 'Cog24', 'Cog25', 'Cog26', 'Cog31', 'Cog32', 'Cog33']
for var in cog_vars:
    data[var + '_rec'] = data[var].apply(lambda x: -x if x != 0 else x)

# Compute mean for NFCog
nfcog_vars = ['Cog1', 'Cog2', 'Cog3', 'Cog4', 'Cog7', 'Cog12', 'Cog19', 'Cog23', 'Cog27', 'Cog28', 'Cog29', 'Cog30', 'Cog34'] + [var + '_rec' for var in cog_vars]
data['NFCog'] = data[nfcog_vars].mean(axis=1)

# Compute mean for IRI_ALL
iri_vars = ['IRI1', 'IRI2', 'IRI5', 'IRI6', 'IRI8', 'IRI9', 'IRI10', 'IRI11', 'IRI16', 'IRI17', 'IRI20', 'IRI21', 'IRI22', 'IRI23', 'IRI24', 'IRI25', 'IRI26', 'IRI27', 'IRI28'] + ['IRI3_rec', 'IRI4_rec', 'IRI7_rec', 'IRI12_rec', 'IRI13_rec', 'IRI14_rec', 'IRI15_rec', 'IRI18_rec', 'IRI19_rec']
data['IRI_ALL'] = data[iri_vars].mean(axis=1)

# Compute mean for Fear_ALL
fear_vars = ['Fear1', 'Fear4', 'Fear5', 'Fear7', 'Fear8', 'Fear9', 'Fear11', 'Fear12', 'Fear13', 'Fear14'] + ['Fear2_rec', 'Fear3_rec', 'Fear6_rec', 'Fear10_rec', 'Fear15_rec', 'Fear17_rec', 'Fear23_rec']
data['Fear_ALL'] = data[fear_vars].mean(axis=1)

# Compute APP_incong and APP_cong
app_vars_incong = ['APP1', 'APP2', 'APP3', 'APP4', 'APP5', 'APP6', 'APP7', 'APP8', 'APP9', 'APP10']
app_vars_cong = ['APP11', 'APP12', 'APP13', 'APP14', 'APP15', 'APP16', 'APP17', 'APP18', 'APP19', 'APP20']
data['APP_incong'] = data[app_vars_incong].apply(pd.to_numeric, errors='coerce').sum(axis=1)
data['APP_cong'] = data[app_vars_cong].apply(pd.to_numeric, errors='coerce').sum(axis=1)

# Compute RH_incong and RH_cong
for var in ['incong', 'cong']:
    data[f'RH_{var}'] = data[f'APP_{var}'] / 10

# Compute Utilitarian and Deontological factors
u = data['RH_cong'] - data['RH_incong']
data['U'] = u
data['D'] = data['RH_incong'] / (1 - u)

# Perform 2x3 mixed ANOVA (condition x dilemma type)
# This is a placeholder for the actual ANOVA implementation
# You would typically use a library like statsmodels or pingouin for this
anova_results = stats.f_oneway(data['RH_PMD'], data['RH_NMD'], data['RH_CMD'])
print('ANOVA results:', anova_results)

# Save the processed data
output_path = '/app/data/processed_data.csv'
data.to_csv(output_path, index=False)
