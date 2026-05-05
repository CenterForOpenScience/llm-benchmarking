# Python translation of the R script data_analysis_code.R

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pyreadr

# Load the dataset
result = pyreadr.read_r('/workspace/replication_data/data_clean.rds')
# Extract the pandas dataframe
data_clean = result[None]
# Reset index to ensure alignment
data_clean.reset_index(drop=True, inplace=True)
data_clean['cntry'] = data_clean['cntry'].astype('category')
data_clean.sort_values(by='cntry', inplace=True)
data_clean.dropna(subset=['trstprl_rev', 'imm_concern', 'happy_rev', 'stflife_rev', 'sclmeet_rev', 'distrust_soc', 'stfeco_rev', 'hincfel', 'stfhlth_rev', 'stfedu_rev', 'vote_gov', 'vote_frparty', 'lrscale', 'hhinc_std', 'agea', 'educ', 'female', 'vote_share_fr', 'socexp', 'lt_imm_cntry', 'wgi', 'gdppc', 'unemp'], inplace=True)

# Load the dataset
# Assuming the dataset is already loaded in a DataFrame named 'data_clean'

# Main analysis (complete cases, weights)
model = smf.mixedlm("trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc",
                    data=data_clean, groups=data_clean['cntry'])
result = model.fit()
print(result.summary())

# Auxiliary analysis 1 (complete cases, no weights)
model_no_weights = smf.mixedlm("trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc +\n                                stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +\n                                vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ + female +\n                                vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp",
                                data=data_clean, groups=data_clean['cntry'])
result_no_weights = model_no_weights.fit()
print(result_no_weights.summary())

# Note: Imputed data analysis will require additional handling for imputation in Python.
