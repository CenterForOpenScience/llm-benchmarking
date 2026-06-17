import pandas as pd
import statsmodels.api as sm
from statsmodels.survey.survey_model import SurveyDesign, SurveyModel

# Load the dataset
file_path = '/app/data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta'
data = pd.read_stata(file_path)

# Transformations
# Dependent variable: Took ACT
data['took_ACT'] = data['drugs_taken_AL']

# Focal independent variable: any ACT voucher subsidy
data['act_subsidy'] = (data['maltest_chw_voucher_given'] == 1).astype(float)
data.loc[data['maltest_chw_voucher_given'] == 98, 'act_subsidy'] = pd.NA

# Other controls
# Household ID
data['hh_id'] = data.index + 1

# Strata
data['strata'] = data['cu_code']

# Declare survey design
design = SurveyDesign(strata=data['strata'], weights=data['weight'], cluster=data['hh_id'])

# Control variables
covariates = ['C(refrigerator)', 'C(mobile)', 'C(vip_toilet)', 'C(composting_toilet)', 'C(other_toilet)', 'C(stone_wall)', 'C(cement_wall)', 'num_sheep']

# Regression model for the impact on ACT access
model = SurveyModel.from_formula('took_ACT ~ act_subsidy + ' + ' + '.join(covariates), design)
result = model.fit()

# Print results
print(result.summary())
