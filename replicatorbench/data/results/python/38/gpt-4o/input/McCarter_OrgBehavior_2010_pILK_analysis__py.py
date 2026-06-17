import pandas as pd
import statsmodels.formula.api as smf

# Load the dataset
file_path = '/app/data/McCarter_OrgBehavior_2010_pILK data CLEAN.csv'
data = pd.read_csv(file_path).dropna(subset=['loss_aversion', 'fear'])

# Mixed model analysis
# Test of H* that contributions will be lower under losses
model_cont = smf.mixedlm("cont ~ losses", data, groups=data["sessioncode"], re_formula="1")
result_cont = model_cont.fit()
print(result_cont.summary())

# Exploratory analysis
# Test that loss aversion and fear of non-contributions are greater under losses
model_loss_aversion = smf.mixedlm("loss_aversion ~ losses", data, groups=data["sessioncode"], re_formula="~id")
result_loss_aversion = model_loss_aversion.fit()
print(result_loss_aversion.summary())

model_fear = smf.mixedlm("fear ~ losses", data, groups=data["sessioncode"], re_formula="~id")
result_fear = model_fear.fit()
print(result_fear.summary())