import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load datasets
full_long = pd.read_stata('/app/data/Full_long.dta')

# Generate new variables
full_long['nvst'] = full_long.apply(lambda row: row['playerinvestment'] if row['round'] > 2 else None, axis=1)
full_long['pstn'] = full_long['playerrole'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5})
full_long['rnd'] = full_long.apply(lambda row: row['round'] - 2 if row['round'] > 2 else None, axis=1)
full_long['dwnld'] = full_long.apply(lambda row: row['playerdownloaded_files'] if row['round'] > 2 else None, axis=1)
full_long['pyff'] = full_long.apply(lambda row: row['playercollected_tokens'] + (10 - row['playerinvestment']) if row['round'] > 2 else None, axis=1)
full_long['grp'] = full_long['Session'] * 100 + full_long['groupid_in_subsession']
full_long['idrnd'] = full_long['id'] * 100 + full_long['rnd']
full_long['grprnd'] = full_long['grp'] * 100 + full_long['rnd']
full_long['dwnldpyff'] = full_long.apply(lambda row: row['playercollected_tokens'] if row['round'] > 2 else None, axis=1)
full_long['bndwdth'] = full_long.apply(lambda row: row['groupbandwidth'] if row['round'] > 2 else None, axis=1)

# Drop rounds < 3
full_long_v1 = full_long[full_long['round'] >= 3]

# Collapse data by group and round
sums = full_long_v1.groupby('grprnd').agg({'nvst': 'sum', 'dwnldpyff': 'sum'}).rename(columns={'nvst': 'grpnvst', 'dwnldpyff': 'grpdwnldpyff'}).reset_index()

# Merge datasets
full_long_v1 = full_long_v1.merge(sums, on='grprnd', how='left')

# Generate share variables
full_long_v1['shrnvst'] = full_long_v1.apply(lambda row: row['nvst'] / row['grpnvst'] if row['grpnvst'] > 0 else 0.2, axis=1)
full_long_v1['shrdwnld'] = full_long_v1.apply(lambda row: row['dwnldpyff'] / row['grpdwnldpyff'] if row['grpdwnldpyff'] > 0 else 0.2, axis=1)

# Lagged share variable
full_long_v1['shr'] = full_long_v1['shrdwnld'].shift(1)

# Interaction terms
full_long_v1['pstnshr'] = full_long_v1['pstn'] * full_long_v1['shr']
full_long_v1['pstnshrnvst'] = full_long_v1['pstn'] * full_long_v1['shrnvst']

# Reset index to ensure alignment
full_long_v1 = full_long_v1.reset_index(drop=True)

# Ensure grp is a categorical variable
full_long_v1['grp'] = full_long_v1['grp'].astype('category')

# Debugging: Check grp variable and data frame
print("Unique groups:", full_long_v1['grp'].nunique())
print("Data frame shape:", full_long_v1.shape)
print("Missing values in grp:", full_long_v1['grp'].isnull().sum())

# Mixed-effects models
try:
    model1 = smf.mixedlm("nvst ~ pstn + shr + pstnshr + rnd", full_long_v1, groups=full_long_v1["grp"]).fit()
    print(model1.summary())
except Exception as e:
    print("Error in model1:", e)

try:
    model2 = smf.mixedlm("dwnldpyff ~ bndwdth + pstn + shrnvst + pstnshrnvst + rnd", full_long_v1, groups=full_long_v1["grp"]).fit()
    print(model2.summary())
except Exception as e:
    print("Error in model2:", e)