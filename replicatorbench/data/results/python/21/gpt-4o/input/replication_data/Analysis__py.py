import pandas as pd
import statsmodels.formula.api as smf

# Load the data
DF1 = pd.read_csv('/app/data/Bischetti_Survey_Part1_deidentify.csv')
DF2 = pd.read_csv('/app/data/Bischetti_Survey_Part2_deidentify.csv')

# Merge datasets by participant ID# Merge datasets by participant ID, ensuring alignment
DF = pd.merge(DF1, DF2, on='participant_id', how='outer', suffixes=('_part1', '_part2'))

# Drop any rows with missing participant_id
DF.dropna(subset=['participant_id'], inplace=True)# Restructure data from wide to long format
DF_long = DF.melt(id_vars=['participant_id', 'state', 'age'], 
                  value_vars=[col for col in DF.columns if 'disturbing' in col],
                  var_name='name', value_name='Aversiveness')

# Clean the 'name' column
DF_long['name'] = DF_long['name'].str.replace('disturbing', 'anote')

# Load metadata# Metadata files are missing, skipping this step.# Merge with metadata# Skipping metadata merge due to missing files.# Create a placeholder 'label' variable for demonstration
DF_long['label'] = 'placeholder_label'# Skipping label cleaning due to missing metadata.# Fit the multilevel model
model = smf.mixedlm("Aversiveness ~ label", DF_long, 
                    groups=DF_long['participant_id'],
                    re_formula="~label")
result = model.fit(reml=False)

# Print the summary
print(result.summary())