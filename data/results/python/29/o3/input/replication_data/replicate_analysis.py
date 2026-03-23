"""
Replicate the focal claim: interaction between self-concept clarity (Clear vs Conflicted) and rating distribution (Bimodal vs Unimodal) on product desirability.

Dataset: SCORE_ALL_DATA.csv located at /app/data/original/29/input/replication_data/
This script:
1. Loads dataset
2. Filters rows passing exclusion == 0
3. Creates outcome variable "desirability" as average of S1_likely, S1_quality, S1_purchase (following data dictionary explanation of AVERAGE used in SPSS tab)
4. Creates dummy-coded variables:
   - experimental: 1 for Bimodal, -1 for Unimodal (based on TYPE)
   - clarity: 1 for Clear, -1 for Conflicted (based on TYPE)
   - interaction = experimental * clarity
5. Runs OLS regression: desirability ~ experimental + clarity + interaction
6. Prints coefficients, standard errors, p-values.
Outputs results table to /app/data/replication_results.csv for downstream comparison.
"""
import os
import pandas as pd
import statsmodels.api as sm

# Determine dataset path relative to this script location to avoid hard-coded absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "SCORE_ALL_DATA.csv")
OUTPUT_PATH = "/app/data/replication_results.csv"

# Load dataset# Load dataset
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"Initial shape: {df.shape}")

# Ensure Exclusion is numeric (values like '0', '1', '2.3', etc.)
df['Exclusion_numeric'] = pd.to_numeric(df['Exclusion'], errors='coerce')
# Clean: keep only rows with Exclusion == 0
clean_df = df[df['Exclusion_numeric'] == 0].copy()
print(f"After exclusion filter: {clean_df.shape}")

# Outcome variables (may have missing) - convert to numeric if not already
for col in ['S1_likely', 'S1_quality', 'S1_purchase']:
    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

clean_df['desirability'] = clean_df[['S1_likely', 'S1_quality', 'S1_purchase']].mean(axis=1)


# Drop rows with missing desirability
clean_df = clean_df.dropna(subset=['desirability'])
print(f"After dropping missing desirability: {clean_df.shape}")

# Coding experimental and clarity from TYPE
# TYPE values: ClB, ClU, CoB, CoU

def code_experimental(type_val):
    return 1 if type_val.endswith('B') else -1  # Bimodal vs Unimodal

def code_clarity(type_val):
    return 1 if type_val.startswith('Cl') else -1  # Clear vs Conflicted

clean_df['experimental'] = clean_df['TYPE'].apply(code_experimental)
clean_df['clarity'] = clean_df['TYPE'].apply(code_clarity)
clean_df['interaction'] = clean_df['experimental'] * clean_df['clarity']

# Regression
X = clean_df[['experimental', 'clarity', 'interaction']]
X = sm.add_constant(X)
y = clean_df['desirability']
model = sm.OLS(y, X, missing='drop').fit()
print(model.summary())

# Save results
results_df = pd.DataFrame({'coefficient': model.params, 'std_error': model.bse, 'p_value': model.pvalues})
results_df.to_csv(OUTPUT_PATH)
print(f"Results saved to {OUTPUT_PATH}")
