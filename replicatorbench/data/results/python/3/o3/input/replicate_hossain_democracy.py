"""
Replication analysis for Hossain (2020) democracy index claim
This script replicates the focal hypothesis that democracy index is positively associated with COVID-19 cases per million people across countries.
It follows the logic of the provided Stata do-file but is implemented in Python using pandas and statsmodels.
All file IO is restricted to the /app/data directory as required.
"""
import os
import sys
import pandas as pd
import statsmodels.api as sm

# ---------------- Configuration ---------------- #
# Determine the path to the RDS dataset relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "COVID replication.rds")

# Choose which timeframe to analyse. The original study used cases up to 03-Apr-2020,
# which corresponds to the column `COVID.12.31_04.03` in the dataset.
# Options mirror those in the original Stata script:
#   1 -> COVID.12.31_04.03  (original timeframe)
#   2 -> COVID.04.04_08.11  (post-original timeframe)
#   3 -> COVID.12.31_08.11  (whole timeframe)
DATASET_FLAG = int(os.getenv("DATASET_FLAG", 1))
assert DATASET_FLAG in {1, 2, 3}, "DATASET_FLAG must be 1, 2, or 3"

# ------------------------------------------------ #

# Helper to load RDS via pyreadr if available, else raise clear error.
try:
    import pyreadr
except ImportError as e:
    sys.stderr.write("pyreadr package is required. Install via pip install pyreadr\n")
    raise

result = pyreadr.read_r(DATA_PATH)
# RDS file contains a single data.frame
key = list(result.keys())[0]
df = result[key]

# Rename columns for convenience (replace spaces and punctuation)
rename_map = {
    "Democracy index (EIU)": "democracy_raw",
    "Annual_temp": "temperature",
    "trade.2016": "openness",
    "COVID.12.31_04.03": "COVID_12_31_04_03",
    "COVID.04.04_08.11": "COVID_04_04_08_11",
    "COVID.12.31_08.11": "COVID_12_31_08_11",
    "popData2019": "population"
}
df = df.rename(columns=rename_map)

# Choose total_cases according to DATASET_FLAG
if DATASET_FLAG == 1:
    df["total_cases"] = df["COVID_12_31_04_03"]
elif DATASET_FLAG == 2:
    df["total_cases"] = df["COVID_04_04_08_11"]
elif DATASET_FLAG == 3:
    df["total_cases"] = df["COVID_12_31_08_11"]

# Compute cases per million people
# Ensure population is numeric and non-zero
mask_valid_pop = df["population"] > 0
if not mask_valid_pop.all():
    df = df.loc[mask_valid_pop]

df["cases_per_million"] = df["total_cases"] / df["population"] * 1e6

# Democracy index: divide by 10 to match 0-10 scale from Economist Intelligence Unit
# Gapminder stores as 0-100 percentage

# Handle missing values if any
if df["democracy_raw"].isna().any():
    df = df.dropna(subset=["democracy_raw"])

df["democracy"] = df["democracy_raw"] / 10.0

# Openness and temperature already appropriately scaled

# Prepare regression variables
X = df[["democracy", "temperature", "openness"]].copy()
X = sm.add_constant(X)  # adds intercept
Y = df["cases_per_million"]

model = sm.OLS(Y, X).fit(cov_type='HC3')  # robust SE like Stata's default robust? (HC3)

print("Replication of Hossain (2020) democracy index claim")
print("---------------------------------------------------")
print(model.summary())

# Save coefficient of democracy to /app/data for downstream comparison
out_path = "/app/data/hossain_replication_results.csv"
coef = model.params["democracy"]
se = model.bse["democracy"]
pval = model.pvalues["democracy"]
summary_df = pd.DataFrame({
    "coefficient": [coef],
    "std_error": [se],
    "p_value": [pval],
    "n_obs": [int(model.nobs)],
    "dataset_flag": [DATASET_FLAG]
})
summary_df.to_csv(out_path, index=False)
print(f"\nSaved key results to {out_path}")
