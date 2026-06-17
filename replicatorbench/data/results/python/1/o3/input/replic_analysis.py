"""
Replication analysis script to test the association between county-level Trump 2016 vote share and changes in social distancing behaviour (proxied by the share of residents staying home) during 19 March–1 April 2020.

The script reproduces, in Python, the core logic of the original R code found in `kavanagh_analysis.R`, restricted to the focal hypothesis of interest.

All input and output paths use the /app/data directory as required by execution guidelines.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Attempt to locate data directory. Prefer /app/data mount but fall back to local path.
DEFAULT_BASE_PATH = "/app/data/original/1/python/replication_data"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(DEFAULT_BASE_PATH):
    BASE_PATH = DEFAULT_BASE_PATH
else:
    # Fallback: data shipped alongside the script inside the container/workspace
    BASE_PATH = SCRIPT_DIR
    print(f"WARNING: Using fallback BASE_PATH={BASE_PATH}. Ensure data files are present.")

COUNTY_FILE = os.path.join(BASE_PATH, "county_variables.csv")
TRANSPORT_FILE = os.path.join(BASE_PATH, "transportation.csv")

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
print("Loading data…")
county_df = pd.read_csv(COUNTY_FILE)
trans_df = pd.read_csv(TRANSPORT_FILE)

# Ensure date column is datetime
trans_df["date"] = pd.to_datetime(trans_df["date"])

# -----------------------------------------------------------------------------
# 2. Construct time-period indicator and proportion staying home
# -----------------------------------------------------------------------------
print("Constructing social-distancing metric…")

def label_time_period(d):
    """Return string label for rows that fall into predefined windows."""
    if pd.Timestamp("2020-02-16") <= d <= pd.Timestamp("2020-02-29"):
        return "reference"
    elif pd.Timestamp("2020-03-19") <= d <= pd.Timestamp("2020-04-01"):
        return "march"
    elif pd.Timestamp("2020-08-16") <= d <= pd.Timestamp("2020-08-29"):
        return "august"
    else:
        return np.nan

trans_df["time_period"] = trans_df["date"].apply(label_time_period)
trans_df = trans_df.dropna(subset=["time_period", "pop_home", "pop_not_home"])

# Proportion of devices at home per county-day
trans_df["prop_home"] = trans_df["pop_home"] / (trans_df["pop_home"] + trans_df["pop_not_home"])

# Average within county x period
period_avg = (
    trans_df.groupby(["fips", "state", "time_period"], as_index=False)["prop_home"]
    .mean()
)

# Pivot to wide format
wide = period_avg.pivot_table(
    index=["fips", "state"], columns="time_period", values="prop_home"
).reset_index()

# Require reference & march for analysis
wide = wide.dropna(subset=["reference", "march"])

# Compute percentage-point change (scaled to 100) relative to reference
wide["prop_home_change_march"] = 100 * (wide["march"] / wide["reference"] - 1)

# -----------------------------------------------------------------------------
# 3. Merge with county-level covariates
# -----------------------------------------------------------------------------
print("Merging county covariates…")

data = pd.merge(wide, county_df, on="fips", how="left", validate="many_to_one")

# -----------------------------------------------------------------------------
# 4. Variable preparation
# -----------------------------------------------------------------------------
print("Preparing variables…")

# Rename for consistency
rename_map = {
    "percent_male": "male_percent",
    "percent_rural": "prop_rural",
}

data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

# Scale percentages from proportion to 0–100 if necessary
pct_cols = [
    "trump_share",
    "male_percent",
    "percent_black",
    "percent_hispanic",
    "percent_college",
    "percent_retail",
    "percent_transportation",
    "percent_hes",
    "prop_rural",
]
for col in pct_cols:
    if col in data.columns:
        # Heuristic: assume <=1 indicates proportion, convert to %
        mask = data[col].between(0, 1, inclusive="both")
        data.loc[mask, col] = data.loc[mask, col] * 100

# Income: convert to thousands USD for scaling
if "income_per_capita" in data.columns:
    data["income_per_capita"] = data["income_per_capita"] / 1000.0

# Drop rows with missing outcome or trump_share
analysis_df = data.dropna(subset=["prop_home_change_march", "trump_share", "state"])
print(f"Analysis sample size: {len(analysis_df):,} counties")

# -----------------------------------------------------------------------------
# 5. Regression model (state fixed effects)
# -----------------------------------------------------------------------------
print("Estimating model…")

# List of covariates (keep only those present)
baseline_covars = [
    "income_per_capita",
    "trump_share",
    "male_percent",
    "percent_black",
    "percent_hispanic",
    "percent_college",
    "percent_retail",
    "percent_transportation",
    "percent_hes",
    "prop_rural",
]

covars_in_data = [c for c in baseline_covars if c in analysis_df.columns]

formula = "prop_home_change_march ~ " + " + ".join(covars_in_data) + " + C(state)"
model = smf.ols(formula, data=analysis_df).fit(cov_type="HC1")
print(model.summary())

# -----------------------------------------------------------------------------
# 6. Effect of an interquartile range (IQR) increase in Trump share
# -----------------------------------------------------------------------------
trump_q = analysis_df["trump_share"].quantile([0.25, 0.75]).values
trump_iqr = trump_q[1] - trump_q[0]
coef = model.params.get("trump_share")
se = model.bse.get("trump_share")

# Multiply coefficient by IQR for interpretation
iqr_effect = coef * trump_iqr
iqr_se = se * trump_iqr
from scipy import stats
z = iqr_effect / iqr_se
p_value = 2 * stats.norm.sf(abs(z))

print("\n----------------------------------------------")
print(f"Trump share IQR (25th–75th): {trump_iqr:.2f} percentage points")
print(f"Estimated effect on social-distancing (pp): {iqr_effect:.2f}")
print(f"p-value: {p_value:.4f}")
print("----------------------------------------------")

# Save main results as CSV for later comparison
out_path = "/app/data/replication_results.csv"
res_df = pd.DataFrame(
    {
        "coef": [coef],
        "se": [se],
        "iqr": [trump_iqr],
        "iqr_effect": [iqr_effect],
        "iqr_p_value": [p_value],
    }
)
res_df.to_csv(out_path, index=False)
print(f"Results saved to {out_path}")
