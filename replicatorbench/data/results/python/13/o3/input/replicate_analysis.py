"""
Replication analysis for McLaren (2012) focal claim
Data source: ESS Round 5 Western Europe clean dataset prepared by Kolczynska (2020)
Performs mixed-effects regression predicting distrust of parliament (trstprl_rev) from immigration concern (imm_concern) and controls, with random intercepts by country.
Outputs summary table to stdout and saves csv of fixed effects.
"""

import os
import sys
import pandas as pd
import numpy as np
import pyreadr
import statsmodels.formula.api as smf

# Allow passing an optional path
# Default data directory: same directory where this script resides (mounted by orchestrator)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", SCRIPT_DIR)
FILE = os.environ.get("CLEAN_FILE", "data_clean_5pct.rds")  # default to smaller 5% sample to reduce run time

dataset_path = os.path.join(DATA_DIR, FILE)
print(f"Reading dataset from {dataset_path}")

# Load RDS
result = pyreadr.read_r(dataset_path)
# RDS returns dict {None: df}

df = result[None]
print(f"Loaded dataframe with shape {df.shape}")

# Drop rows with missing outcome or predictor
required_cols = [
    "trstprl_rev", "imm_concern", "happy_rev", "stflife_rev", "sclmeet_rev", "distrust_soc",
    "stfeco_rev", "hincfel", "stfhlth_rev", "stfedu_rev", "vote_gov", "vote_frparty", "lrscale",
    "hhinc_std", "agea", "educ", "female", "vote_share_fr", "socexp", "lt_imm_cntry", "wgi", "gdppc", "unemp", "cntry"
]

df_model = df[required_cols].dropna()
print(f"Data after dropping NAs: {df_model.shape}")

# Recode categorical predictors as category dtype
for col in ["vote_gov", "vote_frparty", "cntry"]:
    df_model[col] = df_model[col].astype("category")

# Build formula (fixed effects + random intercept for cntry)
formula = "trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc + " \
          "stfeco_rev + hincfel + stfhlth_rev + stfedu_rev + " \
          "vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ + female + " \
          "vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp"

print("Fitting mixed-effects model (random intercept for country)...")

# Use statsmodels MixedLM via lme4-like method; groups=cntry
import statsmodels.api as sm
md = sm.MixedLM.from_formula(formula, groups="cntry", data=df_model, re_formula="1")

# fit with REML by default
model = md.fit(reml=True)
print(model.summary())

# Save fixed effects to CSV
output_path = os.path.join(SCRIPT_DIR, "replication_fixed_effects.csv")
model.fe_params.to_csv(output_path, header=["estimate"])
print(f"Fixed effects saved to {output_path}")
