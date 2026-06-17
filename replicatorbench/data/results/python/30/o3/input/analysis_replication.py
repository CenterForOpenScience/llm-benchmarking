"""
Replication analysis script translating the original Stata .do file `Analysis.do` into Python.

The script reproduces the key interaction test between position (pstn) and the lagged share of
downloaded files (shr) on individual investment in the shared infrastructure (nvst) using a
mixed-effects linear model with a random intercept for experimental group (grp).

The focal coefficient is that for the interaction term `pstn:shr` (pstnshr).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path("/app/data/Full_long.dta")

# -----------------------------
# 1. Load data# -----------------------------
# 1. Load data
# -----------------------------
print(f"Loading dataset from: {DATA_PATH}")

# Use pandas.read_stata with minimal arguments to ensure compatibility across versions
# read_stata returns a DataFrame directly (metadata attribute available if needed)

df = pd.read_stata(DATA_PATH, convert_dates=True, convert_categoricals=False, preserve_dtypes=False)

# -----------------------------
# 2. Variable construction# -----------------------------
# 2. Variable construction (mimicking Stata do-file)
# -----------------------------
# Only keep observations from round > 2, analogous to Stata dropping the first two rounds
mask_round_gt2 = df["round"] > 2

# Helper to set values conditionally, else np.nan to replicate Stata's missing (.)

df.loc[mask_round_gt2, "nvst"] = df.loc[mask_round_gt2, "playerinvestment"]

df.loc[mask_round_gt2, "dwnld"] = df.loc[mask_round_gt2, "playerdownloaded_files"]

df.loc[mask_round_gt2, "pyff"] = (
    df.loc[mask_round_gt2, "playercollected_tokens"]
    + (10 - df.loc[mask_round_gt2, "playerinvestment"])
)

df.loc[mask_round_gt2, "rnd"] = df.loc[mask_round_gt2, "round"] - 2

# Position mapping (A-E -> 1-5)
position_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

df["pstn"] = df["playerrole"].map(position_map)

# Group and (group, round) identifiers
# Replicates: grp = Session*100 + groupid_in_subsession (Stata)

df["grp"] = df["Session"] * 100 + df["groupid_in_subsession"]

df["grprnd"] = df["grp"] * 100 + df["rnd"]

# dwnldpyff & bndwdth

df.loc[mask_round_gt2, "dwnldpyff"] = df.loc[mask_round_gt2, "playercollected_tokens"]

df.loc[mask_round_gt2, "bndwdth"] = df.loc[mask_round_gt2, "groupbandwidth"]

# -----------------------------
# 3. Group-level sums for each group-round
# -----------------------------
# Compute sums of nvst and dwnldpyff within each (grp, rnd)

grp_sums = (
    df[mask_round_gt2]
    .groupby(["grprnd"])
    .agg(grpnvst=("nvst", "sum"), grpdwnldpyff=("dwnldpyff", "sum"))
    .reset_index()
)

# Merge sums back to main df

df = df.merge(grp_sums, on="grprnd", how="left")

# Share variables

df["shrnvst"] = np.where(
    df["grpnvst"] > 0,
    df["nvst"] / df["grpnvst"],
    0.2,
)

df["shrdwnld"] = np.where(
    df["grpdwnldpyff"] > 0,
    df["dwnldpyff"] / df["grpdwnldpyff"],
    0.2,
)

# -----------------------------
# 4. Lag operator for share of downloads (shr) – lagged by one round within id
# -----------------------------
# Sort by player id and round, then shift

df.sort_values(["id", "rnd"], inplace=True)

df["shr"] = df.groupby("id")["shrdwnld"].shift(1)

# Interaction terms

df["pstnshr"] = df["pstn"] * df["shr"]

df["pstnshrnvst"] = df["pstn"] * df["shrnvst"]

# -----------------------------
# 5. Prepare analysis dataset – drop rows with missing relevant variables
# -----------------------------
analysis_cols = [
    "nvst",
    "pstn",
    "shr",
    "pstnshr",
    "rnd",
    "grp",
]

analysis_df = df.loc[mask_round_gt2, analysis_cols].dropna()

print("Analysis dataset dimensions:", analysis_df.shape)

# -----------------------------
# 6. Mixed-effects model (random intercept for grp)
# -----------------------------
model_formula = "nvst ~ pstn + shr + pstnshr + rnd"
print("Fitting mixed-effects model:", model_formula)

md = smf.mixedlm(model_formula, data=analysis_df, groups=analysis_df["grp"])
res = md.fit(reml=True)
print(res.summary())

# Extract focal coefficient
focal_coef = res.params.get("pstnshr", np.nan)
focal_se = res.bse.get("pstnshr", np.nan)
focal_p = res.pvalues.get("pstnshr", np.nan)

print("\nFocal interaction effect (pstn * lagged share of downloads):")
print(f"  Coefficient: {focal_coef:.4f}")
print(f"  Std. Error: {focal_se:.4f}")
print(f"  p-value   : {focal_p:.4g}")

# Optional: save results to a CSV for later comparison
results_path = Path("/app/data/replication_results.csv")
# Save coefficient table to CSV
coef_table = res.summary().tables[1]  # This is a pandas DataFrame-like object
# Convert to DataFrame if it is a SimpleTable
try:
    coef_df = coef_table.as_html()
    import pandas as pd
    coef_df = pd.read_html(coef_df, header=0, index_col=0)[0]
except Exception:
    # Fallback: create DataFrame manually from params, bse, pvalues
    coef_df = pd.DataFrame({
        'coef': res.params,
        'std_err': res.bse,
        'p_value': res.pvalues
    })

coef_df.to_csv(results_path)
print(f"Detailed coefficients saved to: {results_path}")
print(f"Detailed coefficients saved to: {results_path}")
