"""Run replication analysis for Molden (2018) direct replication dataset.
This script fits the preregistered mixed-effects model and writes a summary
of the results to /app/data/replication_results.txt inside the container.
"""

import os
import pandas as pd
import statsmodels.formula.api as smf

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "sandra_replicate.csv")
OUTPUT_PATH = "/app/data/replication_results.txt"

print(f"Reading dataset from {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(DATA_PATH)

# -----------------------------------------------------------------------------
# Load and preprocess
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Total rows in raw data: {len(df):,}")

# Keep correct trials only
accurate = df[df["accuracy"] == 1].copy().reset_index(drop=True)
print(f"Rows after accuracy filter: {len(accurate):,}")

# Z-standardise logRT and NFC
accurate["logRT_z"] = (accurate["logRT"] - accurate["logRT"].mean()) / accurate["logRT"].std(ddof=0)
accurate["NFC_z"]   = (accurate["NFC"]   - accurate["NFC"].mean())   / accurate["NFC"].std(ddof=0)

# Drop rows with any NA in variables used in the model
model_vars = ["logRT_z", "NFC_z", "trial", "rewardlevel", "blocknumber", "SubjectID"]
pre_drop = len(accurate)
accurate = accurate.dropna(subset=model_vars).reset_index(drop=True)
print(f"Dropped {pre_drop - len(accurate)} rows due to missing values.")
print(f"Final N for modelling: {len(accurate):,}")

# -----------------------------------------------------------------------------
# Fit mixed-effects model
# -----------------------------------------------------------------------------
formula = "logRT_z ~ NFC_z * trial * rewardlevel + blocknumber"
print("Fitting model:", formula)

# Use integer group codes to avoid index mismatch bug in statsmodels
group_codes, _ = pd.factorize(accurate["SubjectID"], sort=True)

model = smf.mixedlm(formula, accurate, groups=group_codes, re_formula="1")
result = model.fit(method="lbfgs")

print(result.summary())

# -----------------------------------------------------------------------------
# Extract focal interaction
# -----------------------------------------------------------------------------
term = "NFC_z:trial:rewardlevel"
if term in result.params.index:
    coef = float(result.params[term])
    pval = float(result.pvalues[term])
    focal_line = f"Focal 3-way interaction coefficient: {coef:.4f}, p-value: {pval:.4g}"
else:
    focal_line = "Focal interaction term not found."
print(focal_line)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
with open(OUTPUT_PATH, "w") as fh:
    fh.write(result.summary().as_text())
    fh.write("\n" + focal_line + "\n")

print(f"Results written to {OUTPUT_PATH}")
