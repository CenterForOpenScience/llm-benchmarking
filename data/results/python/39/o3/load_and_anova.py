"""Replicate 2 (Condition) x 3 (Dilemma-Type) mixed ANOVA on proportion rejecting harm (RH variables).

Steps
-----
1. Load SPSS .sav dataset using pyreadstat.
2. Keep variables: participant identifier (row number or explicit id), Condition, RH_PMD, RH_NMD, RH_CMD.
3. Reshape to long format (each row = participant x dilemma_type)
4. Run mixed ANOVA with Condition (between) and dilemma_type (within).
5. Save table of ANOVA results and print focal interaction.

All input/output paths are inside the mounted /app/data directory as required by the orchestration
infrastructure.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

try:
    import pyreadstat
except ImportError:
    raise SystemExit("pyreadstat must be installed inside the environment.")

try:
    import pingouin as pg
except ImportError:
    raise SystemExit("pingouin must be installed inside the environment.")

# -----------------------------------------------------------------------------
# Constants & Paths# -----------------------------------------------------------------------------
# Constants & Paths
# -----------------------------------------------------------------------------
from glob import glob

# Attempt to resolve dataset path robustly regardless of mount point quirks
POSSIBLE_PATHS = [
    "/app/data/original/39/input/replication_data/SCORE_all data.sav",
    "/app/data/data/original/39/input/replication_data/SCORE_all data.sav",
    os.path.join(os.path.dirname(__file__), "SCORE_all data.sav"),
]

DATA_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    print("ERROR: Could not locate SCORE_all data.sav in expected paths. Checked:")
    for p in POSSIBLE_PATHS:
        print(" -", p)
    sys.exit(1)

print("Using dataset path:", DATA_PATH)

RESULTS_DIR = "/app/data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load dataset# -----------------------------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------------------------
print("Loading dataset from:", DATA_PATH)
try:
    df, meta = pyreadstat.read_sav(DATA_PATH, apply_value_formats=True)
except Exception as e:
    print("Failed to load .sav file:", e)
    sys.exit(1)

print("Data loaded. Shape:", df.shape)

# -----------------------------------------------------------------------------
# 2. Select relevant variables
# -----------------------------------------------------------------------------
required_vars = ["Condition", "RH_PMD", "RH_NMD", "RH_CMD"]
missing = [v for v in required_vars if v not in df.columns]
if missing:
    print("ERROR: Missing variables from dataset:", missing)
    sys.exit(1)

sub = df[required_vars].copy()

# Drop rows with any missing RH values or missing condition
sub_clean = sub.dropna()
print(f"After dropping missing, N = {len(sub_clean)} (dropped {len(sub) - len(sub_clean)})")

# Ensure Condition is a simple string label for downstream analysis
sub_clean["Condition"] = sub_clean["Condition"].astype(str)

# -----------------------------------------------------------------------------
# 3. Reshape to long format for pingouin
# -----------------------------------------------------------------------------
long_df = pd.melt(
    sub_clean.reset_index().rename(columns={"index": "participant"}),
    id_vars=["participant", "Condition"],
    value_vars=["RH_PMD", "RH_NMD", "RH_CMD"],
    var_name="dilemma_type",
    value_name="RH"
)

# Map nicer labels for dilemma type
long_df["dilemma_type"] = long_df["dilemma_type"].map({
    "RH_PMD": "Personal",
    "RH_NMD": "NonPersonal",
    "RH_CMD": "Control"
})

# -----------------------------------------------------------------------------
# 4. Run mixed-design ANOVA using pingouin
# -----------------------------------------------------------------------------
print("Running mixed ANOVA (Condition between, dilemma_type within)...")
anova = pg.mixed_anova(
    dv="RH",
    within="dilemma_type",
    between="Condition",
    subject="participant",
    data=long_df
)
print("ANOVA table:\n", anova)

# -----------------------------------------------------------------------------
# 5. Save results
# -----------------------------------------------------------------------------
results_file = os.path.join(RESULTS_DIR, "mixed_anova_results.csv")
anova.to_csv(results_file, index=False)
print("Saved ANOVA results to", results_file)

# Save a JSON summary focusing on the interaction
interaction_row = anova[(anova["Source"] == "Condition * dilemma_type")]
summary = {
    "F": interaction_row["F"].values[0] if not interaction_row.empty else None,
    "p": interaction_row["p-unc"].values[0] if not interaction_row.empty else None,
    "np2": interaction_row["np2"].values[0] if not interaction_row.empty else None,
    "N": len(sub_clean)
}
with open(os.path.join(RESULTS_DIR, "interaction_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Interaction summary:", summary)
print("Done.")
