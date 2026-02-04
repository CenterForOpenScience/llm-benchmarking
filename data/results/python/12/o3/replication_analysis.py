"""Replication analysis script.
Performs OLS regressions with village-clustered SEs testing the effect of locaste_land_v
on various income outcomes.

This script is intended to be run both on the host (during development) and inside the
container runtime orchestrated by the evaluation platform. Therefore all file paths are
resolved dynamically relative to this script as a fallback if the /app/data mount is not
present.
"""

import os
import pandas as pd
import statsmodels.formula.api as smf

###############################################################################
# Helper: resolve dataset path
###############################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_PATH = os.path.join(SCRIPT_DIR, "analysis_data.dta")
CONTAINER_DATA_PATH = os.path.join("/app/data", os.path.relpath(LOCAL_DATA_PATH, start=os.getcwd()))

DATA_PATH = CONTAINER_DATA_PATH if os.path.isfile(CONTAINER_DATA_PATH) else LOCAL_DATA_PATH

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Could not locate analysis_data.dta. Tried {CONTAINER_DATA_PATH} and {LOCAL_DATA_PATH}.")

###############################################################################
# Load data
###############################################################################

df = pd.read_stata(DATA_PATH)

###############################################################################
# Regression helper# Regression helper# Regression helper
###############################################################################

def run_ols_cluster(data, outcome, cluster_var="vill_id"):
    """Run OLS with village-clustered SEs and state×caste fixed effects.
    Performs listwise deletion to align clusters with estimation sample.
    """
    required_cols = [outcome, "literate_hh", "land_owned", "locaste_land_v", "stcode", "caste", cluster_var]
    dsub = data[required_cols].dropna()

    formula = (
        f"{outcome} ~ literate_hh + land_owned + locaste_land_v + "
        "C(stcode)*C(caste)"
    )

    model = smf.ols(formula=formula, data=dsub).fit(
        cov_type="cluster", cov_kwds={"groups": dsub[cluster_var]}
    )
    return model

###############################################################################
# Estimation###############################################################################
# Estimation# Estimation
###############################################################################

results = {}
results["raw_full"] = run_ols_cluster(df, "raw_inc_per_acre")
results["net_full"] = run_ols_cluster(df, "net_inc_per_acre")
subset = df[df["stcode"].isin([2, 15])]
results["raw_subset_upb"] = run_ols_cluster(subset, "raw_inc_per_acre")

###############################################################################
# Save / print concise output
###############################################################################

out_lines = []
for name, mod in results.items():
    coef = mod.params.get("locaste_land_v", float("nan"))
    se = mod.bse.get("locaste_land_v", float("nan"))
    pval = mod.pvalues.get("locaste_land_v", float("nan"))
    out_lines.append(
        f"Model: {name}\n"
        f"  Coef locaste_land_v: {coef:.4f}\n"
        f"  SE: {se:.4f}\n"
        f"  p-value: {pval:.4g}\n"
        f"  N: {int(mod.nobs)}\n"
        f"  R-squared: {mod.rsquared:.4f}\n\n"
    )

RESULT_PATH = "/app/data/replication_results.txt" if os.path.isdir("/app/data") else os.path.join(SCRIPT_DIR, "replication_results.txt")
with open(RESULT_PATH, "w") as f:
    f.writelines(out_lines)

print("".join(out_lines))
