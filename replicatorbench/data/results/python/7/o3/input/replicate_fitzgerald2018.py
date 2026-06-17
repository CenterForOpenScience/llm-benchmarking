#!/usr/bin/env python3
"""
Clean replication script for Fitzgerald (2018) elasticity of CO2 emissions wrt working hours.
This Python translation loads three Stata datasets, merges and constructs variables,
then runs an OLS with State and Year fixed effects and clustered SEs (by State).

Compared with the preregistered plan that used `linearmodels.PanelOLS`, this implementation
uses statsmodels OLS with dummy variables to avoid `linearmodels` seg-fault issues in
QEMU/ARM emulation. Coefficient estimates are identical; only the estimation routine differs.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.simplefilter("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. Locate and load data
# ---------------------------------------------------------------------------
DATA_DIR = Path("/app/data/original/7/python/replication_data")# First try directory where script resides (works when code & data co-located)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
# If datasets not present, fallback to /app/data/replication_data or /app/data
if not (DATA_DIR / "compiled.dta").exists():
    if (Path("/app/data/replication_data") / "compiled.dta").exists():
        DATA_DIR = Path("/app/data/replication_data")
    elif (Path("/app/data") / "compiled.dta").exists():
        DATA_DIR = Path("/app/data")

print(f"Using data directory: {DATA_DIR}")

required_files = ["compiled.dta", "hhsize.dta", "epa.dta"]required_files = ["compiled.dta", "hhsize.dta", "epa.dta"]
for fname in required_files:
    if not (DATA_DIR / fname).exists():
        raise FileNotFoundError(f"Expected {(DATA_DIR / fname)} not found inside container.")

print(f"Using data directory: {DATA_DIR}")

df_main = pd.read_stata(DATA_DIR / "compiled.dta")
df_hh   = pd.read_stata(DATA_DIR / "hhsize.dta")
df_epa  = pd.read_stata(DATA_DIR / "epa.dta")

# ---------------------------------------------------------------------------
# 2. Reshape hhsize from wide to long: columns hhsize07, hhsize08, ...
# ---------------------------------------------------------------------------
value_cols = [c for c in df_hh.columns if c.startswith("hhsize")]
long_list = []
for col in value_cols:
    year_val = int(col.replace("hhsize", ""))  # "07" -> 7
    tmp = df_hh[["State", col]].copy()
    tmp["year"] = year_val
    tmp.rename(columns={col: "hhsize"}, inplace=True)
    long_list.append(tmp)

df_hh_long = pd.concat(long_list, ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Merge datasets on State & year
# ---------------------------------------------------------------------------
df = df_main.merge(df_hh_long, on=["State", "year"], how="left")
df = df.merge(df_epa, on=["State", "year"], how="left")

# ---------------------------------------------------------------------------
# 4. Derived variables (following original R logic)
# ---------------------------------------------------------------------------
# Employed population percentage
df["emppop_pct"] = df["emppop"] / (df["pop"] * 1000) * 100

# Manufacturing share of GDP
df["manu_gdp"] = df["manuf"] / df["gdp"] * 100

# Variables to take logs of
log_vars = [
    "epa", "wrkhrs", "emppop_pct", "laborprod", "pop",
    "manu_gdp", "energy", "hhsize", "workpop"
]

# Prevent log(<=0)
for v in log_vars:
    if (df[v] <= 0).any():
        raise ValueError(f"Variable {v} contains non-positive values; cannot take log.")
    df[v] = np.log(df[v])

# ---------------------------------------------------------------------------
# 5. Drop rows with missing data in these vars
# ---------------------------------------------------------------------------
pre_rows = len(df)
df_clean = df.dropna(subset=log_vars)
print(f"Dropped {pre_rows - len(df_clean)} rows due to NA; analysis sample: {len(df_clean)}.")

# ---------------------------------------------------------------------------
# 6. Fixed-effects (within) regression via OLS with dummies & clustered SEs
# ---------------------------------------------------------------------------
formula = (
    "epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + energy + hhsize + workpop + C(State) + C(year)"
)
model = smf.ols(formula=formula, data=df_clean)
res = model.fit(cov_type="cluster", cov_kwds={"groups": df_clean["State"]})

print(res.summary())

# ---------------------------------------------------------------------------
# 7. Extract focal coefficient
# ---------------------------------------------------------------------------
coef = res.params.get("wrkhrs")
se   = res.bse.get("wrkhrs")
pval = res.pvalues.get("wrkhrs")
print("\nFocal result: Elasticity of CO2 w.r.t working hours (wrkhrs coefficient)")
print(f"Coefficient: {coef:.3f}  SE: {se:.3f}  p-value: {pval:.3g}")
