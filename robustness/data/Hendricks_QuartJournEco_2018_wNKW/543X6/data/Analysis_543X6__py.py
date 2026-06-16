import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Paths
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
INPUT_FILE = os.path.join(DATA_DIR, "wage_gain_table.xlsx")
OUTPUT_FILE = os.path.join(DATA_DIR, "Analysis_543X6__py_results.csv")

# Load data
mData = pd.read_excel(INPUT_FILE, engine="openpyxl")

# Construct schooling dummies
vSchool = mData["edyrs"].astype(float)
vNS = (vSchool < 9).astype(int)
vSHS = vSchool.isin([9,10,11]).astype(int)
vHSD = (vSchool == 12).astype(int)
vSC = vSchool.isin([13,14,15]).astype(int)
# vGRAD defined in R but not included in regression

# Country dummies
country = mData["country"].astype(str)
vCOL = (country == "COL").astype(int)
vDOM = (country == "DOM").astype(int)
vECU = (country == "ECU").astype(int)
vGTM = (country == "GTM").astype(int)
vHTI = (country == "HTI").astype(int)
vMEX = (country == "MEX").astype(int)
vNIC = (country == "NIC").astype(int)
vPER = (country == "PER").astype(int)
vSLV = (country == "SLV").astype(int)

# Outcome: absolute wage difference
vWage = (mData["lastUsWageAdjusted"] - mData["lastHomeWageAdjusted"]).abs()

# Assemble DataFrame for modeling (no intercept)
X = pd.DataFrame({
    "vNS": vNS,
    "vSHS": vSHS,
    "vHSD": vHSD,
    "vSC": vSC,
    # Country FE excluding Peru and Ecuador (as per original R formula)
    "vCOL": vCOL,
    "vDOM": vDOM,
    "vGTM": vGTM,
    "vHTI": vHTI,
    "vMEX": vMEX,
    "vNIC": vNIC,
    "vSLV": vSLV,
})

df = pd.concat([vWage.rename("vWage"), X], axis=1)
# Drop rows with any missing used columns
before = len(df)
df = df.dropna()
after = len(df)

# Fit OLS without intercept
model = sm.OLS(df["vWage"], df.drop(columns=["vWage"]))
res = model.fit()

# Prepare output table
summary_df = pd.DataFrame({
    "coef": res.params,
    "std_err": res.bse,
    "t_value": res.tvalues,
    "p_value": res.pvalues,
})
summary_df.insert(0, "variable", summary_df.index)
summary_df.insert(0, "n_obs", res.nobs)

# Save results
summary_df.to_csv(OUTPUT_FILE, index=False)

print(f"Wrote Task1 results to {OUTPUT_FILE}; N used: {int(res.nobs)}; dropped: {before - after}")
