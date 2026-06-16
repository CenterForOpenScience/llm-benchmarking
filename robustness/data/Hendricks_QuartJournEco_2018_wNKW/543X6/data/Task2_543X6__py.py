import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.contrast import ContrastResults

# Paths
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
INPUT_FILE = os.path.join(DATA_DIR, "wage_gain_table.xlsx")
OUT_DIR = os.path.join(DATA_DIR, "Task2_543X6__py")
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
mData = pd.read_excel(INPUT_FILE, engine="openpyxl")

# Construct schooling dummies
vSchool = mData["edyrs"].astype(float)
vNS = (vSchool < 9).astype(int)
vSHS = vSchool.isin([9,10,11]).astype(int)
vHSD = (vSchool == 12).astype(int)
vSC = vSchool.isin([13,14,15]).astype(int)
# vGRAD defined in R but not included as main predictors in tests

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

# Full model with intercept
X_full = pd.DataFrame({
    "Intercept": 1.0,
    "vNS": vNS,
    "vSHS": vSHS,
    "vHSD": vHSD,
    "vSC": vSC,
    "vCOL": vCOL,
    "vDOM": vDOM,
    "vGTM": vGTM,
    "vHTI": vHTI,
    "vMEX": vMEX,
    "vNIC": vNIC,
    "vSLV": vSLV,
})

df_full = pd.concat([vWage.rename("vWage"), X_full], axis=1).dropna()
model_full = sm.OLS(df_full["vWage"], df_full.drop(columns=["vWage"]))
res_full = model_full.fit()

# Joint F-test: vSHS = vHSD = vSC = 0
R = np.zeros((3, len(res_full.params)))
param_names = res_full.params.index.tolist()
for i, name in enumerate(["vSHS", "vHSD", "vSC"]):
    row = np.zeros(len(param_names))
    if name in param_names:
        row[param_names.index(name)] = 1.0
    R[i, :] = row
r = np.zeros(3)
ftest = res_full.f_test((R, r))

# Save full model coefficients
coef_full = pd.DataFrame({
    "variable": res_full.params.index,
    "coef": res_full.params.values,
    "std_err": res_full.bse.values,
    "t_value": res_full.tvalues.values,
    "p_value": res_full.pvalues.values,
})
coef_full.insert(0, "n_obs", res_full.nobs)
coef_full.to_csv(os.path.join(OUT_DIR, "full_model_coef.csv"), index=False)

# Save joint test
with open(os.path.join(OUT_DIR, "joint_test.txt"), "w") as f:
    f.write(f"F-statistic: {float(ftest.fvalue)}\n")
    f.write(f"p-value: {float(ftest.pvalue)}\n")
    f.write(f"df_denom: {int(res_full.df_resid)}\n")
    f.write(f"df_num: 3\n")

# Reduced model with intercept: only vNS and country dummies
X_red = pd.DataFrame({
    "Intercept": 1.0,
    "vNS": vNS,
    "vCOL": vCOL,
    "vDOM": vDOM,
    "vGTM": vGTM,
    "vHTI": vHTI,
    "vMEX": vMEX,
    "vNIC": vNIC,
    "vSLV": vSLV,
})

df_red = pd.concat([vWage.rename("vWage"), X_red], axis=1).dropna()
model_red = sm.OLS(df_red["vWage"], df_red.drop(columns=["vWage"]))
res_red = model_red.fit()

coef_red = pd.DataFrame({
    "variable": res_red.params.index,
    "coef": res_red.params.values,
    "std_err": res_red.bse.values,
    "t_value": res_red.tvalues.values,
    "p_value": res_red.pvalues.values,
})
coef_red.insert(0, "n_obs", res_red.nobs)
coef_red.to_csv(os.path.join(OUT_DIR, "reduced_model_coef.csv"), index=False)

print(f"Task2 outputs written to {OUT_DIR}. F-stat={float(ftest.fvalue):.4f}, p={float(ftest.pvalue):.4g}")
