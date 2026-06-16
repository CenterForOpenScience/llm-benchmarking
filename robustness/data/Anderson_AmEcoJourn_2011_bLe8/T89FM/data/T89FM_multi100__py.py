# Python translation of T89FM_multi100.R
# Ensures all IO uses /app/data with fallback to subfolder path.

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices


def load_household_data():
    candidates = [
        "/app/data/household.dta",
        "/app/data/AEJApp-2009-0289-data/household.dta"
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_stata(path, convert_categoricals=False)
    raise FileNotFoundError("household.dta not found under /app/data or /app/data/AEJApp-2009-0289-data")


df = load_household_data()
# Remove observations where domlow is missing
if "domlow" not in df.columns:
    raise ValueError("Expected column 'domlow' not found in dataset.")

df = df.copy()
df = df[~pd.isna(df["domlow"])].reset_index(drop=True)

# Cast some variables similar to R factors
# Keep domlow as categorical for regression consistent with R factor coding
# Ensure 'caste', 'borrow', and 'electric' are categorical if present
for cat_col in ["caste", "borrow", "electric"]:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype("category")

# Prepare domlow categorical variable label (0/1 assumed)
df["domlow_cat"] = df["domlow"].astype("category")

# Log transform total income
if "totinc" not in df.columns:
    raise ValueError("Expected column 'totinc' not found in dataset.")

df = df.copy()
# Guard against non-positive incomes when taking log
if (df["totinc"] <= 0).any():
    # Replace non-positive with NaN to be dropped by models
    df.loc[df["totinc"] <= 0, "totinc"] = np.nan

# One-sided t-test: log(totinc) for domlow==1 greater than domlow==0
mask1 = df["domlow"] == 1
mask0 = df["domlow"] == 0
log_totinc_1 = np.log(df.loc[mask1, "totinc"]).dropna()
log_totinc_0 = np.log(df.loc[mask0, "totinc"]).dropna()

if len(log_totinc_1) > 1 and len(log_totinc_0) > 1:
    t_res = stats.ttest_ind(log_totinc_1, log_totinc_0, equal_var=False, alternative="greater")
    print("Task1 t-test (one-sided, greater) on log(totinc), domlow==1 vs 0:")
    print({"statistic": t_res.statistic, "pvalue": t_res.pvalue, "n1": int(log_totinc_1.shape[0]), "n0": int(log_totinc_0.shape[0])})
else:
    print("Insufficient data for t-test after log transform and group split.")

# Build regression models mirroring the R script
# Model 1
model1_formula = (
    "np.log(totinc) ~ bihar + literate + C(borrow) + totland + landirr + "
    "dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist8 + dist9 + dist10 + dist11 + "
    "dist12 + dist13 + dist14 + dist16 + dist17 + dist18 + dist19 + dist20 + dist21 + "
    "dist22 + dist23 + dist24 + dist25 + area + pmixdominant + tolaparea + "
    "gwdevelopment + rainfall + gwavailability + river + canal + noalkal + "
    "nolog + nosoil + noflood + paddy + wheat + cereal + pulse + bulb + seed + cash + "
    "bus3 + tele3 + ps3 + pds3 + bank3 + pps3 + ms3 + ss3 + phc3 + hosp3 + electric + hhelec + "
    "C(caste) + C(domlow_cat)"
)

# Fit with robust (HC0) SE
try:
    m1 = smf.ols(model1_formula, data=df).fit(cov_type='HC0')
    print("Task1 Model1 OLS with HC0 SE summary (truncated):")
    if 'C(domlow_cat)[T.1.0]' in m1.params.index:
        key = 'C(domlow_cat)[T.1.0]'
    else:
        # Fallback naming if category coded differently
        domlow_keys = [k for k in m1.params.index if k.startswith('C(domlow_cat)') and 'T.' in k]
        key = domlow_keys[0] if domlow_keys else None
    if key is not None:
        print({
            "coef_domlow": float(m1.params.get(key, np.nan)),
            "se_domlow": float(m1.bse.get(key, np.nan)),
            "t_domlow": float(m1.tvalues.get(key, np.nan)),
            "p_domlow": float(m1.pvalues.get(key, np.nan))
        })
    else:
        print("domlow term not found in Model1 coefficients.")
except Exception as e:
    print("Model1 failed:", repr(e))

# Model 2
model2_formula = (
    "np.log(totinc) ~ bihar + literate + C(borrow) + totland + landirr + "
    "dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist8 + dist9 + dist11 + "
    "dist12 + dist13 + dist14 + dist16 + dist17 + dist18 + dist19 + dist20 + dist21 + "
    "dist22 + dist23 + dist24 + area + pmixdominant + tolaparea + "
    "river + canal + noalkal + nolog + nosoil + noflood + paddy + wheat + cereal + pulse + bulb + seed + cash + "
    "bus3 + tele3 + ps3 + pds3 + bank3 + pps3 + ms3 + ss3 + phc3 + hosp3 + electric + hhelec + "
    "C(caste) + C(domlow_cat)"
)

try:
    m2 = smf.ols(model2_formula, data=df).fit(cov_type='HC0')
    print("Task1 Model2 OLS with HC0 SE summary (truncated):")
    if 'C(domlow_cat)[T.1.0]' in m2.params.index:
        key2 = 'C(domlow_cat)[T.1.0]'
    else:
        domlow_keys2 = [k for k in m2.params.index if k.startswith('C(domlow_cat)') and 'T.' in k]
        key2 = domlow_keys2[0] if domlow_keys2 else None
    if key2 is not None:
        print({
            "coef_domlow": float(m2.params.get(key2, np.nan)),
            "se_domlow": float(m2.bse.get(key2, np.nan)),
            "t_domlow": float(m2.tvalues.get(key2, np.nan)),
            "p_domlow": float(m2.pvalues.get(key2, np.nan))
        })
    else:
        print("domlow term not found in Model2 coefficients.")
except Exception as e:
    print("Model2 failed:", repr(e))

# Model 3 (reduced set akin to backward elimination outcome)
model3_formula = (
    "np.log(totinc) ~ bihar + literate + totland + landirr + "
    "dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist8 + dist9 + dist11 + "
    "dist12 + dist13 + dist14 + dist16 + dist17 + dist18 + dist19 + dist20 + dist21 + dist22 + dist23 + dist24 + area + "
    "river + canal + nolog + nosoil + paddy + wheat + pulse + bulb + seed + cash + ps3 + pds3 + pps3 + ms3 + ss3 + phc3 + "
    "electric + hhelec + C(caste) + C(domlow_cat)"
)

try:
    m3 = smf.ols(model3_formula, data=df).fit(cov_type='HC0')
    print("Task1 Model3 OLS with HC0 SE summary (truncated):")
    if 'C(domlow_cat)[T.1.0]' in m3.params.index:
        key3 = 'C(domlow_cat)[T.1.0]'
    else:
        domlow_keys3 = [k for k in m3.params.index if k.startswith('C(domlow_cat)') and 'T.' in k]
        key3 = domlow_keys3[0] if domlow_keys3 else None
    if key3 is not None:
        print({
            "coef_domlow": float(m3.params.get(key3, np.nan)),
            "se_domlow": float(m3.bse.get(key3, np.nan)),
            "t_domlow": float(m3.tvalues.get(key3, np.nan)),
            "p_domlow": float(m3.pvalues.get(key3, np.nan))
        })
    else:
        print("domlow term not found in Model3 coefficients.")
except Exception as e:
    print("Model3 failed:", repr(e))

# Oaxaca-Blinder Decomposition (twofold, Reimers 1983 with equal weights)
# Using the variable set from the R script's 'decomp_ln'
decomp_formula = (
    "np.log(totinc) ~ bihar + literate + totland + landirr + area + canal + nolog + nosoil + "
    "paddy + wheat + pulse + bulb + seed + cash + ps3 + pds3 + pps3 + ms3 + ss3 + phc3 + C(electric) + hhelec + C(caste)"
)

try:
    # Drop NA rows for needed columns
    yX = dmatrices(decomp_formula, df, return_type='dataframe', NA_action='drop')
    y = yX[0]
    X = yX[1]

    # Align domlow with rows kept
    kept_index = X.index
    df_kept = df.loc[kept_index]
    g = df_kept["domlow"].values
    # Define groups 0 and 1
    mask_g1 = g == 1
    mask_g0 = g == 0

    if mask_g1.sum() > 0 and mask_g0.sum() > 0:
        X1 = X[mask_g1]
        y1 = y[mask_g1]
        X0 = X[mask_g0]
        y0 = y[mask_g0]

        # Fit separate OLS models
        res1 = sm.OLS(y1, X1).fit()
        res0 = sm.OLS(y0, X0).fit()

        # Means (including intercept column)
        Xbar1 = X1.mean(axis=0)
        Xbar0 = X0.mean(axis=0)

        # Coefficients
        b1 = res1.params
        b0 = res0.params

        # Pooled (Reimers equal weights)
        b_star = 0.5 * (b0 + b1)

        # Mean outcomes
        ybar1 = float(y1.mean())
        ybar0 = float(y0.mean())
        diff = ybar1 - ybar0

        # Explained and unexplained
        explained = float((Xbar1 - Xbar0) @ b_star)
        unexplained = float(diff - explained)

        print("Task1 Oaxaca-Blinder (twofold, Reimers w=0.5) on log(totinc):")
        print({"ybar_diff": diff, "explained": explained, "unexplained": unexplained})
    else:
        print("Oaxaca: insufficient observations in one of the groups.")
except Exception as e:
    print("Oaxaca computation failed:", repr(e))
