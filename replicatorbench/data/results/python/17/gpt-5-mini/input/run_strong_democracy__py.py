"""
Run the 'long' specification on the subset of strong democracies (democ >= 9), matching the focal hypothesis.
Saves output to /app/data/.../replication_data/output/real_strongdemoc.txt
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

RANDOM_SEED = 2020


def run(drop_wave_7=True):
    data_path = os.path.join("/app/data", "data/original/17/0205_python_gpt5-mini/replication_data/data.csv")
    out_dir = os.path.join("/app/data", "data/original/17/0205_python_gpt5-mini/replication_data/output")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    if drop_wave_7:
        df = df[df.get("wave") != 7]

    # Filter to strong democracies (Polity IV >= 9) - the dataset column 'democ' corresponds to Polity-like score
    strong = df[df.get("democ") >= 9].copy()

    # Define model variables
    y_var = "gov_consumption"
    x_vars = ["sd_gov", "mean_gov", "africa", "laam", "asiae", "col_uka", "col_espa", "col_otha", "federal", "oecd", "log_gdp_per_capita", "trade_share", "age_15_64", "age_65_plus"]

    model_df = strong[[y_var] + x_vars].dropna()

    X = sm.add_constant(model_df[x_vars])
    y = model_df[y_var]

    ols = sm.OLS(y, X).fit()
    robust = ols.get_robustcov_results(cov_type="HC1")

    coeffs = robust.params
    bse = robust.bse
    tvals = robust.tvalues
    pvals = robust.pvalues

    out_df = pd.DataFrame({"Estimate": coeffs, "Std. Error": bse, "t value": tvals, "Pr(>|t|)": pvals})

    obs_row = pd.DataFrame({"Estimate": [len(ols.resid)], "Std. Error": [""], "t value": [""], "Pr(>|t|)": [""]}, index=["Obs."])
    r2_row = pd.DataFrame({"Estimate": [ols.rsquared], "Std. Error": [""], "t value": [""], "Pr(>|t|)": [""]}, index=["R-squared"]) 

    out_df = pd.concat([out_df, obs_row, r2_row])

    out_path = os.path.join(out_dir, f"real_strongdemoc.txt")
    out_df.to_csv(out_path, sep='\t', float_format='%.6f')
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    run(drop_wave_7=True)
