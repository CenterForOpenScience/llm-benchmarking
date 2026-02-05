"""
Translated analysis script from analyze.R to Python.
Reads data from /app/data/.../replication_data/data.csv and writes output tables to /app/data/.../replication_data/output/
Mimics R behavior: seed=2020; test mode randomizes columns 3..end by shuffling each column independently.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

RANDOM_SEED = 2020


def run(mode="test", drop_wave_7=True):
    # Paths
    possible_paths = [
        os.path.join("/app/data", "data/original/17/0205_python_gpt5-mini/replication_data/data.csv"),
        os.path.join("/app/data", "original/17/0205_python_gpt5-mini/replication_data/data.csv"),
        os.path.join(os.path.dirname(__file__), "data.csv"),
        os.path.join(os.getcwd(), "replication_data", "data.csv"),
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        # default to the first candidate (will raise FileNotFoundError with helpful path)
        data_path = possible_paths[0]

    out_dir = os.path.join(os.path.dirname(data_path), "output")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    if drop_wave_7:
        df = df[df.get("wave") != 7]
    drop = "drop_7" if drop_wave_7 else "keep_7"

    if mode == "test":
        # Randomize each column independently starting from column index 2 (0-based), similar to R code that randomizes columns 3:ncol
        rng = np.random.RandomState(RANDOM_SEED)
        cols = df.columns.tolist()
        for col in cols[2:]:
            try:
                # shuffle preserving NaNs
                values = df[col].values.copy()
                idx = np.arange(len(values))
                rng.shuffle(idx)
                df[col] = values[idx]
            except Exception:
                # if non-shuffleable, skip
                pass

    # Define model variables to match the R formula
    y_var = "gov_consumption"
    x_vars = ["sd_gov", "mean_gov", "africa", "laam", "asiae", "col_uka", "col_espa", "col_otha", "federal", "oecd", "log_gdp_per_capita", "trade_share", "age_15_64", "age_65_plus"]

    # Subset data to relevant columns and drop rows with missing values in model vars
    model_df = df[[y_var] + x_vars].copy()
    model_df = model_df.dropna()

    X = sm.add_constant(model_df[x_vars])
    y = model_df[y_var]

    ols = sm.OLS(y, X).fit()
    robust = ols.get_robustcov_results(cov_type="HC1")

    # Build output table similar to coeftest: coef, std.err, t, p
    coeffs = robust.params
    bse = robust.bse
    tvals = robust.tvalues
    pvals = robust.pvalues

    out_df = pd.DataFrame({"Estimate": coeffs, "Std. Error": bse, "t value": tvals, "Pr(>|t|)": pvals})

    # Append Obs. and R-squared rows
    obs_row = pd.DataFrame({"Estimate": [len(ols.resid)], "Std. Error": [""], "t value": [""], "Pr(>|t|)": [""]}, index=["Obs."])
    r2_row = pd.DataFrame({"Estimate": [ols.rsquared], "Std. Error": [""], "t value": [""], "Pr(>|t|)": [""]}, index=["R-squared"]) 

    out_df = pd.concat([out_df, obs_row, r2_row])

    out_path = os.path.join(out_dir, f"{mode}_{drop}.txt")
    out_df.to_csv(out_path, sep='\t', float_format='%.6f')
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    # Run the four combinations as in analyze.R
    run(mode="test", drop_wave_7=True)
    run(mode="test", drop_wave_7=False)
    run(mode="real", drop_wave_7=True)
    run(mode="real", drop_wave_7=False)
