import pandas as pd, numpy as np, os, sys
from pathlib import Path
import statsmodels.api as sm

# Robust covariance options in statsmodels "HC1" matches STATA's robust

def run_model(df):
    y = df['gov_consumption']
    X = df[['sd_gov', 'mean_gov', 'africa', 'laam', 'asiae',
            'col_uka', 'col_espa', 'col_otha', 'federal', 'oecd',
            'log_gdp_per_capita', 'trade_share', 'age_15_64', 'age_65_plus']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HC1')
    return model

def main(mode="test", drop_wave_7=True):
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data.csv'
    df = pd.read_csv(data_path)

    if drop_wave_7:
        df = df[df['wave'] != 7].copy()
        drop_tag = 'drop_7'
    else:
        drop_tag = 'keep_7'

    if mode == 'test':
        # Randomize each numeric column independently (except identifiers)
        rng = np.random.default_rng(2020)
        cols_to_shuffle = df.columns.difference(['country', 'scode', 'year'])
        for col in cols_to_shuffle:
            shuffled = rng.permutation(df[col].values)
            df[col] = shuffled

    # Drop any rows with missing values in variables used in the model
    model_vars = ['gov_consumption', 'sd_gov', 'mean_gov', 'africa', 'laam', 'asiae',
                  'col_uka', 'col_espa', 'col_otha', 'federal', 'oecd',
                  'log_gdp_per_capita', 'trade_share', 'age_15_64', 'age_65_plus']
    df_model = df.dropna(subset=model_vars).copy()

    if df_model.empty:
        raise ValueError("No observations left after dropping missing values. Check data preprocessing.")

    model = run_model(df_model)

    # Build output similar to R: estimate | std err | t | p
    params = model.params
    se = model.bse
    tvals = model.tvalues
    pvals = model.pvalues

    out_rows = []
    for name in params.index:
        out_rows.append([name, params.loc[name], se.loc[name], tvals.loc[name], pvals.loc[name]])

    # Append Obs and R^2
    out_rows.append(["Obs.", model.nobs, '', '', ''])
    out_rows.append(["R-squared", model.rsquared, '', '', ''])

    out_df = pd.DataFrame(out_rows, columns=['term', 'estimate', 'std_error', 't', 'p'])

    # Ensure output directory
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"{mode}_{drop_tag}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Wrote results to {out_file}")
if __name__ == "__main__":
    # replicate calls from R script
    main(mode='test', drop_wave_7=True)
    main(mode='test', drop_wave_7=False)

    main(mode='real', drop_wave_7=True)
    main(mode='real', drop_wave_7=False)
