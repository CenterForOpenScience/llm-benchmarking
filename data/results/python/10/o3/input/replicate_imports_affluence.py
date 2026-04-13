import pandas as pd, numpy as np
from linearmodels import PanelOLS
import statsmodels.api as sm

# Paths
DATA_PATH = '/app/data/finaldata_noNA.csv'  # mounted path in container
OUTPUT_PATH = '/app/data/replication_results.csv'


def preprocess(df):
    """Replicate core transformations from KMYR.do in Python."""
    # encode country to numeric
    df['countrynum'] = pd.factorize(df['country'])[0]

    # National Affluence
    df['NAff'] = df['gdp'] / df['pop']

    # Imports from South (% of GDP) (note: gdp variable is in billions? dividing by 10000 replicates Stata scaling)
    df['IMS'] = df['totalimport'] / (df['gdp'] * 10000)

    # Exports to South
    df['EXS'] = df['totalexport'] / (df['gdp'] * 10000)

    # Time dummies (5-year bands)
    for start in range(1970, 2000, 5):
        end = start + 4
        df[f'DUM{str(start)[-2:]}to{str(end)[-2:]}'] = ((df['year'] >= start) & (df['year'] <= end)).astype(int)

    # Additional dummies for 2000-18 (00-04, 05-09, 10-14, 15-18)
    _bands = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2018)]
    for (start, end) in _bands:
        df[f'DUM{str(start)[-2:]}to{str(end)[-2:]}'] = ((df['year'] >= start) & (df['year'] <= end)).astype(int)

    # sort for lags
    df = df.sort_values(['countrynum', 'year'])

    # one-year lags of IMS, EXS, unemp
    df['IMS_lag1'] = df.groupby('countrynum')['IMS'].shift(1)
    df['EXS_lag1'] = df.groupby('countrynum')['EXS'].shift(1)
    df['unemp_lag1'] = df.groupby('countrynum')['unemp'].shift(1)

    # Drop rows with missing due to lag
    df = df.dropna(subset=['IMS_lag1', 'EXS_lag1', 'unemp_lag1'])
    return df


def run_regression(df):
    # Set multiindex for linearmodels PanelOLS
    df = df.set_index(['countrynum', 'year'])

    # Outcome and regressors
    y = df['NAff']

    # Build exogenous matrix
    exog_vars = ['IMS_lag1', 'EXS_lag1', 'unemp_lag1'] + [c for c in df.columns if c.startswith('DUM')]
    X = sm.add_constant(df[exog_vars])

    model = PanelOLS(y, X, entity_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    return res


def main():
    df_raw = pd.read_csv(DATA_PATH)
    df = preprocess(df_raw)
    res = run_regression(df)
    print(res.summary)

    # Save key coefficients to CSV
    coef_table = res.params.to_frame('coef')
    coef_table['std_err'] = res.std_errors
    coef_table.to_csv(OUTPUT_PATH)

if __name__ == '__main__':
    main()
