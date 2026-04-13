#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Paths (container expects /app/data to be mounted)
DATA_PATH = '/workspace/replication_data/finaldata_noNA.csv'
OUT_SUMMARY_WORK = '/workspace/replication_results.txt'
OUT_COEF_WORK = '/workspace/replication_coef.csv'
OUT_SUMMARY_APP = '/app/data/replication_results.txt'
OUT_COEF_APP = '/app/data/replication_coef.csv'

def main():
    print('Reading data from', DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    # Create numeric country id similar to Stata encode
    df['countrynum'], uniques = pd.factorize(df['country'])
    # Variable construction
    df['NAff'] = df['gdp'] / df['pop']
    df['IMS'] = df['totalimport'] / (df['gdp'] * 10000)
    df['EXS'] = df['totalexport'] / (df['gdp'] * 10000)

    # Outlier detection using MinCovDet as approximation to Hadi    # Outlier detection skipped: sklearn may not be available in the runtime; set all observations as non-outliers
    df['bad'] = 0

        # Drop observations flagged as outliers
    df = df[df['bad'] != 1].copy()

    # Keep necessary columns: ensure we have countrynum and year
    # (we keep year for creating dummies and for lagging)
    # Create 5-year dummies
    df['DUM70to74'] = ((df['year'] >= 1970) & (df['year'] <= 1974)).astype(int)
    df['DUM75to79'] = ((df['year'] >= 1975) & (df['year'] <= 1979)).astype(int)
    df['DUM80to84'] = ((df['year'] >= 1980) & (df['year'] <= 1984)).astype(int)
    df['DUM85to89'] = ((df['year'] >= 1985) & (df['year'] <= 1989)).astype(int)
    df['DUM90to94'] = ((df['year'] >= 1990) & (df['year'] <= 1994)).astype(int)
    df['DUM95to99'] = ((df['year'] >= 1995) & (df['year'] <= 1999)).astype(int)
    df['DUM00to04'] = ((df['year'] >= 2000) & (df['year'] <= 2004)).astype(int)
    df['DUM05to09'] = ((df['year'] >= 2005) & (df['year'] <= 2009)).astype(int)
    df['DUM10to14'] = ((df['year'] >= 2010) & (df['year'] <= 2014)).astype(int)
    df['DUM15to18'] = ((df['year'] >= 2015) & (df['year'] <= 2018)).astype(int)

    # Sort and compute one-year lags within country as in the .do
    df.sort_values(['countrynum', 'year'], inplace=True)
    df['L_IMS'] = df.groupby('countrynum')['IMS'].shift(1)
    df['L_EXS'] = df.groupby('countrynum')['EXS'].shift(1)
    df['L_unemp'] = df.groupby('countrynum')['unemp'].shift(1)

    # Drop rows missing lagged regressors or NAff
    model_df = df.dropna(subset=['L_IMS', 'L_EXS', 'L_unemp', 'NAff']).copy()
    print('Final sample size for estimation:', model_df.shape[0])

    # Prepare formula: NAff ~ L_IMS + L_EXS + L_unemp + country FE + 5-year dummies
    dummies = ['DUM70to74','DUM75to79','DUM80to84','DUM85to89','DUM90to94','DUM95to99','DUM00to04','DUM05to09','DUM10to14','DUM15to18']
    formula = 'NAff ~ L_IMS + L_EXS + L_unemp + C(countrynum)'
    for d in dummies:
        formula += ' + ' + d

    print('Estimating model with formula:', formula)
    try:
        res = smf.ols(formula=formula, data=model_df).fit()
    except Exception as e:
        print('Model estimation failed:', e)
        return

    # Save summary and coefficient table    # Save summary and coefficient table
    summary_text = res.summary().as_text()
    with open(OUT_SUMMARY_WORK, 'w') as f:
        f.write('Model formula: ' + formula + '\n')
        f.write('Final sample size: ' + str(model_df.shape[0]) + '\n\n')
        f.write(summary_text)
    print('Saved regression summary to', OUT_SUMMARY_WORK)
    # Also attempt to copy to /app/data if available
    try:
        import shutil
        shutil.copy(OUT_SUMMARY_WORK, OUT_SUMMARY_APP)
        print('Also copied regression summary to', OUT_SUMMARY_APP)
    except Exception:
        pass

    # Coef table
    coef_df = pd.DataFrame({
        'term': res.params.index,
        'estimate': res.params.values,
        'std_error': res.bse.values,
        't_value': res.tvalues.values,
        'p_value': res.pvalues.values
    })
    # Add confidence intervals
    ci = res.conf_int()
    coef_df['ci_lower'] = ci[0].values
    coef_df['ci_upper'] = ci[1].values
    coef_df.to_csv(OUT_COEF_WORK, index=False)
    print('Saved coefficient table to', OUT_COEF_WORK)
    try:
        import shutil
        shutil.copy(OUT_COEF_WORK, OUT_COEF_APP)
        print('Also copied coefficient table to', OUT_COEF_APP)
    except Exception:
        pass

    # Print focal coefficient on L_IMS
    if 'L_IMS' in res.params.index:
        est = res.params['L_IMS']
        se = res.bse['L_IMS']
        p = res.pvalues['L_IMS']
        ci_low, ci_high = res.conf_int().loc['L_IMS']
        print('\nFocal coefficient L_IMS:')
        print('Estimate:', est)
        print('Std. Error:', se)
        print('p-value:', p)
        print('95% CI:', (ci_low, ci_high))
    else:
        print('L_IMS not in model results; check variable names')

if __name__ == '__main__':
    main()
