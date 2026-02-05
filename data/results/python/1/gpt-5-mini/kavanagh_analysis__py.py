"""
Python translation of kavanagh_analysis.R
- Reads county_variables.csv and transportation.csv from /app/data/original/1/python/replication_data/
- Mirrors the R data processing and runs OLS regressions with state fixed effects (via C(state)).
- Saves regression summaries and simple HTML/text outputs to /app/data

Notes:
- This script intentionally samples 5% of counties (to match the original R script behavior) using seed 2982.
- All I/O paths use /app/data to comply with the replication environment.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import datetime
import os

# Input paths (assumed available at /app/data)
BASE = '/app/data/original/1/python/replication_data'
COUNTY_FILE = os.path.join(BASE, 'county_variables.csv')
TRANSPORT_FILE = os.path.join(BASE, 'transportation.csv')

# Output paths
OUT_DIR = '/app/data'
REG_TABLE_TXT = os.path.join(OUT_DIR, 'regression_table.txt')
REG_TABLE_HTML = os.path.join(OUT_DIR, 'regression_table.html')
SPATIAL_REG_HTML = os.path.join(OUT_DIR, 'spatial_regression_table.html')

def main():
    # Load data
    county_variables = pd.read_csv(COUNTY_FILE)
    # sample 5% as the original R script does
    county_variables = county_variables.sample(frac=0.05, random_state=2982)
    transportation = pd.read_csv(TRANSPORT_FILE)

    # Parse dates
    transportation['date'] = pd.to_datetime(transportation['date'])

    # Compute prop_home
    transportation['prop_home'] = transportation['pop_home'] / (transportation['pop_home'] + transportation['pop_not_home'])

    # Define time periods
    def time_period(dt):
        if pd.Timestamp('2020-02-16') <= dt <= pd.Timestamp('2020-02-29'):
            return 'AAA Reference'
        if pd.Timestamp('2020-03-19') <= dt <= pd.Timestamp('2020-04-01'):
            return 'March'
        if pd.Timestamp('2020-08-16') <= dt <= pd.Timestamp('2020-08-29'):
            return 'August'
        return None

    transportation['time_period'] = transportation['date'].apply(time_period)

    # Filter
    flat = transportation[transportation['time_period'].notna() & transportation['prop_home'].notna()].copy()

    # Average over county, time period
    flat = flat.groupby(['time_period', 'fips', 'state'], as_index=False).agg({'prop_home': 'mean'})

    # Sort and compute relative change to reference
    flat = flat.sort_values(['state','fips','time_period'])

    # For each fips,state, compute prop_home_change = 100*(prop_home / first(prop_home) -1)
    # Pivot so we can compute first(prop_home) per group (the reference)
    flat['prop_home_change'] = np.nan
    # We'll compute per (fips,state)
    def compute_changes(group):
        # ensure sorted by time_period so AAA Reference should be present before others
        # But rely on existence of AAA Reference
        try:
            ref = group.loc[group['time_period']=='AAA Reference', 'prop_home'].values[0]
        except IndexError:
            # no reference, set NaN
            group['prop_home_change'] = np.nan
            return group
        group['prop_home_change'] = 100.0 * (group['prop_home'] / ref - 1.0)
        return group

    flat = flat.groupby(['fips','state'], group_keys=False).apply(compute_changes)

    # Keep only March and August
    flat = flat[flat['time_period'] != 'AAA Reference'].copy()

    # Pivot wider
    prop_cols = flat.pivot(index=['fips','state'], columns='time_period', values='prop_home')
    change_cols = flat.pivot(index=['fips','state'], columns='time_period', values='prop_home_change')

    # Rename columns to match R output
    prop_cols.columns = [f'prop_home_{c}' for c in prop_cols.columns]
    change_cols.columns = [f'prop_home_change_{c}' for c in change_cols.columns]

    flat_wide = prop_cols.join(change_cols).reset_index()

    # Right join county_variables by fips
    merged = pd.merge(flat_wide, county_variables, on='fips', how='right', suffixes=('','_y'))

    # Compute trumpIQR from county_variables (full file) as the R did
    # Use original (unsampled) county_variables file to compute IQR to mirror R code
    full_county = pd.read_csv(COUNTY_FILE)
    trump_series = full_county['trump_share'].dropna().unique()
    # Use pandas quantile on the trump_share column
    trump_q = full_county['trump_share'].dropna().quantile([0.25,0.75])
    trumpIQR = float(trump_q.loc[0.75] - trump_q.loc[0.25])

    # Variable construction similar to R script
    df = merged.copy()
    # Ensure state is treated as categorical
    df['state'] = df['state'].astype('category')

    # Select relevant columns and transform percentages
    # The R script selected a particular set; we'll include the main variables used there if present
    # Make safe accesses with .get
    sel_cols = [
        'prop_home_change_March',
        'prop_home_change_August',
        'income_per_capita',
        'trump_share',
        'male_percent',
        'percent_black',
        'percent_hispanic',
        'percent_college',
        'percent_retail',
        'percent_transportation',
        'percent_hes',
        'percent_rural',
        'ten_nineteen',
        'twenty_twentynine',
        'thirty_thirtynine',
        'forty_fortynine',
        'fifty_fiftynine',
        'sixty_sixtynine',
        'seventy_seventynine',
        'over_eighty',
        'state',
        'fips'
    ]

    # Some of these columns may not exist in the merged file if column names differ; keep intersection
    sel_cols = [c for c in sel_cols if c in df.columns]
    df = df[sel_cols].copy()

    # Transform percent_ columns: in R they multiplied starts_with('percent_') by 100, then male_percent*100, then percent_college = percent_college/100
    percent_cols = [c for c in df.columns if c.startswith('percent_')]
    for c in percent_cols:
        df[c] = df[c] * 100.0
    if 'male_percent' in df.columns:
        df['male_percent'] = df['male_percent'] * 100.0
    if 'percent_college' in df.columns:
        # R divided percent_college by 100
        df['percent_college'] = df['percent_college'] / 100.0

    if 'income_per_capita' in df.columns:
        df['income_per_capita'] = df['income_per_capita'] / 1000.0

    # Drop rows with missing dependent variable for March (to mirror felm which uses available obs)
    df_march = df.dropna(subset=['prop_home_change_March']).copy()
    df_august = df.dropna(subset=['prop_home_change_August']).copy()

    # Helper to create formula string excluding certain columns
    def formula_maker(depvar, data):
        exclude = {'fips', 'prop_home_change_March', 'prop_home_change_August', 'state'}
        vnames = [c for c in data.columns if c not in exclude and c != depvar]
        # Build formula with state fixed effects via C(state)
        rhs = ' + '.join(vnames + ['C(state)'])
        formula = f"{depvar} ~ {rhs}"
        return formula

    # Run OLS regressions
    results = {}
    if not df_march.empty:
        form_m1 = formula_maker('prop_home_change_March', df_march)
        m1 = smf.ols(formula=form_m1, data=df_march).fit()
        results['m1'] = m1
    else:
        print('No data for March regression')

    if not df_august.empty:
        form_m2 = formula_maker('prop_home_change_August', df_august)
        m2 = smf.ols(formula=form_m2, data=df_august).fit()
        results['m2'] = m2
    else:
        print('No data for August regression')

    # Save regression summaries
    with open(REG_TABLE_TXT, 'w') as f:
        for name, mod in results.items():
            f.write(f"Model: {name}\n")
            f.write(mod.summary().as_text())
            f.write('\n\n')

    # Simple HTML output
    with open(REG_TABLE_HTML, 'w') as f:
        f.write('<html><body>')
        for name, mod in results.items():
            f.write(f'<h2>Model: {name}</h2>')
            f.write(f'<pre>{mod.summary().as_text()}</pre>')
        f.write('</body></html>')

    # Effect of a one-IQR change in Trump share: compute estimate and Wald test
    def iqr_effect_test(mod, varname='trump_share'):
        if varname not in mod.params.index:
            return None
        beta = mod.params[varname]
        se = mod.bse[varname]
        est = trumpIQR * beta
        est_se = trumpIQR * se
        tstat = est / est_se if est_se != 0 else np.nan
        pval = 2 * (1 - sm.stats.t.cdf(abs(tstat), df=mod.df_resid)) if not np.isnan(tstat) else np.nan
        return {'beta': beta, 'se': se, 'trumpIQR': trumpIQR, 'est_scaled': est, 'est_se_scaled': est_se, 'tstat': tstat, 'pval': pval}

    iqr_results = {}
    for name, mod in results.items():
        res = iqr_effect_test(mod, varname='trump_share')
        iqr_results[name] = res

    # Save IQR results
    iqr_out = os.path.join(OUT_DIR, 'trump_iqr_effect.json')
    import json
    with open(iqr_out, 'w') as f:
        json.dump(iqr_results, f, indent=2)

    print('Done. Outputs saved to /app/data: regression_table.txt, regression_table.html, trump_iqr_effect.json')

if __name__ == '__main__':
    main()
