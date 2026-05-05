import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Paths# Paths
candidates = [
    '/app/data/original/19/0205_gpt5-mini/replication_data/gelfand_replication_data.csv',
    '/app/data/data/original/19/0205_gpt5-mini/replication_data/gelfand_replication_data.csv',
    '/app/data/replication_data/gelfand_replication_data.csv',
    '/workspace/replication_data/gelfand_replication_data.csv',
    'data/original/19/0205_gpt5-mini/replication_data/gelfand_replication_data.csv'
]
DATA_CSV = None
for p in candidates:
    if os.path.exists(p):
        DATA_CSV = p
        break
if DATA_CSV is None:
    # fallback to original expected path (may fail later with informative error)
    DATA_CSV = candidates[0]

OUT_SLOPES = '/app/data/estimatedcoefficients_replication.csv'
OUT_SUMMARY = '/app/data/regression_summary_replication.txt'

# Read data# Read data
print('Reading CSV from', DATA_CSV)
df = pd.read_csv(DATA_CSV)
print('Initial shape:', df.shape)

# Drop specified countries
exclude = ['Belgium','France','New Zealand','Norway','Pakistan','Venezuela']
df = df[~df['country'].isin(exclude)].copy()
print('After exclusion shape:', df.shape)

# Parse date
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    df['date'] = pd.to_datetime(df[['year','month','day']])

# Ensure sorting
df = df.sort_values(['country','date'])

# Forward fill total_covid_per_million and gdp within each country after reindexing to continuous dates
out_rows = []
for country, g in df.groupby('country'):
    g = g.set_index('date')
    # reindex to continuous daily index between min and max
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
    g = g.reindex(full_idx)
    # bring back country-level static vars
    for col in ['country','tightness','efficiency','gdp','gini_val','alternative_gini','median_age','geoId','countryterritoryCode','popData2019']:
        if col in df.columns:
            g[col] = g[col].fillna(method='ffill').fillna(method='bfill')
    # forward fill totals
    if 'total_covid_per_million' in g.columns:
        g['total_covid_per_million'] = g['total_covid_per_million'].fillna(method='ffill')
    # reset index
    g = g.reset_index().rename(columns={'index':'date'})
    out_rows.append(g)

df2 = pd.concat(out_rows, ignore_index=True)
print('After reindex+ffill shape:', df2.shape)

# Keep rows where total_covid_per_million > 1 (first day threshold)
if 'total_covid_per_million' not in df2.columns:
    raise RuntimeError('total_covid_per_million column missing')

# Create helper to compute days since first >1
res_rows = []
for country, g in df2.groupby('country'):
    g = g.sort_values('date').copy()
    mask = g['total_covid_per_million'] > 1
    if mask.any():
        first_date = g.loc[mask, 'date'].iloc[0]
        g = g[g['date'] >= first_date].copy()
        g['time'] = (g['date'] - first_date).dt.days + 1
        g = g[g['time'] <= 30]
        g['ltotalcases'] = np.log(g['total_covid_per_million'].replace(0, np.nan))
        res_rows.append(g)

df3 = pd.concat(res_rows, ignore_index=True)
print('After threshold and 30-day cut shape:', df3.shape)

# First-stage: per-country OLS of ltotalcases ~ time to get slope
slopes = []
for country, g in df3.groupby('country'):
    g = g.dropna(subset=['ltotalcases','time'])
    if len(g) >= 2:
        X = sm.add_constant(g['time'])
        y = g['ltotalcases']
        try:
            model = sm.OLS(y, X).fit()
            slope = model.params.get('time', np.nan)
            slopes.append({'country': country, 'slope': slope, 'n_obs': len(g)})
        except Exception as e:
            slopes.append({'country': country, 'slope': np.nan, 'n_obs': len(g)})
    else:
        slopes.append({'country': country, 'slope': np.nan, 'n_obs': len(g)})

slopes_df = pd.DataFrame(slopes)
print('Slopes computed for countries:', slopes_df.shape)
slopes_df.to_csv(OUT_SLOPES, index=False)
print('Saved slopes to', OUT_SLOPES)

# Construct country-level dataset: take first row after threshold per country and merge slope
first_rows = df3.sort_values(['country','date']).groupby('country').first().reset_index()
country_df = first_rows.merge(slopes_df, on='country', how='left')

# Prepare gini fallback
if 'gini_val' in country_df.columns:
    country_df['gini'] = country_df['gini_val']
else:
    country_df['gini'] = np.nan
country_df['gini'] = country_df['gini'].fillna(country_df.get('alternative_gini', np.nan))

# Compute interaction
country_df['eff_tight'] = country_df['efficiency'] * country_df['tightness']

# Select variables for regression
regs = country_df[['country','slope','eff_tight','gdp','gini','median_age','efficiency','tightness']].copy()
regs = regs.dropna(subset=['slope','eff_tight','gdp','gini','median_age','efficiency','tightness'])
print('Country-level regression sample size:', regs.shape)

# Second-stage OLS: slope ~ eff_tight + gdp + gini + median_age + efficiency + tightness
Y = regs['slope']
X = regs[['eff_tight','gdp','gini','median_age','efficiency','tightness']]
X = sm.add_constant(X)
model2 = sm.OLS(Y, X).fit()

# Save regression summary
with open(OUT_SUMMARY, 'w') as f:
    f.write(model2.summary().as_text())

print('Saved regression summary to', OUT_SUMMARY)
print(model2.summary())
