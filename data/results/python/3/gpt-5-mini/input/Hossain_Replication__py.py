import os
import re
import json

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), 'COVID replication.dta')
OUTPUT_SUMMARY = '/app/data/Hossain_Replication_results.txt'
OUTPUT_JSON = '/app/data/Hossain_Replication_coefs.json'

# Read data
print('Reading data from', DATA_PATH)
# Try to use pandas to read .dta if available and compatible
try:
    import pandas as pd
    try:
        df = pd.read_stata(DATA_PATH)
        print('Read .dta using pandas')
    except Exception as e:
        print('pandas.read_stata failed with:', e)
        raise
except Exception:
    # pandas is not available or failed; fallback to pyreadstat and ensure compatible numpy installed
    print('Falling back to pyreadstat to read the Stata file...')
    import site
    try:
        site.addsitedir(site.USER_SITE)
    except Exception:
        pass
    try:
        import pyreadstat
    except Exception:
        # Ensure numpy has legacy aliases expected by some C extensions
        try:
            import numpy as _np
            if not hasattr(_np, 'float'):
                _np.float = float
            if not hasattr(_np, 'int'):
                _np.int = int
        except Exception:
            pass
        try:
            import pyreadstat
        except Exception:
            import sys, subprocess
            print('Installing numpy==1.23.5 and pyreadstat==1.1.4...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.23.5'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyreadstat==1.1.4'])
            try:
                site.addsitedir(site.USER_SITE)
            except Exception:
                pass
            import importlib
            importlib.invalidate_caches()
            import numpy as _np
            if not hasattr(_np, 'float'):
                _np.float = float
            if not hasattr(_np, 'int'):
                _np.int = int
            import pyreadstat
    df, meta = pyreadstat.read_dta(DATA_PATH)
    # after using pyreadstat, import pandas and statsmodels for later steps
    import importlib
    importlib.invalidate_caches()
    import pandas as pd
    import statsmodels.api as sm

print('Initial columns:', list(df.columns)[:30])

# Helper to find column by pattern
def find_col(df, patterns):
    cols = df.columns
    for p in patterns:
        for c in cols:
            if re.search(p, c, flags=re.I):
                return c
    return None

# Select dataset = 1 (cases up to 04-03-2020)
cov_patterns = [r'04[\._]03', r'12[\._]31[\._]04[\._]03', r'12\.31_04\.03', r'COVID.*04']
cov_col = find_col(df, cov_patterns)
if cov_col is None:
    raise ValueError('Could not find COVID cases column for 04-03-2020. Available columns: ' + ','.join(df.columns))
print('Using COVID column:', cov_col)

df['total_cases'] = df[cov_col]

# Population variable
pop_col = find_col(df, [r'pop', r'population', r'popData2019'])
if pop_col is None:
    raise ValueError('Could not find population column. Columns: ' + ','.join(df.columns))
print('Using population column:', pop_col)

df['cases_per_million'] = df['total_cases'] / df[pop_col] * 1e6

# Democracy variable (find column containing Democracy)
dem_col = find_col(df, [r'democracy', r'democrat'])
if dem_col is None:
    raise ValueError('Could not find democracy column. Columns: ' + ','.join(df.columns))
print('Using democracy column:', dem_col)
# Convert to numeric if needed
try:
    df['democracy_raw'] = pd.to_numeric(df[dem_col], errors='coerce')
except Exception:
    df['democracy_raw'] = df[dem_col]
# Original do-file divides Gapminder scores by 10 to match EIU 0-10 scale
# The replication .do replaces democracy = democracy/10
# We follow the same operation
df['democracy'] = df['democracy_raw'] / 10.0

# Temperature variable
temp_col = find_col(df, [r'temp', r'Annual_temp', r'temperature'])
if temp_col is None:
    raise ValueError('Could not find temperature column. Columns: ' + ','.join(df.columns))
print('Using temperature column:', temp_col)
df['temperature'] = pd.to_numeric(df[temp_col], errors='coerce')

# Openness variable (trade_2016, trade.2016, imputed_trade, trade_recent)
op_col = find_col(df, [r'trade[_\.]?2016', r'imputed_trade', r'trade[_\.]?recent', r'openness'])
if op_col is None:
    # fallback: try any column with trade
    op_col = find_col(df, [r'trade'])
if op_col is None:
    raise ValueError('Could not find trade/openness column. Columns: ' + ','.join(df.columns))
print('Using openness column:', op_col)
df['openness'] = pd.to_numeric(df[op_col], errors='coerce')

# Keep necessary columns
model_df = df[['cases_per_million', 'democracy', 'temperature', 'openness']].copy()
# Drop missing
model_df = model_df.dropna()
print('Final sample size for regression:', model_df.shape[0])

# Regression: cases_per_million ~ democracy + temperature + openness
Y = model_df['cases_per_million']
X = model_df[['democracy', 'temperature', 'openness']]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
summary = model.summary().as_text()
print(summary)

# Save outputs
with open(OUTPUT_SUMMARY, 'w') as f:
    f.write('Regression summary for Hossain replication\n')
    f.write(summary)

coefs = {
    'params': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'nobs': int(model.nobs),
    'rsquared_adj': model.rsquared_adj
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(coefs, f, indent=2)

print('Saved summary to', OUTPUT_SUMMARY)
print('Saved coefficients to', OUTPUT_JSON)
