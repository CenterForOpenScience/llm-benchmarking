# Python replication of the R analysis in LeBeouf_JournMarketRes_2010...Final.R
# Reads data from /app/data (expected location) and writes outputs to /app/data

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Paths (assume data will be placed in /app/data)
DATA_DIR = '/app/data'
RAW_DATA_CSV = os.path.join(DATA_DIR, 'LeBeouf_replication_data.csv')
ITEMS_CSV = os.path.join(DATA_DIR, 'ItemsList_Final.csv')

# Read files
print('Reading data from', RAW_DATA_CSV)
df = pd.read_csv(RAW_DATA_CSV, dtype=str)
print('Reading items from', ITEMS_CSV)
items = pd.read_csv(ITEMS_CSV)

# Convert some columns to numeric where needed
# Ensure columns exist
for col in ['Status','Attention1','Attention2','screenQ','Cond']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter as in the R script
# Exclude Status != 0 (preview), screenQ NA, Attention1 == 7, Attention2 == 1, screenQ == 2
df_clean = df.copy()
df_clean = df_clean[df_clean['Status'] == 0]
df_clean = df_clean[~df_clean['screenQ'].isna()]
df_clean = df_clean[df_clean['Attention1'] == 7]
df_clean = df_clean[df_clean['Attention2'] == 1]
df_clean = df_clean[df_clean['screenQ'] == 2]

# Drop metadata columns roughly equivalent to dplyr::select(dat, -c(StartDate:Q1))
meta_cols = ['StartDate','EndDate','Status','Progress','Duration (in seconds)',
             'Finished','RecordedDate','DistributionChannel','UserLanguage','Q1']
for c in meta_cols:
    if c in df_clean.columns:
        df_clean = df_clean.drop(columns=[c])

# Drop Familiarity_* and Bipol columns
familiarity_cols = [c for c in df_clean.columns if c.startswith('Familiarity')]
if familiarity_cols:
    df_clean = df_clean.drop(columns=familiarity_cols)
bipol_cols = [c for c in df_clean.columns if 'Bipol' in c or 'Bipolar' in c]
if bipol_cols:
    df_clean = df_clean.drop(columns=bipol_cols)

# Reset index and create ID
df_clean = df_clean.reset_index(drop=True)
df_clean.insert(0, 'ID', df_clean.index + 1)
df_clean['ID'] = df_clean['ID'].astype(str)

# Identify util and symbol columns (in order)
util_cols = [c for c in df_clean.columns if 'Util' in c]
symbol_cols = [c for c in df_clean.columns if 'Symbol' in c or 'Symbo' in c]

# Ensure ordering as in the file by column position
cols = list(df_clean.columns)
util_cols = [c for c in cols if c in util_cols]
symbol_cols = [c for c in cols if c in symbol_cols]

# Expect these to be 24 each (3 blocks x 8 items). Group into blocks of 8
block_size = 8
if len(util_cols) % block_size != 0 or len(symbol_cols) % block_size != 0:
    print('Warning: unexpected number of util/symbol columns', len(util_cols), len(symbol_cols))

n_blocks = max(1, len(util_cols) // block_size)

# Build long dataframe by participant selecting the block based on Cond
rows = []
for _, r in df_clean.iterrows():
    cond = int(r['Cond']) if not pd.isna(r['Cond']) else np.nan
    # block mapping: Cond 1 or 2 -> block 0; Cond 3 -> block 1; Cond 4 -> block 2
    if cond in [1,2]:
        bidx = 0
    elif cond == 3:
        bidx = 1
    elif cond == 4:
        bidx = 2
    else:
        # default to first block
        bidx = 0
    for item in range(1, block_size+1):
        util_col_idx = bidx*block_size + (item-1)
        sym_col_idx = bidx*block_size + (item-1)
        if util_col_idx < len(util_cols) and sym_col_idx < len(symbol_cols):
            util_col = util_cols[util_col_idx]
            sym_col = symbol_cols[sym_col_idx]
            # parse numeric, coerce non-numeric to NaN
            try:
                benefits = float(r[util_col])
            except Exception:
                benefits = np.nan
            try:
                symbols = float(r[sym_col])
            except Exception:
                symbols = np.nan
            rows.append({
                'ID': r['ID'],
                'Cond': cond,
                'ItemCatID': item,
                'Benefits': benefits,
                'Symbols': symbols
            })

long_df = pd.DataFrame(rows)
# Map Condition label
long_df['Condition'] = long_df['Cond'].apply(lambda x: 'Category' if x in [1,2] else ('Brand' if x in [3,4] else np.nan))

# Merge with items by ItemCatID and Cond to get ProductType and Category
# Items file has Cond values matching 1,2,3,4 per block
items_subset = items.copy()
# Ensure ItemCatID type
items_subset['ItemCatID'] = pd.to_numeric(items_subset['ItemCatID'], errors='coerce')
long_df['ItemCatID'] = pd.to_numeric(long_df['ItemCatID'], errors='coerce')
long_df = pd.merge(long_df, items_subset, on=['ItemCatID','Cond'], how='left')

# Compute ScoreDiff
long_df['ScoreDiff'] = long_df['Benefits'] - long_df['Symbols']

# Collapse by ID, ProductType, Condition computing mean ScoreDiff as in R
dat_collapsed = long_df.groupby(['ID','ProductType','Condition'], dropna=False)['ScoreDiff'].mean().reset_index()

# Save collapsed data
out_collapsed = os.path.join(DATA_DIR, 'dat_collapsed.csv')
dat_collapsed.to_csv(out_collapsed, index=False)
print('Saved collapsed data to', out_collapsed)

# Fit mixed effects model equivalent to aov(ScoreDiff ~ Condition*ProductType + Error(ID/ProductType))
# Use mixed linear model with random intercept for ID; interaction ProductType*Condition is fixed
# Encode categorical variables
dat_collapsed['ProductType'] = dat_collapsed['ProductType'].astype('category')
dat_collapsed['Condition'] = dat_collapsed['Condition'].astype('category')

# Drop rows with NaN ScoreDiff
model_df = dat_collapsed.dropna(subset=['ScoreDiff'])

# Fit mixed linear model
# Use patsy formula with categorical factors
model = smf.mixedlm('ScoreDiff ~ ProductType * Condition', data=model_df, groups=model_df['ID'])
model_fit = model.fit(reml=False)
print(model_fit.summary())

# Save model summary
out_model = os.path.join(DATA_DIR, 'model_summary.txt')
with open(out_model, 'w') as f:
    f.write(model_fit.summary().as_text())
print('Saved model summary to', out_model)

# Compute descriptive means table
means = model_df.groupby(['Condition','ProductType'])['ScoreDiff'].mean().unstack()
out_means = os.path.join(DATA_DIR, 'means_by_condition_producttype.csv')
means.to_csv(out_means)
print('Saved means table to', out_means)

print('Replication script finished.')
