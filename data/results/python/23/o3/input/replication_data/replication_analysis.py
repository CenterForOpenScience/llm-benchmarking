"""
Python replication script for LeBeouf et al. (2010) focal Level x Product Type interaction
=========================================================================================
This script reproduces the 2 (Level: NoBrand vs Brand) x 2 (Product Type: Utilitarian vs Symbolic)
mixed ANOVA on benefit-symbolism difference scores using the replication dataset.

Input files (assumed to be located inside /app/data):
    - LeBeouf_replication_data.csv : Raw Qualtrics data (responses)
    - ItemsList_Final.csv          : Metadata linking ItemCatID to Category & ProductType

Output:
    - prints ANOVA table and descriptive statistics to stdout
    - saves tidy long-format and collapsed datasets for inspection as CSV in the same folder
      (tidy_long.csv, collapsed.csv)

Dependencies: pandas, numpy, pingouin, statsmodels, scipy

The script follows as closely as possible the logic implemented in
`LeBoeuf_JournMarketRes_2010_EKBZ_Yoon_Direct Replication_618K_Final.R`.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pingouin as pg

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def wide_to_long(df: pd.DataFrame, cond_name: str) -> pd.DataFrame:
    """Reshape wide util / symbol columns into long tidy format.

    Parameters
    ----------
    df : pd.DataFrame      Filtered dataframe for a single condition.
    cond_name : str        Either 'Category' (NoBrand) or 'Brand'.

    Returns
    -------
    pd.DataFrame  long format with columns [ID, Cond, ItemCatID, Benefits, Symbols, Condition]
    """
    # Identify the paired util / symbol columns. They are of the pattern 'X_Util...' and 'X_Symbol...'.
    util_cols = [c for c in df.columns if '_Util' in c]
    sym_cols = [c for c in df.columns if '_Symbol' in c]

    util_cols_sorted = sorted(util_cols, key=lambda x: int(x.split('_')[0]))
    sym_cols_sorted = sorted(sym_cols, key=lambda x: int(x.split('_')[0]))

    # Ensure we have equal number of util and symbol columns
    assert len(util_cols_sorted) == len(sym_cols_sorted), "Mismatch util/symbol columns"

    varying = list(zip(util_cols_sorted, sym_cols_sorted))
    long_records = []
    for idx, row in df.iterrows():
        for item_id, (u_col, s_col) in enumerate(varying, start=1):
            long_records.append({
                'ID': row['ID'],
                'Cond': row['Cond'],
                'ItemCatID': item_id,
                'Benefits': row[u_col],
                'Symbols': row[s_col],
                'Condition': cond_name
            })
    long_df = pd.DataFrame(long_records)
    return long_df


# -----------------------------------------------------------------------------
# Paths & data loading# -----------------------------------------------------------------------------
# Paths & data loading
# -----------------------------------------------------------------------------
# Determine DATA_DIR dynamically: assume that replication_data directory contains this script
SCRIPT_DIR = Path(__file__).resolve().parent
# If CSV files are in the same replication_data directory, use that
DATA_DIR = SCRIPT_DIR  # points to .../replication_data
raw_file = DATA_DIR / 'LeBeouf_replication_data.csv'
item_file = DATA_DIR / 'ItemsList_Final.csv'

print(f"Reading main data from {raw_file}")
if not raw_file.exists():
    # Fallback: look for data under /app/data hierarchy in case of mounting differences
    alt_raw = Path('/app/data') / 'original/23/input/replication_data/LeBeouf_replication_data.csv'
    alt_item = Path('/app/data') / 'original/23/input/replication_data/ItemsList_Final.csv'
    if alt_raw.exists():
        raw_file = alt_raw
        item_file = alt_item
        print(f"Primary path not found. Using fallback path {raw_file}")

raw = pd.read_csv(raw_file)
print(f"Original shape: {raw.shape}")

print("Reading item metadata …")
items = pd.read_csv(item_file)

# -----------------------------------------------------------------------------
# Pre-processing / cleaning following the R script logic
# -----------------------------------------------------------------------------# -----------------------------------------------------------------------------
# Pre-processing / cleaning following the R script logic
# -----------------------------------------------------------------------------

# Exclude non-main / preview sessions (Status != 0 already filtered?)
clean = raw.copy()
clean = clean.loc[clean['Status'] == 0]

# Drop incomplete respondents (ScreenQ NA) & failed attention checks
clean = clean.loc[~clean['screenQ'].isna()]
clean = clean.loc[clean['Attention1'] == 7]
clean = clean.loc[clean['Attention2'] == 1]
clean = clean.loc[clean['screenQ'] == 2]

print(f"After exclusions: {clean.shape}")

# Remove administrative columns up to column 'Q1'
cols_to_drop = []
for col in clean.columns:
    if col.startswith('StartDate') or col == 'Q1' or col in ['EndDate', 'Progress', 'Duration (in seconds)',
                                                             'Finished', 'RecordedDate', 'DistributionChannel',
                                                             'UserLanguage']:
        cols_to_drop.append(col)
clean = clean.drop(columns=cols_to_drop, errors='ignore')

# Drop familiarity & bipolar ratings
clean = clean[[c for c in clean.columns if not c.startswith('Familiarity')]]
clean = clean[[c for c in clean.columns if 'Bipol' not in c]]

# Assign ID as factor
clean = clean.reset_index(drop=True)
clean.insert(0, 'ID', clean.index + 1)

# -----------------------------------------------------------------------------
# Split by condition
# -----------------------------------------------------------------------------
control = clean.loc[clean['Cond'].isin([1, 2])].copy()
brandA  = clean.loc[clean['Cond'] == 3].copy()
brandB  = clean.loc[clean['Cond'] == 4].copy()

# For each, drop the duplicate util/symbol pairs that belong to other brand list set
control = control[[c for c in control.columns if ('Util_1' not in c and 'Util_2' not in c and
                                                 'Symbol_1' not in c and 'Symbol_2' not in c)]]
brandA  = brandA[[c for c in brandA.columns if not (c.endswith('Util') or c.endswith('Symbol')) or '_2' not in c]]
brandB  = brandB[[c for c in brandB.columns if not (c.endswith('Util') or c.endswith('Symbol')) or '_1' not in c]]

# -----------------------------------------------------------------------------
# Reshape to long
# -----------------------------------------------------------------------------
control_long = wide_to_long(control, 'Category')
brandA_long  = wide_to_long(brandA, 'Brand')
brandB_long  = wide_to_long(brandB, 'Brand')

long_df = pd.concat([control_long, brandA_long, brandB_long], ignore_index=True)

# Merge with items metadata
long_df = long_df.merge(items, on=['ItemCatID', 'Cond'], how='left')

# Create ScoreDiff
long_df['ScoreDiff'] = long_df['Benefits'] - long_df['Symbols']

# Save tidy
long_df.to_csv(DATA_DIR / 'tidy_long.csv', index=False)
print("Saved tidy_long.csv")

# Collapse by participant, product type, and condition
collapsed = long_df.groupby(['ID', 'ProductType', 'Condition'], as_index=False)['ScoreDiff'].mean()
collapsed.to_csv(DATA_DIR / 'collapsed.csv', index=False)
print("Saved collapsed.csv")

# -----------------------------------------------------------------------------
# Mixed ANOVA using pingouin (within: ProductType, between: Condition)
# -----------------------------------------------------------------------------

anova_res = pg.mixed_anova(data=collapsed, dv='ScoreDiff', within='ProductType', between='Condition', subject='ID')
print("\nMixed ANOVA results (Pingouin):")
print(anova_res.round(4))

# Print counts per between-subject Condition for sanity check
print("\nCondition counts (collapsed):")
print(collapsed['Condition'].value_counts())

# Descriptive stats for report
print("\nDescriptive means:")
print(collapsed.groupby(['Condition', 'ProductType'])['ScoreDiff'].mean())

# Optional: post-hoc pairwise comparisons# Optional: post-hoc pairwise comparisons (disabled due to computation errors in pingouin 0.5.3)
try:
    posthoc = pg.pairwise_tests(data=collapsed, dv='ScoreDiff', within='ProductType', between='Condition',
                               subject='ID', padjust='bonf', effsize='hedges')
    print("\nPost-hoc pairwise tests:")
    print(posthoc.round(4))
except Exception as e:
    print("\nPost-hoc pairwise tests could not be computed:", e)
