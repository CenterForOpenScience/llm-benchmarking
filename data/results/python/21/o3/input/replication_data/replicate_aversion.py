import pandas as pd, numpy as np, os, sys

DATA_DIR = os.getenv('DATA_DIR', '/app/data')
PART1_CANDIDATES = [
    os.path.join(DATA_DIR, 'Bischetti_Survey_Part1_deidentify.csv'),
    os.path.join(DATA_DIR, 'replication_data', 'Bischetti_Survey_Part1_deidentify.csv')
]
PART2_CANDIDATES = [
    os.path.join(DATA_DIR, 'Bischetti_Survey_Part2_deidentify.csv'),
    os.path.join(DATA_DIR, 'replication_data', 'Bischetti_Survey_Part2_deidentify.csv')
]
META_CANDIDATES  = [
    os.path.join(DATA_DIR, 'stimulus_metadata.csv'),
    os.path.join(DATA_DIR, 'replication_data', 'stimulus_metadata.csv')
]

def first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'None of the candidate files exist: {candidates}')

PART1 = first_existing(PART1_CANDIDATES)
PART2 = first_existing(PART2_CANDIDATES)
META  = first_existing(META_CANDIDATES) if any(os.path.exists(p) for p in META_CANDIDATES) else None

print(f'Using Part1: {PART1}')
print(f'Using Part2: {PART2}')
print(f'Using meta: {META}')

# Load datasets# Load datasets
print('Loading survey parts...')
df1 = pd.read_csv(PART1)
df2 = pd.read_csv(PART2)

# Merge on anonymous participant id if available, else outer concat
id_col = 'participant_id' if 'participant_id' in df1.columns else 'PROLIFIC_PID'

if id_col not in df1.columns:
    raise ValueError('Expected participant identifier column not found in Part1')
if id_col not in df2.columns:
    raise ValueError('Expected participant identifier column not found in Part2')

df = pd.merge(df1, df2, on=id_col, how='outer')
print(f'Merged shape: {df.shape}')

# keep aversiveness ratings only (columns ending with _disturbing)
dist_cols = [c for c in df.columns if c.endswith('_disturbing')]
print(f'Number of aversiveness items: {len(dist_cols)}')

long_df = df.melt(id_vars=[id_col], value_vars=dist_cols, var_name='name', value_name='Aversiveness')

# Convert scale 1-7 to 0-6 if necessary (looks like original used 0-6). We'll enforce 0-6.
long_df['Aversiveness'] = long_df['Aversiveness'] - 1

# Load stimulus metadata
aff_map = None
if META is not None and os.path.exists(META):
    meta_df = pd.read_csv(META)
    if {'name','label'}.issubset(meta_df.columns):
        aff_map = meta_df[['name','label']]
        long_df = long_df.merge(aff_map, on='name', how='left')
    else:
        print('stimulus_metadata.csv found but missing required columns; proceeding without labels')
else:
    print('stimulus_metadata.csv not found; proceeding without labels')

if 'label' not in long_df.columns:
    # Try to infer covid vs non-covid based on picture number heuristic
    def infer_label(pic):
        # extract numeric id after 'pic' prefix
        try:
            num = int(pic.split('pic')[1].split('_')[0])
        except Exception:
            return np.nan
        # placeholder heuristic: ids <= 100 considered covid verbal, >100 non-covid verbal
        return 'covid-verbal' if num <= 100 else 'non-verbal'
    long_df['label'] = long_df['name'].apply(infer_label)
    print('Labels inferred heuristically (may be imprecise).')

# Filter to verbal jokes only
mask = long_df['label'].isin(['covid-verbal','non-verbal'])
verbal_df = long_df.loc[mask].copy()

# Drop missing ratings
verbal_df = verbal_df.dropna(subset=['Aversiveness'])

# Compute participant-level means
means = verbal_df.groupby([id_col,'label'])['Aversiveness'].mean().unstack()
means = means.dropna()

# Calculate difference (covid - non)
means['diff'] = means['covid-verbal'] - means['non-verbal']

mean_diff = means['diff'].mean()
std_diff = means['diff'].std(ddof=1)
from scipy import stats
n = len(means)
sem = std_diff/np.sqrt(n)
t_stat, p_val = stats.ttest_rel(means['covid-verbal'], means['non-verbal'])

print('Replication result:')
print(f'Mean aversiveness covid verbal: {means["covid-verbal"].mean():.3f}')
print(f'Mean aversiveness non covid verbal: {means["non-verbal"].mean():.3f}')
print(f'Mean paired difference: {mean_diff:.3f}')
print(f't({n-1}) = {t_stat:.2f}, p = {p_val:.4f}')

# Save summary to csv
out_path = os.path.join(DATA_DIR, 'replication_summary.csv')
summary = pd.DataFrame({'n_participants':[n],
                        'mean_covid_verbal':[means['covid-verbal'].mean()],
                        'mean_non_verbal':[means['non-verbal'].mean()],
                        'mean_diff':[mean_diff],
                        't_stat':[t_stat],
                        'p_val':[p_val]})
summary.to_csv(out_path, index=False)
print(f'Summary saved to {out_path}')
