import os
import re
import json
import numpy as np
import pandas as pd
from scipy import stats

# Configuration
INPUT_PATH = os.getenv("REPLICATION_INPUT", "/app/data/Data_Cleaned_22102020.csv")
OUTPUT_DIR = "/app/data"
DATA_S_OUT = os.path.join(OUTPUT_DIR, "dataS.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "results_summary.txt")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "results_summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helpers
_non_numeric_re = re.compile(r"[^0-9.\-]")

def to_numeric_clean(series):
    if series is None:
        return None
    # Convert to string, strip currency and non-numeric, then to float
    return pd.to_numeric(series.astype(str).str.replace("\u00a0", "", regex=False)\
                          .str.replace(",", "", regex=False)\
                          .str.replace(_non_numeric_re, "", regex=True)\
                          .str.strip(), errors='coerce')

# Load data (robust to BOM and encoding quirks)
read_errors = []
df = None
for enc in ["utf-8-sig", "utf-8", "latin-1"]:
    try:
        df = pd.read_csv(INPUT_PATH, encoding=enc)
        break
    except Exception as e:
        read_errors.append(f"{enc}: {e}")

if df is None:
    raise RuntimeError(f"Failed to read input CSV from {INPUT_PATH}. Tried encodings: {read_errors}")

# Standardize expected columns
expected_cols = {
    'Lot_WTPc': None,
    'Gift_WTPc': None,
    'Lot_check': None,
    'Gift_check': None
}
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in dataset: {missing}. Found: {list(df.columns)}")

# Clean numeric columns
lot_wtp = to_numeric_clean(df['Lot_WTPc'])
gift_wtp = to_numeric_clean(df['Gift_WTPc'])
lot_chk = pd.to_numeric(df['Lot_check'], errors='coerce')
gift_chk = pd.to_numeric(df['Gift_check'], errors='coerce')

# Condition assignment: lottery if Lot_check present, gift if Gift_check present (use object dtype to avoid numpy dtype issues)
cond = pd.Series(np.nan, index=df.index, dtype=object)
cond[~lot_chk.isna()] = 'lottery'
cond[lot_chk.isna() & ~gift_chk.isna()] = 'gift'

# Assign WTP and check based on condition
wtp = pd.Series(np.nan, index=df.index, dtype=float)
wtp[cond == 'lottery'] = lot_wtp
wtp[cond == 'gift'] = gift_wtp

check = pd.Series(np.nan, index=df.index, dtype=float)
check[cond == 'lottery'] = lot_chk
check[cond == 'gift'] = gift_chk

work = pd.DataFrame({
    'Cond': cond,
    'WTP': wtp,
    'check': check
})

# Drop missing WTP or condition
work = work.dropna(subset=['Cond', 'WTP'])

# Filter to those who answered lowest-outcome correctly (code == 3)
dataS = work[work['check'] == 3].copy()

# Save filtered dataset
try:
    dataS.to_csv(DATA_S_OUT, index=False)
except Exception as e:
    # Fallback to a safe name if permission/path issues
    alt_path = os.path.join(OUTPUT_DIR, "dataS_filtered.csv")
    dataS.to_csv(alt_path, index=False)
    DATA_S_OUT = alt_path

# Group stats helper

def group_summary(df_in):
    out = {}
    for name, g in df_in.groupby('Cond'):
        vals = g['WTP'].dropna().astype(float)
        out[str(name)] = {
            'n': int(vals.shape[0]),
            'mean': float(vals.mean()) if vals.size else np.nan,
            'std': float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
            'median': float(vals.median()) if vals.size else np.nan,
            'min': float(vals.min()) if vals.size else np.nan,
            'max': float(vals.max()) if vals.size else np.nan
        }
    return out

# Cohen's d for independent groups

def cohens_d_independent(x, y):
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    d = (x.mean() - y.mean()) / sp
    # Hedges' g correction for small samples
    J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    return float(d * J)

# Perform tests on dataS (critical test)
summary = {
    'input_csv': INPUT_PATH,
    'dataS_csv': DATA_S_OUT,
    'analysis': []
}

if dataS.empty or dataS['Cond'].nunique() < 2:
    summary['analysis'].append({
        'subset': 'check_correct==3',
        'error': 'Insufficient data after filtering for comprehension check to run two-sample test.'
    })
else:
    grp = group_summary(dataS)
    lot = dataS.loc[dataS['Cond'] == 'lottery', 'WTP']
    gift = dataS.loc[dataS['Cond'] == 'gift', 'WTP']
    t_stat, p_val = stats.ttest_ind(lot, gift, equal_var=False, nan_policy='omit')
    d = cohens_d_independent(gift, lot)  # positive if gift > lottery
    summary['analysis'].append({
        'subset': 'check_correct==3',
        'group_summary': grp,
        'test': 'Welch t-test (independent samples)',
        't_stat': float(t_stat) if np.isfinite(t_stat) else None,
        'p_value': float(p_val) if np.isfinite(p_val) else None,
        'effect_size': {
            'cohens_d_hedges_g': d,
            'interpretation': 'Positive d indicates higher WTP in gift vs lottery.'
        }
    })

    # Subset WTP <= 20 (as in the R script sensitivity)
    dataSC = dataS[dataS['WTP'] <= 20].copy()
    if dataSC.empty or dataSC['Cond'].nunique() < 2:
        summary['analysis'].append({
            'subset': 'check_correct==3 & WTP<=20',
            'error': 'Insufficient data in subset to run two-sample test.'
        })
    else:
        grp_sc = group_summary(dataSC)
        lot_sc = dataSC.loc[dataSC['Cond'] == 'lottery', 'WTP']
        gift_sc = dataSC.loc[dataSC['Cond'] == 'gift', 'WTP']
        t_stat_sc, p_val_sc = stats.ttest_ind(lot_sc, gift_sc, equal_var=False, nan_policy='omit')
        d_sc = cohens_d_independent(gift_sc, lot_sc)
        summary['analysis'].append({
            'subset': 'check_correct==3 & WTP<=20',
            'group_summary': grp_sc,
            'test': 'Welch t-test (independent samples)',
            't_stat': float(t_stat_sc) if np.isfinite(t_stat_sc) else None,
            'p_value': float(p_val_sc) if np.isfinite(p_val_sc) else None,
            'effect_size': {
                'cohens_d_hedges_g': d_sc,
                'interpretation': 'Positive d indicates higher WTP in gift vs lottery.'
            }
        })

# Persist summaries
with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
    f.write("Replication of Yang et al. (2013) focal test: Lottery vs Gift frame on WTP\n")
    f.write(f"Input data: {INPUT_PATH}\n")
    f.write(f"Filtered data (check==3) saved to: {DATA_S_OUT}\n\n")
    for block in summary['analysis']:
        f.write(f"Subset: {block.get('subset')}\n")
        if 'error' in block:
            f.write(f"  ERROR: {block['error']}\n\n")
            continue
        f.write("  Group summaries (WTP):\n")
        gs = block.get('group_summary', {})
        for k, v in gs.items():
            f.write(f"    {k}: n={v['n']}, mean={v['mean']:.3f}, sd={v['std']}, median={v['median']}, min={v['min']}, max={v['max']}\n")
        f.write(f"  Test: {block.get('test')}\n")
        f.write(f"    t = {block.get('t_stat')}, p = {block.get('p_value')}\n")
        es = block.get('effect_size', {})
        f.write(f"  Effect size (Hedges g): {es.get('cohens_d_hedges_g')} ({es.get('interpretation')})\n\n")

print(f"Analysis complete. Summary written to {SUMMARY_TXT} and {SUMMARY_JSON}.")
