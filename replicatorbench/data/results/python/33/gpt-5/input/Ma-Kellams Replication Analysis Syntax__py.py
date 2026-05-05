import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols; from patsy.contrasts import Sum

# Paths
DATA_PATH = "/app/data/Ma-Kellams Replication Study Data.sav"
OUT_DIR = "/app/data"

os.makedirs(OUT_DIR, exist_ok=True)

summary_lines = []

def log(line):
    print(line)
    summary_lines.append(line)

# Safe import of pyreadstat
try:
    import pyreadstat
except Exception as e:
    raise RuntimeError("pyreadstat is required to read SPSS .sav files. Please ensure it is installed.")

# Load dataset
if not os.path.exists(DATA_PATH):
    log(f"WARNING: Data file not found at {DATA_PATH}. Ensure the .sav file is mounted to /app/data.")

df, meta = pyreadstat.read_sav(DATA_PATH)
log(f"Loaded dataset with shape: {df.shape}")

# Helper functions

def recode_reverse(series):
    # SPSS recode: -3->3, -2->2, -1->1, 0->0, 1->-1, 2->-2, 3->-3
    return -1 * series

# Derived variables mirroring SPSS syntax
if 'WordCount' in df.columns:
    df['Wrote30WordsOrMore'] = np.where(df['WordCount'] > 29, 1, 0)
else:
    df['Wrote30WordsOrMore'] = np.nan
    log("NOTE: WordCount not found; Wrote30WordsOrMore set to NaN.")

# Ethnicity recodes
if 'CulturalBackground' in df.columns:
    df['EuropeanAmerican'] = (df['CulturalBackground'] == 1).astype(int)
    df['EastAsian'] = (df['CulturalBackground'] == 4).astype(int)
else:
    df['EuropeanAmerican'] = np.nan
    df['EastAsian'] = np.nan
    log("WARNING: CulturalBackground not found; ethnicity dummies set to NaN.")

# Culture: 0 = European American, 1 = East Asian, else NaN
culture = pd.Series(np.nan, index=df.index)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    culture = np.where(df.get('EuropeanAmerican', 0) == 1, 0, culture)
    culture = np.where(df.get('EastAsian', 0) == 1, 1, culture)

df['Culture'] = culture

# Born variables (not essential for focal but included for parity)
if 'BornCountry' in df.columns:
    df['USBorn'] = np.where(df['BornCountry'].isin([0, 187]), 1, 0)
    east_asian_country_codes = {162, 156, 140, 86, 75, 36, 1358}
    df['EastAsianBorn'] = np.where(df['BornCountry'].isin(east_asian_country_codes), 1, 0)
else:
    df['USBorn'] = np.nan
    df['EastAsianBorn'] = np.nan

# PAS reverse-scoring and average
for col in ['PAS2', 'PAS5']:
    if col in df.columns:
        df[col + 'Positive'] = recode_reverse(df[col])
    else:
        df[col + 'Positive'] = np.nan
        log(f"WARNING: {col} not found; {col}Positive set to NaN.")

pas_components = []
for col in ['PAS1', 'PAS2Positive', 'PAS3', 'PAS4', 'PAS5Positive']:
    if col in df.columns:
        pas_components.append(col)
    else:
        log(f"WARNING: {col} not found; PASAverage may be incomplete.")

if pas_components:
    df['PASAverage'] = df[pas_components].mean(axis=1)
else:
    df['PASAverage'] = np.nan
    log("ERROR: No PAS items found; cannot compute PASAverage.")

# PANAS Net Score
pos_items = ['PANAS1','PANAS3','PANAS5','PANAS9','PANAS10','PANAS12','PANAS14','PANAS16','PANAS17','PANAS19']
neg_items = ['PANAS2','PANAS4','PANAS6','PANAS7','PANAS8','PANAS11','PANAS13','PANAS15','PANAS18','PANAS20']

have_pos = all([(c in df.columns) for c in pos_items])
have_neg = all([(c in df.columns) for c in neg_items])
if have_pos and have_neg:
    df['PANASNetScore'] = df[pos_items].sum(axis=1) - df[neg_items].sum(axis=1)
else:
    df['PANASNetScore'] = np.nan
    log("NOTE: PANAS items incomplete; PANASNetScore set to NaN.")

# Ensure WritingCondition exists
if 'WritingCondition' not in df.columns:
    log("ERROR: WritingCondition not found. Focal 2x2 ANOVA cannot be computed.")

# Helper to run ANOVA Type III and save table

def run_anova_typ3(formula, data, label, file_stub):
    d = data.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.shape[0] < 4:
        log(f"WARNING: Not enough non-missing rows for {label} after dropna.")
        return None
    try:
        model = ols(formula, data=d).fit()
        anova_t = sm.stats.anova_lm(model, typ=3)
        out_path = os.path.join(OUT_DIR, f"{file_stub}.csv")
        anova_t.to_csv(out_path)
        log(f"ANOVA ({label}) saved to {out_path}")
        # Also log key rows
        log(f"ANOVA summary for {label} (Type III):\n{anova_t}")
        return anova_t
    except Exception as e:
        log(f"ERROR running ANOVA for {label}: {e}")
        return None

# Helper to compute cell means with SE

def cell_means(df_in, dv, factors, file_stub):
    d = df_in.copy()
    d = d[factors + [dv]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        log(f"WARNING: No data for cell means {dv} by {factors}")
        return None
    grp = d.groupby(factors)[dv]
    res = pd.DataFrame({
        'mean': grp.mean(),
        'count': grp.count(),
        'se_mean': grp.std(ddof=1) / np.sqrt(grp.count())
    }).reset_index()
    out_path = os.path.join(OUT_DIR, f"{file_stub}.csv")
    res.to_csv(out_path, index=False)
    log(f"Cell means for {dv} by {factors} saved to {out_path}")
    return res

# Main replication test: PASAverage ~ Culture * WritingCondition# Main replication test: PASAverage ~ Culture * WritingCondition
if {'PASAverage','Culture','WritingCondition'}.issubset(df.columns):
    anova_main = run_anova_typ3('PASAverage ~ C(Culture, Sum) * C(WritingCondition, Sum)', df[['PASAverage','Culture','WritingCondition']].copy(),
                                'PASAverage ~ Culture * WritingCondition', 'anova_pas_culture_x_condition')
    cell_means(df, 'PASAverage', ['Culture', 'WritingCondition'], 'cell_means_pas')
else:
    log("ERROR: Required columns for main test missing.")

# Mood effects
# Mood effects
if {'PASAverage','PANASNetScore'}.issubset(df.columns):
    run_anova_typ3('PASAverage ~ PANASNetScore', df[['PASAverage','PANASNetScore']].copy(), 'PASAverage ~ PANASNetScore', 'anova_pas_mood')
if {'PASAverage','PANASNetScore','WritingCondition'}.issubset(df.columns):
    run_anova_typ3('PASAverage ~ PANASNetScore * C(WritingCondition, Sum)', df[['PASAverage','PANASNetScore','WritingCondition']].copy(),
                   'PASAverage ~ PANASNetScore * WritingCondition', 'anova_pas_mood_x_condition')

# Bail analyses
if 'BailAmount' in df.columns:
    if {'Culture','WritingCondition'}.issubset(df.columns):
        run_anova_typ3('BailAmount ~ C(Culture, Sum) * C(WritingCondition, Sum)', df[['BailAmount','Culture','WritingCondition']].copy(),
                       'BailAmount ~ Culture * WritingCondition', 'anova_bail_culture_x_condition')
        cell_means(df, 'BailAmount', ['Culture','WritingCondition'], 'cell_means_bail')
    # Outlier handling and transforms
    if df['BailAmount'].notna().any():
        m = df['BailAmount'].mean(skipna=True)
        s = df['BailAmount'].std(skipna=True)
        thresh = m + 2*s
        df_bail_in = df[df['BailAmount'] < thresh].copy()
        if {'Culture','WritingCondition'}.issubset(df_bail_in.columns):
            run_anova_typ3('BailAmount ~ C(Culture, Sum) * C(WritingCondition, Sum)', df_bail_in[['BailAmount','Culture','WritingCondition']].copy(),
                           'BailAmount (outliers removed) ~ Culture * WritingCondition', 'anova_bail_outliers_removed')
        # Transforms
        df['BailAmountSqrt'] = np.sqrt(df['BailAmount'].clip(lower=0))
        df['BailAmountLog'] = np.log(df['BailAmount'].clip(lower=0) + 1)
        if {'Culture','WritingCondition'}.issubset(df.columns):
            run_anova_typ3('BailAmountLog ~ C(Culture, Sum) * C(WritingCondition, Sum)', df[['BailAmountLog','Culture','WritingCondition']].copy(),
                           'BailAmountLog ~ Culture * WritingCondition', 'anova_bail_log')

# Simple effects: within groups and within conditions
# EA only
if {'Culture','WritingCondition','PASAverage'}.issubset(df.columns):
    d_ea = df[df['Culture'] == 0][['PASAverage','WritingCondition']].dropna()
    if not d_ea.empty:
        run_anova_typ3('PASAverage ~ C(WritingCondition, Sum)', d_ea, 'EA only: PASAverage ~ WritingCondition', 'anova_pas_ea_only')

# East Asian only (per SPSS: EastAsian == 1)
if {'EastAsian','WritingCondition','PASAverage'}.issubset(df.columns):
    d_as = df[df['EastAsian'] == 1][['PASAverage','WritingCondition']].dropna()
    if not d_as.empty:
        run_anova_typ3('PASAverage ~ C(WritingCondition, Sum)', d_as, 'East Asian only: PASAverage ~ WritingCondition', 'anova_pas_asia_only')

# Cultural differences by condition
if {'PASAverage','Culture','WritingCondition'}.issubset(df.columns):
    for cond_val, stub in [(0,'control_only'), (1,'mortality_only')]:
        d_cond = df[df['WritingCondition'] == cond_val][['PASAverage','Culture']].dropna()
        if not d_cond.empty:
            run_anova_typ3('PASAverage ~ C(Culture, Sum)', d_cond, f'PASAverage ~ Culture ({stub})', f'anova_pas_culture_{stub}')

# Robustness: exclude short responses
if {'PASAverage','Culture','WritingCondition','Wrote30WordsOrMore'}.issubset(df.columns):
    d_long = df[df['Wrote30WordsOrMore'] == 1][['PASAverage','Culture','WritingCondition']].dropna()
    if not d_long.empty:
        run_anova_typ3('PASAverage ~ C(Culture) * C(WritingCondition)', d_long, 'PAS (>=30 words) ~ Culture * Condition', 'anova_pas_30words')

# Exclude PAS negative outliers (mean - 2*SD)
if 'PASAverage' in df.columns:
    pas = df['PASAverage']
    if pas.notna().any():
        mu, sd = pas.mean(skipna=True), pas.std(ddof=1, skipna=True)
        cutoff = mu - 2*sd
        d_pas_no_out = df[df['PASAverage'] > cutoff][['PASAverage','Culture','WritingCondition']].dropna()
        if not d_pas_no_out.empty and {'Culture','WritingCondition'}.issubset(d_pas_no_out.columns):
            run_anova_typ3('PASAverage ~ C(Culture) * C(WritingCondition)', d_pas_no_out, 'PAS (no low outliers) ~ Culture * Condition', 'anova_pas_no_low_outliers')

# Prolific subsets if available
if 'Prolific' in df.columns and {'PASAverage','Culture','WritingCondition'}.issubset(df.columns):
    for val, stub in [(0,'non_prolific'), (1,'prolific')]:
        d_sub = df[df['Prolific'] == val][['PASAverage','Culture','WritingCondition']].dropna()
        if not d_sub.empty:
            run_anova_typ3('PASAverage ~ C(Culture) * C(WritingCondition)', d_sub, f'PAS ~ Culture * Condition ({stub})', f'anova_pas_{stub}')
            cell_means(d_sub, 'PASAverage', ['Culture','WritingCondition'], f'cell_means_pas_{stub}')

# Religion subset
if 'Religion' in df.columns and {'PASAverage','Culture','WritingCondition'}.issubset(df.columns):
    religious = np.where(df['Religion'] > 2, 1, np.where(df['Religion'] < 3, 0, np.nan))
    df['Religious'] = religious
    d_nonrel = df[df['Religious'] == 0][['PASAverage','Culture','WritingCondition']].dropna()
    if not d_nonrel.empty:
        run_anova_typ3('PASAverage ~ C(Culture) * C(WritingCondition)', d_nonrel, 'PAS ~ Culture * Condition (non-religious)', 'anova_pas_nonreligious')
        cell_means(d_nonrel, 'PASAverage', ['Culture','WritingCondition'], 'cell_means_pas_nonreligious')

# Save summary log
summary_path = os.path.join(OUT_DIR, 'replication_results_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_lines))
log(f"Summary written to {summary_path}")
