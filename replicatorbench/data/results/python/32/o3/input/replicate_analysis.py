import pandas as pd
import numpy as np
from scipy import stats
import os

# All IO paths must be inside /app/data according to policy
DATA_PATH = '/app/data/Data_Cleaned_22102020.csv'

# Because the original CSV has an extra leading empty column and possibly mixed encodings,
# we attempt to read it with latin-1 encoding and drop the unnamed first column.

def load_and_clean(path=DATA_PATH):
    # try different encodings
    for enc in ['utf-8', 'latin1', 'utf-16']:  # utf-16 unlikely but try
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError('Failed to read CSV with tried encodings')

    # Drop the unnamed index-like first column if present
    if df.columns[0].lower().startswith(('unnamed', 'ï..id', '')):
        df = df.drop(columns=[df.columns[0]])

    # Standardize column names by stripping whitespace
    df.columns = [c.strip() for c in df.columns]

    # Create independent variable Cond similar to R script
    # initialize all NaN
    df['Cond'] = np.nan
    # Assign 0 for lottery condition where Lot_check is non-NA / non-empty
    df.loc[df['Lot_check'].notna(), 'Cond'] = 0
    # Assign 1 for gift condition where Gift_check is non-NA / non-empty
    df.loc[df['Gift_check'].notna(), 'Cond'] = 1

    # Dependent variable WTP: choose Lot_WTPc or Gift_WTPc according to Cond
    df['WTP'] = np.where(df['Cond'] == 0, df['Lot_WTPc'],
                         np.where(df['Cond'] == 1, df['Gift_WTPc'], np.nan))

    # cast WTP to numeric, coerce errors to NaN (some cells have strings like "$20")
    df['WTP'] = pd.to_numeric(df['WTP'].replace({'\$':''}, regex=True), errors='coerce')

    # Checking variable
    df['check'] = np.where(df['Cond'] == 0, df['Lot_check'],
                           np.where(df['Cond'] == 1, df['Gift_check'], np.nan))

    # Convert check and Cond to numeric / category as appropriate
    df['check'] = pd.to_numeric(df['check'], errors='coerce')
    df['Cond'] = df['Cond'].map({0: 'lottery', 1: 'gift'})

    # Keep rows with finite WTP
    df = df[np.isfinite(df['WTP'])]
    return df


def analyse(df):
    # Select those who answered correctly (check == 3)
    dataS = df[df['check'] == 3].copy()

    # t-test between groups
    lottery_wtp = dataS.loc[dataS['Cond'] == 'lottery', 'WTP']
    gift_wtp = dataS.loc[dataS['Cond'] == 'gift', 'WTP']

    t_stat, p_val = stats.ttest_ind(lottery_wtp, gift_wtp, equal_var=False, nan_policy='omit')

    # Cohen's d
    n1, n2 = len(lottery_wtp), len(gift_wtp)
    s1, s2 = lottery_wtp.std(ddof=1), gift_wtp.std(ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohend = (lottery_wtp.mean() - gift_wtp.mean()) / s_pooled if s_pooled else np.nan

    print('Sample sizes (after comprehension check ==3):')
    print(f'Lottery: {n1}, Gift: {n2}')
    print('Means:')
    print(f'Lottery mean = {lottery_wtp.mean():.3f}, Gift mean = {gift_wtp.mean():.3f}')
    print(f'T-statistic = {t_stat:.3f}, p-value = {p_val:.4f}')
    print(f"Cohen's d = {cohend:.3f}")

    # Additional filter WTP<=20 as sensitivity analysis
    dataSC = dataS[dataS['WTP'] <= 20]
    lot_sc = dataSC.loc[dataSC['Cond'] == 'lottery', 'WTP']
    gift_sc = dataSC.loc[dataSC['Cond'] == 'gift', 'WTP']
    t_stat_sc, p_val_sc = stats.ttest_ind(lot_sc, gift_sc, equal_var=False, nan_policy='omit')
    n1_sc, n2_sc = len(lot_sc), len(gift_sc)
    s1_sc, s2_sc = lot_sc.std(ddof=1), gift_sc.std(ddof=1)
    s_pooled_sc = np.sqrt(((n1_sc - 1) * s1_sc**2 + (n2_sc - 1) * s2_sc**2) / (n1_sc + n2_sc - 2))
    cohend_sc = (lot_sc.mean() - gift_sc.mean()) / s_pooled_sc if s_pooled_sc else np.nan

    print('\nSensitivity analysis (WTP <= 20):')
    print(f'Lottery: {n1_sc}, Gift: {n2_sc}')
    print(f'Means: Lottery = {lot_sc.mean():.3f}, Gift = {gift_sc.mean():.3f}')
    print(f'T-statistic = {t_stat_sc:.3f}, p-value = {p_val_sc:.4f}')
    print(f"Cohen's d = {cohend_sc:.3f}")


if __name__ == '__main__':
    df_clean = load_and_clean()
    analyse(df_clean)
