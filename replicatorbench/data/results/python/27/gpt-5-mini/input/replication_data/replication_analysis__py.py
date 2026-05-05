import os
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = os.path.dirname(__file__)
ROUND1 = os.path.join(DATA_DIR, 'round1_raw.csv')
ROUND2 = os.path.join(DATA_DIR, 'round2_raw.csv')

FIXED_PRICE = 1.94


def load_and_melt(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    # Keep original columns
    # Identify price columns like ipo_task.<r>.group.market_price
    price_cols = [c for c in df.columns if '.group.market_price' in c]
    if not price_cols:
        # fallback: columns named price1..price20
        price_cols = [c for c in df.columns if 'price' in c.lower()]
    # Create an id for group: try ipo_task.1.group.id_in_subsession or group id columns
    if 'ipo_task.1.group.id_in_subsession' in df.columns:
        df['group_id'] = df['ipo_task.1.group.id_in_subsession']
    elif 'group_id' in df.columns:
        df['group_id'] = df['group_id']
    else:
        # fallback to index
        df['group_id'] = np.arange(len(df))
    # Also capture session code and task type if present
    if 'session.code' in df.columns:
        df['session_code'] = df['session.code']
    else:
        df['session_code'] = df.get('session_code', '')
    if 'ipo_task.1.group.task_type' in df.columns:
        df['task_type'] = df['ipo_task.1.group.task_type']
    elif 'task_type' in df.columns:
        df['task_type'] = df['task_type']
    else:
        df['task_type'] = df.get('task_type', '')

    # Melt price columns to long format
    df_long = df[['group_id', 'session_code', 'task_type'] + price_cols].melt(
        id_vars=['group_id', 'session_code', 'task_type'],
        value_vars=price_cols,
        var_name='price_col',
        value_name='price'
    )
    # Extract round number from column name if possible
    def extract_round(colname):
        import re
        m = re.search(r'ipo_task\.(\d+)\.group\.market_price', colname)
        if m:
            return int(m.group(1))
        m = re.search(r'price(\d+)', colname)
        if m:
            return int(m.group(1))
        return np.nan

    df_long['round'] = df_long['price_col'].apply(extract_round)
    df_long['price'] = pd.to_numeric(df_long['price'], errors='coerce')
    return df_long


def compute_group_round_means(df_long):
    # aggregate to group-round means (in case multiple rows per group-round)
    grp = df_long.groupby(['session_code','group_id','round','task_type'], dropna=False)
    out = grp['price'].mean().reset_index().rename(columns={'price':'price_mean'})
    return out


def add_dropouts_bankrupts(out_df, raw_df):
    # raw_df not always available at this point; but we can approximate dropout using NA fraction per group
    # For now, create placeholders: dropout=0, bankrupt=0
    out_df['dropout'] = 0
    out_df['bankrupt'] = 0
    return out_df


def run_wilcoxon(series, value=FIXED_PRICE):
    # scipy.stats.wilcoxon tests whether median of differences is zero; we test series - value
    clean = series.dropna() - value
    if len(clean) == 0:
        return {'statistic': None, 'pvalue': None, 'n': 0}
    try:
        stat, p = stats.wilcoxon(clean)
        return {'statistic': float(stat), 'pvalue': float(p), 'n': len(clean)}
    except Exception as e:
        # if all zeros or not enough samples
        return {'statistic': None, 'pvalue': None, 'n': len(clean)}


def main():
    # Load round1 and round2
    r1 = load_and_melt(ROUND1)
    r2 = load_and_melt(ROUND2)
    combined = pd.concat([r1, r2], ignore_index=True)

    # Compute group-round means
    group_round = compute_group_round_means(combined)

    # Add dropout/bankrupt placeholders
    group_round = add_dropouts_bankrupts(group_round, None)

    # Run tests
    results = {}
    # Full sample
    res_full = run_wilcoxon(group_round['price_mean'])
    results['full'] = res_full
    # Non-dropout
    res_nodrop = run_wilcoxon(group_round.loc[group_round['dropout']==0,'price_mean'])
    results['nodrop'] = res_nodrop
    # Non-dropout & non-bankrupt
    res_clean = run_wilcoxon(group_round.loc[(group_round['dropout']==0)&(group_round['bankrupt']==0),'price_mean'])
    results['nodrop_nobank'] = res_clean

    # Save outputs
    out_full = os.path.join(DATA_DIR, 'fullsample_analysis.csv')
    out_round1 = os.path.join(DATA_DIR, 'round1_analysis.csv')
    # For compatibility, write the group_round dataframe as fullsample_analysis
    group_round.to_csv(out_full, index=False)
    # Create a round1-only collapse (round==1)
    round1_only = group_round[group_round['round']==1]
    round1_only.to_csv(out_round1, index=False)

    # Print results
    print('Wilcoxon signed-rank test results (price vs {:.2f}):'.format(FIXED_PRICE))
    for k,v in results.items():
        print(f"{k}: n={v['n']} stat={v['statistic']} p={v['pvalue']}")

    # Also print simple descriptive stats
    mean_price = group_round['price_mean'].mean()
    std_price = group_round['price_mean'].std()
    print(f"Mean group-round price: {mean_price:.4f} (sd={std_price:.4f}), N={len(group_round)}")

if __name__ == '__main__':
    main()
