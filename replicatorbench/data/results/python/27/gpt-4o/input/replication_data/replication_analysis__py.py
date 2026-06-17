import pandas as pd
from scipy.stats import wilcoxon

# Load and clean data for Round 1
round1_dat = pd.read_csv('/app/data/round1_raw.csv')
round1_dat = round1_dat[round1_dat['ipo_task.1.group.id_in_subsession'] != 1]
round1_dat = round1_dat.rename(columns={'ipo_task.1.group.id_in_subsession': 'group_in_session',
                                         'ipo_task.20.player.total_missing_responses': 'rounds_missed'})

# Reshape data to long format
round1_dat_long = pd.melt(round1_dat, id_vars=['session.code', 'group_in_session', 'participant.id_in_session', 'rounds_missed'],
                          value_vars=[col for col in round1_dat.columns if 'group.market_price' in col],
                          var_name='round', value_name='market_price')

# Calculate average market price by group
round1_dat_collapsed = round1_dat_long.groupby(['session.code', 'group_in_session']).agg(
    market_price=('market_price', 'mean'),
    rounds_missed_max=('rounds_missed', 'max'),
    rounds_missed_min=('rounds_missed', 'min')
).reset_index()

# Filter out bankrupt and dropout groups
round1_dat_collapsed['bankrupt'] = round1_dat_collapsed['rounds_missed_min'] == -99
round1_dat_collapsed['dropout'] = round1_dat_collapsed['rounds_missed_max'] >= 6
round1_dat_filtered = round1_dat_collapsed[(~round1_dat_collapsed['bankrupt']) & (~round1_dat_collapsed['dropout'])]

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(round1_dat_filtered['market_price'] - 1.94)
print(f'Round 1 Wilcoxon test statistic: {stat}, p-value: {p_value}')

# Repeat for Round 2
def process_round(file_path):
    dat = pd.read_csv(file_path)
    dat = dat[dat['ipo_task.1.group.id_in_subsession'] != 1]
    dat = dat.rename(columns={'ipo_task.1.group.id_in_subsession': 'group_in_session',
                              'ipo_task.20.player.total_missing_responses': 'rounds_missed'})
    dat_long = pd.melt(dat, id_vars=['session.code', 'group_in_session', 'participant.id_in_session', 'rounds_missed'],
                       value_vars=[col for col in dat.columns if 'group.market_price' in col],
                       var_name='round', value_name='market_price')
    dat_collapsed = dat_long.groupby(['session.code', 'group_in_session']).agg(
        market_price=('market_price', 'mean'),
        rounds_missed_max=('rounds_missed', 'max'),
        rounds_missed_min=('rounds_missed', 'min')
    ).reset_index()
    dat_collapsed['bankrupt'] = dat_collapsed['rounds_missed_min'] == -99
    dat_collapsed['dropout'] = dat_collapsed['rounds_missed_max'] >= 6
    dat_filtered = dat_collapsed[(~dat_collapsed['bankrupt']) & (~dat_collapsed['dropout'])]
    return dat_filtered

round2_dat_filtered = process_round('/app/data/round2_raw.csv')

# Perform Wilcoxon signed-rank test for Round 2
stat, p_value = wilcoxon(round2_dat_filtered['market_price'] - 1.94)
print(f'Round 2 Wilcoxon test statistic: {stat}, p-value: {p_value}')

# Combine Round 1 and Round 2
combined_dat_filtered = pd.concat([round1_dat_filtered, round2_dat_filtered])

# Perform Wilcoxon signed-rank test for combined data
stat, p_value = wilcoxon(combined_dat_filtered['market_price'] - 1.94)
print(f'Combined Wilcoxon test statistic: {stat}, p-value: {p_value}')
