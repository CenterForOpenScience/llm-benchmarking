import pandas as pd
from scipy.stats import ranksums

# Load and clean TS Treatment decision data files
# Assuming the CSV files are in the /app/data/raw_data directory

def clean_ts_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['v2'] == 2]
    df = df[df['v3'] == 'subjects']
    df.drop(index=0, inplace=True)
    df.drop(columns=['v2', 'v3', 'v8', 'v9', 'v18', 'v19', 'v20', 'v21', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v50', 'v51', 'v52', 'v53', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60'], inplace=True)
    df.rename(columns={
        'v1': 'sessiondate_time',
        'v4': 'period',
        'v5': 'subject',
        'v6': 'group',
        'v7': 'profit',
        'v10': 'treatment',
        'v11': 'cost',
        'v12': 'prize1',
        'v13': 'prize2',
        'v14': 'prize3',
        'v15': 'support',
        'v16': 'effort_t1',
        'v17': 'effort_t2',
        'v22': 'random_t1',
        'v23': 'random_t2',
        'v24': 'output_t1',
        'v25': 'output_t2',
        'v26': 'cost_t1',
        'v27': 'cost_t2',
        'v34': 'expectation1_t1',
        'v35': 'expectation2_t1',
        'v36': 'expectation3_t1',
        'v37': 'expectation1_t2',
        'v38': 'expectation2_t2',
        'v45': 'rank',
        'v46': 'finalist',
        'v47': 'winner',
        'v48': 'time_spent_effort_t1',
        'v49': 'time_spent_belief_t1'
    }, inplace=True)
    return df

# Load and clean OS Treatment decision data files
def clean_os_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['v2'] == 2]
    df = df[df['v3'] == 'subjects']
    df.drop(index=0, inplace=True)
    df.drop(columns=['v2', 'v3', 'v8', 'v9', 'v18', 'v17', 'v19', 'v20', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v38', 'v39', 'v40', 'v41', 'v42', 'v48', 'v49', 'v50', 'v51'], inplace=True)
    df.rename(columns={
        'v1': 'sessiondate_time',
        'v4': 'period',
        'v5': 'subject',
        'v6': 'group',
        'v7': 'profit',
        'v10': 'treatment',
        'v11': 'cost',
        'v12': 'prize1',
        'v13': 'prize2',
        'v14': 'support',
        'v15': 'effort_t1',
        'v16': 'effort_t2',
        'v21': 'random_t1',
        'v22': 'random_t2',
        'v23': 'output_t1',
        'v24': 'output_t2',
        'v25': 'cost_t1',
        'v26': 'cost_t2',
        'v33': 'expectation1_t1',
        'v34': 'expectation2_t1',
        'v35': 'expectation3_t1',
        'v36': 'expectation1_t2',
        'v37': 'expectation2_t2',
        'v43': 'rank',
        'v44': 'finalist',
        'v45': 'winner',
        'v46': 'time_spent_effort_t1',
        'v47': 'time_spent_belief_t1'
    }, inplace=True)
    return df

# Perform ranksum test on effort_t1 by treatment
def perform_ranksum_test(df):
    ts_effort = df[df['treatment'] == 'TS']['effort_t1']
    os_effort = df[df['treatment'] == 'OS']['effort_t1']
    stat, p_value = ranksums(ts_effort, os_effort)
    return stat, p_value

# Example usage
# ts_data = clean_ts_data('/app/data/raw_data/211102_1054.csv')
# os_data = clean_os_data('/app/data/raw_data/211102_1054.csv')
# stat, p_value = perform_ranksum_test(ts_data)
# print(f'Ranksum Test Statistic: {stat}, p-value: {p_value}')

# Note: Adjust file paths and ensure all necessary files are available in the /app/data directory.