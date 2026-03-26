import pandas as pd

# Load datasets
t1 = pd.read_csv('/app/data/Raw data/k17_t1_data.csv')
t2 = pd.read_csv('/app/data/Raw data/k17_t2_data.csv')
t3 = pd.read_csv('/app/data/Raw data/k17_t3_data.csv')

# Identifying participants present in all waves
intersection = set(t1['RecipientEmail']).intersection(t2['RecipientEmail'], t3['RecipientEmail'])

t1i = t1[t1['RecipientEmail'].isin(intersection) & (t1['Finished'] == 1)]
t2i = t2[t2['RecipientEmail'].isin(intersection) & (t2['Finished'] == 1)]
t3i = t3[t3['RecipientEmail'].isin(intersection) & (t3['Finished'] == 1)]

# Updating column names
T1 = t1i.drop(columns=t1i.columns[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 83, 84]])
T1.columns = ['Finished_T1', 'gender', 'birthyear', 'education', 'children', 'work_hours', 'work_days'] + [f'T1_panas_{i}' for i in range(1, 62)]

T2 = t2i.drop(columns=t2i.columns[list(range(17, 77)) + [200]])
T2 = T2.drop(columns=T2.columns[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 139]])
T2.columns = ['Finished_T2'] + [f'req_detach_{i}' for i in range(1, 5)] + [f'req_relax_{i}' for i in range(1, 5)] + [f'req_mastery_{i}' for i in range(1, 5)] + [f'req_control_{i}' for i in range(1, 5)] + [f'has_{i}' for i in range(1, 55)] + [f'upl_{i}' for i in range(1, 55)]

T3 = t3i.drop(columns=t3i.columns[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 77, 78]])
T3.columns = ['Finished_T3'] + [f'T3_panas_{i}' for i in range(1, 63)]

# Combining into one dataset
score_all = pd.merge(T1, T2, on='RecipientEmail')
score_all = pd.merge(score_all, T3, on=['RecipientEmail', 'Wave'])

# Removing duplicates
score_all = score_all.drop_duplicates(subset='RecipientEmail')

# Dealing with factors
score_all['gender'] = pd.to_numeric(score_all['gender'], errors='coerce')
score_all['birthyear'] = pd.to_numeric(score_all['birthyear'], errors='coerce')

score_all['children'] = pd.to_numeric(score_all['children'], errors='coerce')
score_all['children'] = (score_all['children'] >= 1).astype(int)

# Recode age
score_all['age'] = 2021 - score_all['birthyear']

# Recovery Experiences Questionnaire
score_all['req_control'] = score_all[[f'req_control_{i}' for i in range(1, 5)]].mean(axis=1)
score_all['req_detach'] = score_all[[f'req_detach_{i}' for i in range(1, 5)]].mean(axis=1)
score_all['req_relax'] = score_all[[f'req_relax_{i}' for i in range(1, 5)]].mean(axis=1)
score_all['req_mastery'] = score_all[[f'req_mastery_{i}' for i in range(1, 5)]].mean(axis=1)

# Hassles
score_all['hassles'] = score_all[[f'has_{i}' for i in range(1, 54)]].sum(axis=1)

# PANAS T1
score_all['T1_panas_joviality'] = score_all[[f'T1_panas_{i}' for i in range(1, 9)]].mean(axis=1)

# PANAS T3
score_all['T3_panas_joviality'] = score_all[[f'T3_panas_{i}' for i in range(1, 9)]].mean(axis=1)

# Save processed data
score_all.to_csv('/app/data/k17_processed_data.csv', index=False)
