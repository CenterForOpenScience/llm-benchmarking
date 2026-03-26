import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Load the data
all_data = pd.read_csv('/app/data/analysis-data.csv')

# Filter subjects who indicate using external materials to help remember
all_data['response'] = all_data['response'].apply(lambda x: eval(x) if isinstance(x, str) else x)
good_subjects = all_data[(all_data['trial_type'] == 'survey-multi-choice') & (all_data['correct'].isna()) & (all_data['study_id'] != '60edd13110225f80f9f3709b')]
good_subjects = good_subjects[good_subjects['response'].apply(lambda x: 'No' in x)]

# Count groupings
n_subjects = all_data['subject'].nunique()
n_included_subjects = good_subjects['subject'].nunique()
n_full_info = good_subjects[good_subjects['condition'] == 'full-information'].shape[0]
n_conditional = good_subjects[good_subjects['condition'] == 'contingent'].shape[0]

# Filter to Critical Data
data = all_data[all_data['subject'].isin(good_subjects['subject'])]
data = data[data['trial_type'] == 'image-button-response']
data['phase'] = np.where(data['action'].isna(), 'Test', 'Learning')

# Calculate participant 1D and 2D scores at test
test_data = data[data['phase'] == 'Test']
test_summary = test_data.groupby(['subject', 'condition']).apply(lambda df: pd.Series({
    'acc_2d': df['correct'].mean(),
    'acc_1d_a': df['correct_1d_a'].mean(),
    'acc_1d_b': df['correct_1d_b'].mean()
})).reset_index()
test_summary['acc_1d'] = test_summary[['acc_1d_a', 'acc_1d_b']].max(axis=1)

# Group test summary
group_test_summary = test_summary.groupby('condition').agg(M=('acc_1d', 'mean'), SD=('acc_1d', 'std')).reset_index()

# Perform t-test on 1D accuracy scores by information condition
contingent_group = test_summary[test_summary['condition'] == 'contingent']
full_info_group = test_summary[test_summary['condition'] == 'full-information']
t_stat, p_value = ttest_ind(contingent_group['acc_1d'], full_info_group['acc_1d'], equal_var=True)

# Calculate effect size
cohens_d = (group_test_summary.loc[group_test_summary['condition'] == 'contingent', 'M'].values[0] - group_test_summary.loc[group_test_summary['condition'] == 'full-information', 'M'].values[0]) / np.sqrt((group_test_summary['SD'] ** 2).mean())
            group_test_summary.loc[group_test_summary['condition'] == 'full-information', 'M'].values[0]) / np.sqrt((group_test_summary['SD'] ** 2).mean())
           np.sqrt((group_test_summary['SD'] ** 2).mean())

# Print results
print(f"T-test: t-statistic = {t_stat}, p-value = {p_value}")
print(f"Cohen's d: {cohens_d}")
