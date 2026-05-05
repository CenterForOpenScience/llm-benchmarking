import pandas as pd
from scipy.stats import ttest_ind

# Load the dataset
file_path = '/app/data/REPExperiment1DataBothRounds.dta'
data = pd.read_stata(file_path)

# Function to summarize and perform t-tests
def summarize_and_ttest(group, variable, priming_var):
    control_group = data[(data[group] == 1) & (data[priming_var] == 0)][variable]
    primed_group = data[(data[group] == 1) & (data[priming_var] == 1)][variable]
    
    # Summary statistics
    control_summary = control_group.describe()
    primed_summary = primed_group.describe()
    
    # T-test
    t_stat, p_value = ttest_ind(control_group, primed_group, nan_policy='omit')
    
    return control_summary, primed_summary, t_stat, p_value

# Asians
asian_control_summary, asian_primed_summary, asian_t_stat, asian_p_value = summarize_and_ttest('asian', 'impatient', 'givenprimingquestionnaire')

# Whites
white_control_summary, white_primed_summary, white_t_stat, white_p_value = summarize_and_ttest('white', 'impatient', 'givenprimingquestionnaire')

# Blacks
black_control_summary, black_primed_summary, black_t_stat, black_p_value = summarize_and_ttest('black', 'impatient', 'givenprimingquestionnaire')

# Print results
print('Asian Control Summary:', asian_control_summary)
print('Asian Primed Summary:', asian_primed_summary)
print('Asian T-test:', asian_t_stat, asian_p_value)

print('White Control Summary:', white_control_summary)
print('White Primed Summary:', white_primed_summary)
print('White T-test:', white_t_stat, white_p_value)

print('Black Control Summary:', black_control_summary)
print('Black Primed Summary:', black_primed_summary)
print('Black T-test:', black_t_stat, black_p_value)
