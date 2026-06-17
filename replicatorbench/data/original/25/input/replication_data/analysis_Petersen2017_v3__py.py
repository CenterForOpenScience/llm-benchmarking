import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to perform Nelder-Mead curve fitting

def curve_fit_nelder_mead(tdata, ydata):
    def objective(x):
        return np.sum((ydata - ((1 - np.exp(-x[0] * (tdata - x[1]))) + (np.exp(-x[0] * (tdata - x[1]))) * x[2] * 1/20)) ** 2)

    x0 = [20, 0.025, 0.5]  # Initial guesses for v, t0, pg
    result = minimize(objective, x0, method='Nelder-Mead')
    return result.x

# Main analysis function

def analysis_petersen(subj=None):
    filelist = ['1901_Day1_10-24-2020', '1902_Day1_11-03-2020', '1902_Day2_11-12-2020', '1903_Day1_11-11-2020', '1903_Day2_11-17-2020']
    subjlist = list(set(filelist)) if subj is None else [subj]

    datadir = '/app/data/'
    subjdata = []

    for subj in subjlist:
        daydata = []
        for iday in range(1, 3):  # Assuming two days per subject
            filepath = os.path.join(datadir, f'{subj}-VAT_replication_scenario.xlsx')
            df = pd.read_excel(filepath, skiprows=5)

            trial_index = []
            exposure_duration = []
            answer_value = []
            sound_cue = []
            CTOA = []
            ITI = []
            target_value = []
            response_value = []

            for i, row in df.iterrows():
                if 'target: ' not in row[3]:
                    trial_index.append(0)
                    exposure_duration.append(np.nan)
                    answer_value.append(np.nan)
                    sound_cue.append(np.nan)
                    CTOA.append(np.nan)
                    ITI.append(np.nan)
                    target_value.append(np.nan)
                    response_value.append(np.nan)
                else:
                    trial_index.append(1)
                    workingstring = row[3]
                    exposure_duration.append(float(workingstring[-3:]))
                    answer_value.append(row[5] == 'correct answer')
                    sound_cue.append(row[2] == 'Sound')
                    CTOA.append(float(row[3][-3:]))
                    ITI.append(float(row[3][-4:]))
                    target_value.append(float(workingstring.split(':')[1].split(';')[0]))
                    response_value.append(row[2])

            findex = [i for i, x in enumerate(trial_index) if x == 1]
            trial_exposures = [exposure_duration[i] for i in findex]
            trial_answers = [answer_value[i] for i in findex]
            trial_sounds = [sound_cue[i] for i in findex]

            durations = list(set(trial_exposures))
            prob_Cue = [sum(trial_answers[i] for i in range(len(trial_exposures)) if trial_exposures[i] == d and trial_sounds[i] == 1) /
                        sum(1 for i in range(len(trial_exposures)) if trial_exposures[i] == d and trial_sounds[i] == 1) for d in durations]
            prob_NoCue = [sum(trial_answers[i] for i in range(len(trial_exposures)) if trial_exposures[i] == d and trial_sounds[i] == 0) / 
                          sum(1 for i in range(len(trial_exposures)) if trial_exposures[i] == d and trial_sounds[i] == 0) for d in durations]

            daydata.append({'prob_Cue': prob_Cue, 'prob_NoCue': prob_NoCue})

        # Combine data from Day1 and Day2
        combined_prob_Cue = np.mean([d['prob_Cue'] for d in daydata], axis=0)
        combined_prob_NoCue = np.mean([d['prob_NoCue'] for d in daydata], axis=0)

        tdata = np.array(durations) / 1000
        v_Cue, t0_Cue, pg_Cue = curve_fit_nelder_mead(tdata, combined_prob_Cue)
        v_NoCue, t0_NoCue, pg_NoCue = curve_fit_nelder_mead(tdata, combined_prob_NoCue)

        subjdata.append({'v_Cue': v_Cue, 't0_Cue': t0_Cue, 'pg_Cue': pg_Cue,
                         'v_NoCue': v_NoCue, 't0_NoCue': t0_NoCue, 'pg_NoCue': pg_NoCue})

    return subjdata

# Example usage
if __name__ == '__main__':
    results = analysis_petersen()
    print(results)
