"""
Python translation of analysis_Petersen2017_v3.m
Performs TVA model fitting for each participant based on the Excel logs in
replication_data folder. The script:
1. Searches for all files matching the pattern "[0-9][0-9][0-9][0-9]_Day*.xlsx".
2. Groups files by participant id (the 4-digit prefix).
3. For each participant it extracts trial level information (exposure duration, accuracy, sound cue).
4. Computes proportion correct for each exposure duration in Cue vs No-Cue conditions.
5. Fits TVA single-letter model using Nelder Mead to obtain v, t0, pg.
6. Outputs a CSV summary of v and t0 for each participant and prints a paired t-test comparing Cue vs No-Cue v and t0.
All files are assumed to live under "/app/data" when the container runs.  
"""

import glob
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
# Ensure openpyxl# Ensure openpyxl is installed for pandas Excel engine
import openpyxl  # noqa: F401

from scipy.optimizefrom scipy.optimize import fmin
from scipy import stats

DATA_ROOT = "/app/data"  # docker policy path

# Search recursively for replication excel files
pattern = os.path.join(DATA_ROOT, "**", "[0-9][0-9][0-9][0-9]_Day*.xlsx")
all_files = glob.glob(pattern, recursive=True)
if not all_files:
    raise FileNotFoundError("No replication Excel files were found under /app/data. Ensure the replication_data folder is mounted correctly.")

# Group by participant id (first 4 digits)
file_dict = defaultdict(list)
for f in all_files:
    basename = os.path.basename(f)
    m = re.match(r"(\d{4})_Day(\d).*", basename)
    if m:
        pid = m.group(1)
        file_dict[pid].append(f)

print(f"Found {len(file_dict)} participants")

def extract_trials(xlsx_path):
    """Extract trial information from a single day file"""
    # The MATLAB code reads with xlsread, skipping first 5 rows then parsing text column 4 etc.
    # We'll mimic this by reading as pandas and applying similar rules.
    df = pd.read_excel(xlsx_path, header=None, dtype=str)
    # Ensure at least 6 rows
    if df.shape[0] < 6:
        return None
    txt = df.iloc[5:, :]  # zero-indexed skip first 5 rows
    # Convert to list of lists for faster iteration
    txt_values = txt.values.tolist()

    trial_rows = []
    for i, row in enumerate(txt_values):
        col4 = row[3] if len(row) > 3 else None
        if col4 and isinstance(col4, str) and "target:" in col4:
            # This is a trial row
            try:
                exposure_duration = float(col4[-3:])
            except ValueError:
                exposure_duration = np.nan
            # answer value in row i+2, col4 equals "correct answer"
            answer_value = 1 if (i + 2 < len(txt_values) and txt_values[i + 2][3] == "correct answer") else 0
            # sound cue present if row i-2, col3 == 'Sound'
            sound_cue = 1 if (i - 2 >= 0 and len(txt_values[i - 2]) > 2 and txt_values[i - 2][2] == "Sound") else 0
            trial_rows.append((exposure_duration, answer_value, sound_cue))
    return trial_rows


def curve_fit(tdata, ydata):
    """Fit TVA model parameters using Nelder-Mead optimisation. Returns v, t0, pg."""
    def loss(x):
        v, t0, pg = x
        # Numerical stability: ensure pg in [0,1], t0 >=0, v>0
        if v <= 0 or t0 < 0 or not (0 <= pg <= 1):
            return 1e6
        model = 1 - np.exp(-v * (tdata - t0)) + np.exp(-v * (tdata - t0)) * pg * 1 / 20
        model[tdata < t0] = 0
        return np.sum((ydata - model) ** 2)

    x0 = np.array([20, 0.025, 0.5])
    bestx = fmin(loss, x0, disp=False)
    return bestx

results = []

durations_expected = None

for pid, files in file_dict.items():
    all_trials = []
    for f in files:
        trials = extract_trials(f)
        if trials:
            all_trials.extend(trials)
    if not all_trials:
        continue
    # Create arrays
    exposure = np.array([t[0] for t in all_trials])
    accuracy = np.array([t[1] for t in all_trials])
    cue = np.array([t[2] for t in all_trials])

    unique_durs = np.unique(exposure)
    if durations_expected is None:
        durations_expected = unique_durs
    # proportions
    prob_cue = []
    prob_nocue = []
    for dur in unique_durs:
        idx_cue = (exposure == dur) & (cue == 1)
        idx_nc = (exposure == dur) & (cue == 0)
        prob_cue.append(accuracy[idx_cue].mean() if idx_cue.any() else np.nan)
        prob_nocue.append(accuracy[idx_nc].mean() if idx_nc.any() else np.nan)
    prob_cue = np.array(prob_cue)
    prob_nocue = np.array(prob_nocue)

    # Fit curves (convert ms to s)
    tdata = unique_durs / 1000.0
    v_cue, t0_cue, pg_cue = curve_fit(tdata, prob_cue)
    v_nc, t0_nc, pg_nc = curve_fit(tdata, prob_nocue)

    results.append({
        "participant": pid,
        "v_cue": v_cue,
        "v_nocue": v_nc,
        "t0_cue": t0_cue,
        "t0_nocue": t0_nc,
        "pg_cue": pg_cue,
        "pg_nocue": pg_nc,
    })

# Convert to DataFrame and save
res_df = pd.DataFrame(results)
output_csv = os.path.join(DATA_ROOT, "tva_replication_summary.csv")
res_df.to_csv(output_csv, index=False)
print(f"Saved per-participant TVA parameters to {output_csv}")

# Paired t-tests for v and t0
v_t = stats.ttest_rel(res_df["v_cue"], res_df["v_nocue"])
t0_t = stats.ttest_rel(res_df["t0_cue"], res_df["t0_nocue"])
print("Paired t-test results:")
print(f"Processing speed v: t = {v_t.statistic:.3f}, p = {v_t.pvalue:.4f}")
print(f"Perceptual threshold t0: t = {t0_t.statistic:.3f}, p = {t0_t.pvalue:.4f}")
