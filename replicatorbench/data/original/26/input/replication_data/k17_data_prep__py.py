import os
import pandas as pd
import numpy as np
import runpy

# Preferred IO paths under /app/data per policy
PREFERRED_RAW_BASE = "/app/data/replication_data/Raw data"
PREFERRED_OUT_PATH = "/app/data/replication_data/k17_processed_data.csv"


def run_analysis():
    candidates = [
        "/app/data/replication_data/k17_analysis__py.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "k17_analysis__py.py"),
        "/workspace/replication_data/k17_analysis__py.py",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Executing analysis script at: {path}")
            runpy.run_path(path, run_name="__main__")
            return
    raise FileNotFoundError(f"Analysis script not found. Checked: {candidates}")


def processed_exists():
    candidates = [
        PREFERRED_OUT_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "k17_processed_data.csv"),
        "/workspace/replication_data/k17_processed_data.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_raw_base():
    """Return a directory containing the expected raw CSVs, trying several fallbacks."""
    candidates = []
    # Preferred mount
    candidates.append(PREFERRED_RAW_BASE)
    # Directory next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "Raw data"))
    # Workspace fallback
    candidates.append("/workspace/replication_data/Raw data")

    for base in candidates:
        t1 = os.path.join(base, "k17_t1_data.csv")
        t2 = os.path.join(base, "k17_t2_data.csv")
        t3 = os.path.join(base, "k17_t3_data.csv")
        if os.path.exists(t1) and os.path.exists(t2) and os.path.exists(t3):
            return base
    # If none found, return preferred to trigger a clear error downstream
    return candidates[0]


def parse_date(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def save_outputs(df: pd.DataFrame):
    # Save to preferred /app/data path
    try:
        os.makedirs(os.path.dirname(PREFERRED_OUT_PATH), exist_ok=True)
        df.to_csv(PREFERRED_OUT_PATH, index=False)
        print(f"Saved processed data to {PREFERRED_OUT_PATH} with shape {df.shape}")
    except Exception as e:
        print(f"Warning: failed to save to {PREFERRED_OUT_PATH}: {e}")

    # Also save next to this script to ensure availability if /app/data not mounted
    script_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k17_processed_data.csv")
    try:
        df.to_csv(script_out, index=False)
        print(f"Saved processed data to {script_out} with shape {df.shape}")
    except Exception as e:
        print(f"Warning: failed to save to {script_out}: {e}")


def main():
    # Fast path: if processed data already exists, skip raw processing and just run analysis
    existing = processed_exists()
    if existing:
        print(f"Processed data already available at: {existing}. Skipping data prep and proceeding to analysis.")
        run_analysis()
        return

    RAW_BASE = find_raw_base()
    print(f"Using RAW_BASE: {RAW_BASE}")

    # Verify existence
    for fname in ["k17_t1_data.csv", "k17_t2_data.csv", "k17_t3_data.csv"]:
        fpath = os.path.join(RAW_BASE, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required raw file not found: {fpath}")

    # Load raw
    t1 = pd.read_csv(os.path.join(RAW_BASE, 'k17_t1_data.csv'))
    t2 = pd.read_csv(os.path.join(RAW_BASE, 'k17_t2_data.csv'))
    t3 = pd.read_csv(os.path.join(RAW_BASE, 'k17_t3_data.csv'))

    # Participants present in all waves
    inter = set(t1['RecipientEmail']).intersection(set(t2['RecipientEmail'])).intersection(set(t3['RecipientEmail']))
    t1i = t1[t1['RecipientEmail'].isin(inter)].copy()
    t2i = t2[t2['RecipientEmail'].isin(inter)].copy()
    t3i = t3[t3['RecipientEmail'].isin(inter)].copy()

    # Keep only those who finished questionnaire
    t1i = t1i[t1i['Finished'] == 1].copy()
    t2i = t2i[t2i['Finished'] == 1].copy()
    t3i = t3i[t3i['Finished'] == 1].copy()

    # Compute Wave based on time differences
    t1i['StartDate_dt'] = t1i['StartDate'].apply(parse_date)
    t2i['StartDate_dt'] = t2i['StartDate'].apply(parse_date)
    t3i['StartDate_dt'] = t3i['StartDate'].apply(parse_date)

    anchor = t1i['StartDate_dt'].iloc[0]
    T1 = t1i.drop(columns=[c for c in list(t1i.columns)[0:6] + list(t1i.columns)[8:11] + list(t1i.columns)[13:17] + [t1i.columns[84], t1i.columns[85]]])
    # Ensure StartDate_dt is retained
    T1['StartDate_dt'] = t1i['StartDate_dt'].values
    # Rename demographics and flags
    T1 = T1.rename(columns={
        T1.columns[0]: 'Finished_T1',
        T1.columns[2]: 'gender',
        T1.columns[3]: 'birthyear',
        T1.columns[4]: 'education',
        T1.columns[5]: 'children',
        T1.columns[6]: 'work_hours',
        T1.columns[7]: 'work_days',
    })

    # T2: drop columns by positions as in R and rename blocks
    T2 = t2i.drop(columns=[*list(t2i.columns)[18:77], t2i.columns[200]])
    T2 = T2.drop(columns=[*list(T2.columns)[0:6], *list(T2.columns)[8:11], *list(T2.columns)[13:17], T2.columns[140]])
    rename_seq = ['Finished_T2','req_detach_1','req_detach_2','req_detach_3','req_detach_4','req_relax_1','req_relax_2','req_relax_3','req_relax_4','req_mastery_1','req_mastery_2','req_mastery_3','req_mastery_4','req_control_1','req_control_2','req_control_3','req_control_4']
    for idx, nm in enumerate(rename_seq):
        T2 = T2.rename(columns={T2.columns[idx]: nm})

    # Hassles 1..53
    for k in range(19, 72):
        T2 = T2.rename(columns={T2.columns[k-1]: f'has_{k-18}'})
    # Uplifts 1..53
    for k in range(72, 125):
        T2 = T2.rename(columns={T2.columns[k-1]: f'upl_{k-71}'})

    # T3: drop like in R and rename PANAS T3 items
    T3 = t3i.drop(columns=[*list(t3i.columns)[0:6], *list(t3i.columns)[8:11], *list(t3i.columns)[13:17], *list(t3i.columns)[78:79]])
    T3 = T3.rename(columns={T3.columns[0]: 'Finished_T3'})
    panas_t3_names = [
        'T3_panas_jov_1','T3_panas_sad_1','T3_panas_gen_pos_2','T3_panas_guilt_4','T3_panas_hos_1','T3_panas_ser_2','T3_panas_guilt_2','T3_panas_jov_7','T3_panas_att_1','T3_panas_fear_1',
        'T3_panas_jov_4','T3_panas_sad_5','T3_panas_shy_1','T3_panas_fat_2','T3_panas_fear_3','T3_panas_shy_4','T3_panas_fat_1','T3_panas_sur_2','T3_panas_sad_4','T3_panas_gen_neg_2',
        'T3_panas_self_ass_1','T3_panas_fear_2','T3_panas_fat_3','T3_panas_guilt_5','T3_panas_sur_1','T3_panas_jov_3','T3_panas_jov_5','T3_panas_att_3','T3_panas_self_ass_2','T3_panas_shy_2',
        'T3_panas_hos_5','T3_panas_fear_6','T3_panas_hos_2','T3_panas_sad_2','T3_panas_self_ass_5','T3_panas_sur_3','T3_panas_ser_1','T3_panas_att_2','T3_panas_fear_4','T3_panas_gen_pos_3',
        'T3_panas_hos_3','T3_panas_gen_neg_1','T3_panas_jov_6','T3_panas_hos_6','T3_panas_jov_2','T3_panas_hos_4','T3_panas_guilt_3','T3_panas_self_ass_6','T3_panas_gen_pos_1','T3_panas_self_ass_4',
        'T3_panas_ser_3','T3_panas_jov_8','T3_panas_self_ass_3','T3_panas_sad_3','T3_panas_fear_5','T3_panas_att_4','T3_panas_guilt_1','T3_panas_shy_3','T3_panas_fat_4','T3_panas_guilt_6'
    ]
    for i, name in enumerate(panas_t3_names, start=3):
        T3 = T3.rename(columns={T3.columns[i-1]: name})

    # Compute Waves similar to R logic
    T1['Wave'] = np.round((T1['StartDate_dt'] - anchor).dt.total_seconds() / (7*24*3600)).astype('float') + 1
    T2['Wave'] = np.round(((t2i['StartDate_dt'] - anchor).dt.total_seconds() - 2*24*3600) / (7*24*3600)).astype('float') + 1
    T3['Wave'] = np.round(((t3i['StartDate_dt'] - anchor).dt.total_seconds() - 7*24*3600) / (7*24*3600)).astype('float') + 1

    # Merge
    score_all = pd.merge(T1, T2, on=['RecipientEmail', 'Wave'])
    score_all = pd.merge(score_all, T3, on=['RecipientEmail', 'Wave'])

    # Remove duplicated participants (keep first)
    score_all = score_all.drop_duplicates(subset=['RecipientEmail'])

    # Type conversions
    score_all['gender'] = pd.to_numeric(score_all['gender'], errors='coerce')
    score_all['birthyear'] = pd.to_numeric(score_all['birthyear'], errors='coerce')

    score_all['children'] = pd.to_numeric(score_all['children'], errors='coerce')
    score_all.loc[score_all['children'] >= 1, 'children'] = 1
    score_all.loc[score_all['children'] < 1, 'children'] = 0

    # Convert remaining to numeric where appropriate
    for col in score_all.columns[2:252]:
        score_all[col] = pd.to_numeric(score_all[col], errors='coerce')

    # Derived variables
    score_all['age'] = 2021 - score_all['birthyear']

    # REQ indices
    score_all['req_control'] = score_all[[f'req_control_{i}' for i in range(1,5)]].mean(axis=1)
    score_all['req_detach'] = score_all[[f'req_detach_{i}' for i in range(1,5)]].mean(axis=1)
    score_all['req_relax'] = score_all[[f'req_relax_{i}' for i in range(1,5)]].mean(axis=1)
    score_all['req_mastery'] = score_all[[f'req_mastery_{i}' for i in range(1,5)]].mean(axis=1)

    # Hassles total
    has_cols = [f'has_{i}' for i in range(1,54)]
    score_all['hassles'] = score_all[has_cols].sum(axis=1)

    # PANAS composites
    def rowmean(cols):
        return score_all[cols].mean(axis=1)

    # T1 PANAS indices
    score_all['T1_panas_negative'] = rowmean(['T1_panas_fear_1','T1_panas_fear_5','T1_panas_fear_3','T1_panas_fear_4','T1_panas_guilt_2','T1_panas_guilt_3','T1_panas_hos_3','T1_panas_hos_5','T1_panas_gen_neg_1','T1_panas_gen_neg_2'])
    score_all['T1_panas_fear'] = rowmean([f'T1_panas_fear_{i}' for i in [1,2,3,4,5,6]])
    score_all['T1_panas_sadness'] = rowmean([f'T1_panas_sad_{i}' for i in [1,2,3,4,5]])
    score_all['T1_panas_guilt'] = rowmean([f'T1_panas_guilt_{i}' for i in [1,2,3,4,5,6]])
    score_all['T1_panas_hostility'] = rowmean([f'T1_panas_hos_{i}' for i in [1,2,3,4,5,6]])
    score_all['T1_panas_shyness'] = rowmean([f'T1_panas_shy_{i}' for i in [1,2,3,4]])
    score_all['T1_panas_fatigue'] = rowmean([f'T1_panas_fat_{i}' for i in [1,2,3,4]])
    score_all['T1_panas_positive'] = rowmean(['T1_panas_gen_pos_1','T1_panas_gen_pos_2','T1_panas_gen_pos_3','T1_panas_att_2','T1_panas_att_1','T1_panas_jov_7','T1_panas_jov_5','T1_panas_self_ass_5','T1_panas_self_ass_2','T1_panas_att_3'])
    score_all['T1_panas_joviality'] = rowmean([f'T1_panas_jov_{i}' for i in [1,2,3,4,5,6,7,8]])
    score_all['T1_panas_self_assurance'] = rowmean([f'T1_panas_self_ass_{i}' for i in [1,2,3,4,5,6]])
    score_all['T1_panas_attentiveness'] = rowmean([f'T1_panas_att_{i}' for i in [1,2,3,4]])
    score_all['T1_panas_serenity'] = rowmean([f'T1_panas_ser_{i}' for i in [1,2,3]])
    score_all['T1_panas_surprise'] = rowmean([f'T1_panas_sur_{i}' for i in [1,2,3]])

    # T3 PANAS indices
    score_all['T3_panas_negative'] = rowmean(['T3_panas_fear_1','T3_panas_fear_5','T3_panas_fear_3','T3_panas_fear_4','T3_panas_guilt_2','T3_panas_guilt_3','T3_panas_hos_3','T3_panas_hos_5','T3_panas_gen_neg_1','T3_panas_gen_neg_2'])
    score_all['T3_panas_fear'] = rowmean([f'T3_panas_fear_{i}' for i in [1,2,3,4,5,6]])
    score_all['T3_panas_sadness'] = rowmean([f'T3_panas_sad_{i}' for i in [1,2,3,4,5]])
    score_all['T3_panas_guilt'] = rowmean([f'T3_panas_guilt_{i}' for i in [1,2,3,4,5,6]])
    score_all['T3_panas_hostility'] = rowmean([f'T3_panas_hos_{i}' for i in [1,2,3,4,5,6]])
    score_all['T3_panas_shyness'] = rowmean([f'T3_panas_shy_{i}' for i in [1,2,3,4]])
    score_all['T3_panas_fatigue'] = rowmean([f'T3_panas_fat_{i}' for i in [1,2,3,4]])
    score_all['T3_panas_positive'] = rowmean(['T3_panas_gen_pos_1','T3_panas_gen_pos_2','T3_panas_gen_pos_3','T3_panas_att_2','T3_panas_att_1','T3_panas_jov_7','T3_panas_jov_5','T3_panas_self_ass_5','T3_panas_self_ass_2','T3_panas_att_3'])
    score_all['T3_panas_joviality'] = rowmean([f'T3_panas_jov_{i}' for i in [1,2,3,4,5,6,7,8]])
    score_all['T3_panas_self_assurance'] = rowmean([f'T3_panas_self_ass_{i}' for i in [1,2,3,4,5,6]])
    score_all['T3_panas_attentiveness'] = rowmean([f'T3_panas_att_{i}' for i in [1,2,3,4]])
    score_all['T3_panas_serenity'] = rowmean([f'T3_panas_ser_{i}' for i in [1,2,3]])
    score_all['T3_panas_surprise'] = rowmean([f'T3_panas_sur_{i}' for i in [1,2,3]])

    save_outputs(score_all)

    # Proceed to analysis
    run_analysis()


if __name__ == '__main__':
    main()
