import os
import re
import glob
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ttest_rel

DATA_DIR = "/app/data"


def _safe_get(txt, i, j):
    try:
        return txt[i][j]
    except Exception:
        return ""


def _to_str_array(df: pd.DataFrame) -> List[List[str]]:
    # Convert DataFrame to 2D list of strings, preserving empty cells as ''
    arr = df.astype(object).where(pd.notnull(df), None).values.tolist()
    out: List[List[str]] = []
    for row in arr:
        out_row: List[str] = []
        for val in row:
            if val is None:
                out_row.append("")
            else:
                out_row.append(str(val))
        out.append(out_row)
    return out


def _parse_trials_from_day(xlsx_path: str) -> Dict[str, List]:
    # Read excel; skip first 5 header rows to mirror MATLAB txt(6:end,:)
    try:
        df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file '{xlsx_path}': {e}")

    txt = _to_str_array(df.iloc[5:, :])

    trial_index = []
    exposure_duration = []  # ms
    answer_value = []  # 1/0 correct
    sound_cue = []  # True if 'Sound' line present as in MATLAB (col 3 two rows above)

    for i in range(len(txt)):
        cell_c4 = _safe_get(txt, i, 3)  # MATLAB column 4 => index 3
        if "target: " not in cell_c4:
            trial_index.append(0)
            exposure_duration.append(np.nan)
            answer_value.append(np.nan)
            sound_cue.append(np.nan)
            continue

        # Mark as trial
        trial_index.append(1)

        # Exposure duration: last 3 characters of cell_c4
        try:
            exposure_duration.append(float(cell_c4[-3:]))
        except Exception:
            # Fallback: extract any trailing number
            m = re.search(r"(\d+)$", cell_c4)
            exposure_duration.append(float(m.group(1)) if m else np.nan)

        # Correct answer present two rows below column 4
        ans_c4 = _safe_get(txt, i + 2, 3)
        answer_value.append(1.0 if ans_c4 == "correct answer" else 0.0)

        # Sound cue flag: two rows above, column 3 equals 'Sound'
        sound_flag = _safe_get(txt, i - 2, 2)
        sound_cue.append(True if sound_flag == "Sound" else False)

    findex = [idx for idx, v in enumerate(trial_index) if v == 1]
    trial_exposures = np.array([exposure_duration[i] for i in findex], dtype=float)
    trial_answers = np.array([answer_value[i] for i in findex], dtype=float)
    trial_sounds = np.array([1 if sound_cue[i] else 0 for i in findex], dtype=float)

    return {
        "trial_exposures": trial_exposures,
        "trial_answers": trial_answers,
        "trial_sounds": trial_sounds,
    }


def _aggregate_prob_by_duration(trial_exposures: np.ndarray,
                                trial_answers: np.ndarray,
                                trial_sounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Compute unique durations
    durations = np.unique(trial_exposures[~np.isnan(trial_exposures)]).astype(float)
    durations = np.sort(durations)
    if durations.size == 0:
        return durations, np.array([]), np.array([])

    prob_cue = []
    prob_nocue = []
    for d in durations:
        mask_d = (trial_exposures == d)
        # Cue
        mask_cue = mask_d & (trial_sounds == 1)
        n_cue = np.sum(mask_cue)
        p_cue = np.nan if n_cue == 0 else float(np.sum(trial_answers[mask_cue]) / n_cue)
        prob_cue.append(p_cue)
        # No cue
        mask_nc = mask_d & (trial_sounds == 0)
        n_nc = np.sum(mask_nc)
        p_nc = np.nan if n_nc == 0 else float(np.sum(trial_answers[mask_nc]) / n_nc)
        prob_nocue.append(p_nc)

    return durations, np.array(prob_cue, dtype=float), np.array(prob_nocue, dtype=float)


def _model_prob(t: np.ndarray, v: float, t0: float, pg: float) -> np.ndarray:
    # Mirror MATLAB objective (no clamping for t < t0 inside objective)
    return (1.0 - np.exp(-v * (t - t0))) + np.exp(-v * (t - t0)) * pg * (1.0 / 20.0)


def _fit_params(t_ms: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    # Prepare data: drop NaNs aligned
    mask = ~np.isnan(t_ms) & ~np.isnan(y)
    t_ms = t_ms[mask]
    y = y[mask]
    if t_ms.size == 0:
        return np.nan, np.nan, np.nan

    t = t_ms / 1000.0

    def sse(x):
        v, t0, pg = x[0], x[1], x[2]
        yhat = _model_prob(t, v, t0, pg)
        return float(np.nansum((y - yhat) ** 2))

    x0 = np.array([20.0, 0.0, 0.0], dtype=float)
    res = minimize(sse, x0=x0, method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    v, t0, pg = res.x
    return float(v), float(t0), float(pg)


def analyze_subject(subject_id: str, day_files: List[str]) -> Dict[str, float]:
    # Combine day1 and day2 trials
    all_trials = []
    for f in sorted(day_files):
        try:
            trials = _parse_trials_from_day(f)
            all_trials.append(trials)
        except Exception as e:
            warnings.warn(f"Skipping file due to parse error: {f} -> {e}")

    if not all_trials:
        return {
            "subject_id": subject_id,
            "v_Cue": np.nan,
            "v_NoCue": np.nan,
            "t0_Cue_ms": np.nan,
            "t0_NoCue_ms": np.nan,
            "pg_Cue": np.nan,
            "pg_NoCue": np.nan,
        }

    trial_exposures = np.concatenate([t["trial_exposures"] for t in all_trials])
    trial_answers = np.concatenate([t["trial_answers"] for t in all_trials])
    trial_sounds = np.concatenate([t["trial_sounds"] for t in all_trials])

    durations, prob_cue, prob_nocue = _aggregate_prob_by_duration(trial_exposures, trial_answers, trial_sounds)

    v_cue, t0_cue, pg_cue = _fit_params(durations, prob_cue)
    v_nocue, t0_nocue, pg_nocue = _fit_params(durations, prob_nocue)

    return {
        "subject_id": subject_id,
        "v_Cue": v_cue,
        "v_NoCue": v_nocue,
        "t0_Cue_ms": 1000.0 * t0_cue if not np.isnan(t0_cue) else np.nan,
        "t0_NoCue_ms": 1000.0 * t0_nocue if not np.isnan(t0_nocue) else np.nan,
        "pg_Cue": pg_cue,
        "pg_NoCue": pg_nocue,
    }


def run_analysis(save_outputs: bool = True) -> Dict:
    # Discover files: pattern '####_Day#_*.xlsx'
    pattern_glob = "[0-9][0-9][0-9][0-9]_Day[12]_*.xlsx"
    candidate_dirs = [
        DATA_DIR,
        "/workspace/replication_data",
        os.path.join(os.getcwd(), "replication_data"),
    ]
    files = []
    for d in candidate_dirs:
        if d and os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, pattern_glob)))
    files = sorted(set(files))
    if len(files) == 0:
        raise FileNotFoundError(
            "No Excel files found matching pattern '####_Day#_*.xlsx' in any of: " + ", ".join(candidate_dirs) + ". "
            "Please place the replication .xlsx logs into /app/data or ensure they are under replication_data.")

    subj_to_files: Dict[str, List[str]] = {}
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"^(\d{4})_Day([12])_.*\.xlsx$", base)
        if not m:
            continue
        sid = m.group(1)
        subj_to_files.setdefault(sid, []).append(f)

    results: List[Dict[str, float]] = []
    for sid, day_files in sorted(subj_to_files.items(), key=lambda x: x[0]):
        res = analyze_subject(sid, day_files)
        results.append(res)

    df = pd.DataFrame(results)
    # Drop subjects with missing either param for paired tests
    valid_mask_v = df[["v_Cue", "v_NoCue"]].notnull().all(axis=1)
    valid_mask_t0 = df[["t0_Cue_ms", "t0_NoCue_ms"]].notnull().all(axis=1)

    t_v = p_v = np.nan
    n_v = int(valid_mask_v.sum())
    if n_v >= 2:
        tstat_v, pval_v = ttest_rel(df.loc[valid_mask_v, "v_NoCue"], df.loc[valid_mask_v, "v_Cue"])
        t_v, p_v = float(tstat_v), float(pval_v)

    t_t0 = p_t0 = np.nan
    n_t0 = int(valid_mask_t0.sum())
    if n_t0 >= 2:
        tstat_t0, pval_t0 = ttest_rel(df.loc[valid_mask_t0, "t0_NoCue_ms"], df.loc[valid_mask_t0, "t0_Cue_ms"])
        t_t0, p_t0 = float(tstat_t0), float(pval_t0)

    summary = {
        "n_subjects_total": int(len(df)),
        "n_subjects_used_v": n_v,
        "n_subjects_used_t0": n_t0,
        "paired_t_v_NoCue_vs_Cue": {"t": t_v, "p": p_v, "direction": "NoCue < Cue expected (positive t implies v_Cue > v_NoCue)"},
        "paired_t_t0_NoCue_vs_Cue": {"t": t_t0, "p": p_t0, "direction": "NoCue > Cue expected (negative t implies t0_Cue < t0_NoCue)"},
    }

    if save_outputs:
        out_params = os.path.join(DATA_DIR, "replication_tva_params.csv")
        out_tests = os.path.join(DATA_DIR, "replication_ttests.json")
        df.to_csv(out_params, index=False)
        with open(out_tests, "w") as f:
            json.dump(summary, f, indent=2)

    return {"subject_params": results, "summary": summary}


if __name__ == "__main__":
    res = run_analysis(save_outputs=True)
    print(json.dumps(res["summary"], indent=2))
