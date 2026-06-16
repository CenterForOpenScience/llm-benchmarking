import json, sys, os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Constants
DATA_PATH = "/app/data/data.csv"
RESULT_JSON_PATH = "/app/data/results_ph39a.json"
EXECUTION_RESULT_PATH = "/app/data/execution_result.json"

TASKS = [
    {
        "task_id": "Task1",
        "task_role": "conclusion_oriented_reanalysis",
        "path_name": "spearman_minik_psychopathology"
    },
    {
        "task_id": "Task2",
        "task_role": "comparable_result_oriented_reanalysis",
        "path_name": "spearman_minik_psychopathology_t2"
    }
]

MAIN_VARS = {
    "outcome": "DSM5_Total",
    "predictor": "MiniK_Total"
}


def one_sided_p_less(rho: float, p_two_sided: float) -> float:
    if np.isnan(rho) or np.isnan(p_two_sided):
        return np.nan
    # For H1: rho < 0
    if rho < 0:
        return max(min(p_two_sided / 2.0, 1.0), 0.0)
    else:
        return max(min(1.0 - (p_two_sided / 2.0), 1.0), 0.0)


def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    for col in [MAIN_VARS["predictor"], MAIN_VARS["outcome"]]:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in data.", file=sys.stderr)
            sys.exit(1)

    # Subset and drop missing
    sub = df[[MAIN_VARS["predictor"], MAIN_VARS["outcome"]]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    x = sub[MAIN_VARS["predictor" ]].to_numpy()
    y = sub[MAIN_VARS["outcome"   ]].to_numpy()
    n = int(len(sub))

    if n < 3:
        print("ERROR: Not enough observations after dropping missing values to compute Spearman correlation.", file=sys.stderr)
        sys.exit(1)

    rho, p_two = spearmanr(x, y)
    p_one_less = one_sided_p_less(rho, p_two)

    direction = "zero"
    if rho > 0:
        direction = "positive"
    elif rho < 0:
        direction = "negative"

    result_text = (
        f"Spearman correlation between {MAIN_VARS['predictor']} and {MAIN_VARS['outcome']} is "
        f"{rho:.4f} (two-sided p={p_two:.4g}; one-sided less p={p_one_less:.4g}; N={n})."
    )

    base_result = {
        "metric": "spearman_rho",
        "value": float(rho) if rho is not None else None,
        "direction": direction,
        "test_statistics": {
            "p_value": p_one_less,
            "t_value": None,
            "f_value": None,
            "z_value": None,
            "standard_error": None,
            "confidence_interval": None,
            "sample_size": n,
            "other": {
                "p_value_two_sided": p_two
            }
        },
        "result_text": result_text
    }

    # Determine conclusion
    if direction == "negative" and (not np.isnan(p_one_less)) and p_one_less < 0.05:
        conclusion_class = "support"
        conclusion_text = "The result supports the focal claim: Mini-K (slower) is negatively associated with psychopathology (one-sided p<0.05)."
    elif direction == "negative":
        conclusion_class = "inconclusive"
        conclusion_text = "Negative association observed but not statistically significant (one-sided p>=0.05)."
    elif direction == "positive":
        conclusion_class = "opposite"
        conclusion_text = "Observed association is positive, opposite the focal claim."
    else:
        conclusion_class = "inconclusive"
        conclusion_text = "No clear association detected."

    # Prepare per-task outputs (identical stats for Task1 and Task2 in this plan)
    task_outputs = []
    for t in TASKS:
        task_outputs.append({
            "task_id": t["task_id"],
            "task_role": t["task_role"],
            "execution_status": "success",
            "executed_analysis": {
                "path_name": t["path_name"],
                "software": "Python",
                "model_family": "correlation",
                "executed_files": [
                    "data/analysis__py.py"
                ],
                "run_command": "python data/analysis__py.py",
                "code_source": "provided_analysis_code"
            },
            "method_fidelity": {
                "followed_planned_path": "yes",
                "deviations": [],
                "fidelity_note": "Followed the planned Spearman correlation path using the Mini-K total and DSM5 total."
            },
            "result_raw": base_result,
            "conclusion": {
                "conclusion_class": conclusion_class,
                "conclusion_text": conclusion_text
            },
            "artifacts": {
                "output_files": [
                    "data/results_ph39a.json",
                    "data/execution_result.json"
                ]
            },
            "failure": {
                "failure_reason": None,
                "failure_stage": None,
                "repair_attempts_used": 0
            }
        })

    overall_summary = (
        "Executed Spearman correlation between MiniK_Total and DSM5_Total. "
        f"Observed rho={rho:.4f} with one-sided less p={p_one_less:.4g} (N={n})."
    )

    execution_result = {
        "execution_overview": {
            "overall_execution_status": "success",
            "target_task_ids": [t["task_id"] for t in TASKS],
            "completed_task_ids": [t["task_id"] for t in TASKS],
            "failed_task_ids": [],
            "overall_summary": overall_summary
        },
        "task_outputs": task_outputs
    }

    # Write compact results and the standardized execution result
    simple_result = {
        "vars": {
            "predictor": MAIN_VARS["predictor"],
            "outcome": MAIN_VARS["outcome"]
        },
        "n": n,
        "rho": float(rho),
        "p_value_two_sided": float(p_two),
        "p_value_one_sided_less": float(p_one_less),
        "direction": direction,
        "result_text": result_text
    }

    with open(RESULT_JSON_PATH, "w") as f:
        json.dump(simple_result, f, indent=2)

    with open(EXECUTION_RESULT_PATH, "w") as f:
        json.dump(execution_result, f, indent=2)

    # Also print the simple result to stdout for logging
    print(json.dumps(simple_result, indent=2))


if __name__ == "__main__":
    main()
