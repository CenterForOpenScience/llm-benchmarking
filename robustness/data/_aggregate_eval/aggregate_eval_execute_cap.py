import json
import os
from pathlib import Path
import pandas as pd

ignore_study_ids = ["_aggregate_eval", ".DS_Store"]
data_path = "robustness/data/"
all_studies_ids = os.listdir(data_path)
all_studies_ids = [study_id for study_id in all_studies_ids if study_id not in ignore_study_ids]
eval_model = "gpt-5"

gt_path = "robustness/data/_aggregate_eval/gt/multi100_peer-eval-not-reviewed_processed_data.csv"
gt_df = pd.read_csv(gt_path)

eval_results = []
for study_id in all_studies_ids:
    path = Path(os.path.join(data_path, study_id))
    if path.is_dir():
        all_analysts_id = os.listdir(path)
        for analyst_id in all_analysts_id:
            path = Path(os.path.join(data_path, study_id, analyst_id))
            if path.is_dir():
                lookup_file = Path(os.path.join(data_path, study_id, 
                                                analyst_id, "evals", 
                                                eval_model, 
                                                "evaluated_execute_capability.json"))
                if lookup_file in Path(path).rglob('*'):
                    with open(lookup_file) as f:
                        eval_json = json.load(f)
                    print(study_id, analyst_id, eval_json)
                    rel_gt = gt_df[(gt_df['paper_id'] == study_id) & (gt_df['analyst_id'] == analyst_id)]
                    if len(rel_gt) > 0:
                        eval_results.append({
                            "paper_id": study_id,
                            "analyst_id": analyst_id,
                            "agent_exec_results": eval_json.get("evaluated_execute_performance", None),
                            "task1_pipeline_acceptable": rel_gt["task1_pipeline_acceptable"].to_list(),
                            "task1_conclusion_follows_results": rel_gt["task1_conclusion_follows_results"].to_list(),
                            "task1_categorisation_is_accurate": rel_gt["task1_categorisation_is_accurate"].to_list(),
                            "task2_pipeline_acceptable": rel_gt["task2_pipeline_acceptable"].to_list(),
                            "any_code_mismatches": rel_gt["any_code_mismatches"].to_list(),
                            "any_code_mismatches": rel_gt["any_code_mismatches"].to_list(),
                        })

                        
with open("robustness/data/_aggregate_eval/eval_execute_cap_aggregated.json", "w") as fout:
    json.dump(eval_results, fout, indent=2)