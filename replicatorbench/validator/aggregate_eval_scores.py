import os
import json

def summarize_eval_execute(eval_data):
    eval_scores = {}
    for sub_stage, sub_stage_eval_data in eval_data.items():
        eval_scores[f"execute_{sub_stage}"] = {
            "aspect_scores": {}
        }
        sub_stage_scores = []
        for aspect in sub_stage_eval_data:
            clean_scores = []
            for rubric_id, rubric_info in sub_stage_eval_data[aspect].items():
                raw_score = rubric_info.get('score')
                # Safely attempt to convert the score to a float
                try:
                    clean_scores.append(float(raw_score))
                except (ValueError, TypeError):
                    # If the LLM output "N/A" or something weird, we just skip it
                    continue 
            
            # Calculate average, protecting against division by zero
            if clean_scores:
                aspect_avg = sum(clean_scores) / len(clean_scores)
            else:
                aspect_avg = 0.0
                
            eval_scores[f"execute_{sub_stage}"]["aspect_scores"][aspect] = aspect_avg
            sub_stage_scores.append(aspect_avg)
            
        # Calculate sub-stage average, protecting against division by zero
        if sub_stage_scores:
            eval_scores[f"execute_{sub_stage}"]["avg_score"] = sum(sub_stage_scores) / len(sub_stage_scores)
        else:
            eval_scores[f"execute_{sub_stage}"]["avg_score"] = 0.0
            
    return eval_scores

def _to_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s.upper() in {"NA", "N/A", ""}:
            return None
        return float(s)  # will still raise if it's something else
    return None

def summarize_eval_scores(study_path, evaluator_model="gpt-4o"):
    stages = ["extract", "design", "execute", "interpret"]
    eval_summary = {}
    
    # Use the dynamic evaluation directory
    evals_dir = os.path.join(study_path, "evals", evaluator_model)
    
    for stage in stages:
        stage_file_path = os.path.join(evals_dir, f"{stage}_llm_eval.json")
        
        # Safety check: If the execute JSON (or any other) is missing, skip it without crashing!
        if not os.path.exists(stage_file_path):
            print(f"Skipping {stage} evaluation - file not found: {stage_file_path}")
            continue
            
        with open(stage_file_path) as f:
            eval_json = json.load(f)
            
        if stage == "execute":
            # If the LLM suffered from prompt leakage and hallucinated the wrong schema, 
            # gracefully skip it or handle it instead of crashing.
            if "evaluate_design" not in eval_json or "execute" not in eval_json:
                print(f"[WARNING] Schema mismatch in {stage_file_path}! The LLM hallucinated the wrong JSON structure. Skipping execution scoring.")
                continue
            eval_data = {
                "design": eval_json["evaluate_design"],
                "execute": eval_json["execute"] 
            }
            eval_summary.update(summarize_eval_execute(eval_data))
        else:
            aspect_totals = {}
            for eval_field, eval_info in eval_json.items():
                aspect = eval_field.split(".")[0]
                if aspect not in aspect_totals:
                    aspect_totals[aspect] = [0.0, 0.0]
                score = _to_float_or_none(eval_info.get("score"))
                if score is None:
                    continue
                aspect_totals[aspect][0] += score
                aspect_totals[aspect][1] += 3.0

            eval_summary[stage] = {"aspect_scores": {}}
            stage_scores = []
            for aspect, (score_sum, max_sum) in aspect_totals.items():
                aspect_avg = (score_sum / max_sum) if max_sum else 0.0
                eval_summary[stage]["aspect_scores"][aspect] = aspect_avg
                stage_scores.append(aspect_avg)

            eval_summary[stage]["avg_score"] = (
                sum(stage_scores) / len(stage_scores) if stage_scores else 0.0
            )
    
    summary_file_path = os.path.join(evals_dir, "eval_summary.json")
    with open(summary_file_path, "w") as fout:
        json.dump(eval_summary, fout, indent=2)