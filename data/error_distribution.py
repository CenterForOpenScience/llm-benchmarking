import os
import re
import json
import pandas as pd
import glob

RESULTS_DIR = "./results"
MODES = ["native", "python"]
LOG_PATTERN = "execute_*.log"

def parse_log_file(file_path):
    """
    Parses a single log file to extract error signatures.
    Returns a list of error dictionaries.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return []

    errors_found = []
    turns = content.split('--- Turn')
    
    for turn_idx, turn_content in enumerate(turns):
        if turn_idx == 0: continue 

        # 1. Tool Failures (JSON "ok": false)
        obs_match = re.search(r'\*\*\*Agent input: Observation: ({.*})', turn_content, re.DOTALL)
        if obs_match:
            try:
                json_str = obs_match.group(1).split('\n202')[0]
                data = json.loads(json_str)
                
                if data.get("ok") is False:
                    err_msg = (
                        data.get("error") or 
                        data.get("message") or 
                        data.get("failure") or 
                        data.get("content") or 
                        json.dumps(data)
                    )
                    errors_found.append({
                        "type": "Tool Failure",
                        "raw_error": str(err_msg)[:300], 
                        "turn": turn_idx
                    })
            except json.JSONDecodeError:
                errors_found.append({
                    "type": "Tool Failure", 
                    "raw_error": "Invalid JSON Observation", 
                    "turn": turn_idx
                })

        # 2. Code Crashes (Tracebacks)
        if "Traceback (most recent call last)" in turn_content:
            tb_match = re.findall(r'^([A-Z][a-zA-Z]*Error: .+)$', turn_content, re.MULTILINE)
            if tb_match:
                errors_found.append({
                    "type": "Code Crash",
                    "raw_error": tb_match[-1],
                    "turn": turn_idx
                })

        # 3. Build Failures
        if "build_log" in turn_content and "error" in turn_content.lower():
             if "pip dependency mismatch" in turn_content:
                 errors_found.append({"type": "Build Failure", "raw_error": "Pip Dependency Mismatch", "turn": turn_idx})
             elif "executor failed running" in turn_content:
                 errors_found.append({"type": "Build Failure", "raw_error": "Docker Executor Failed", "turn": turn_idx})

    return errors_found

def simple_categorize(error_text):
    txt = error_text.lower()
    
    if "unknown tool error" in txt: return "Agent: Tool Failure (No Msg)"
    if "invalid json observation" in txt: return "Agent: JSON/Format Hallucination"

    if any(x in txt for x in ["json", "format", "parsing", "unterminated string", "expecting value"]):
        return "Agent: JSON/Format Hallucination"
    if any(x in txt for x in ["invalid tool", "unknown action", "no such tool", "entry not found"]):
        return "Agent: Tool Misuse"
    
    if any(x in txt for x in ["timeout", "timed out", "connection", "httperror", "urlerror", "incomplete", "max retries"]):
        return "Env: Network/Timeout"
    if any(x in txt for x in ["docker", "build failed", "executor failed", "returned non-zero exit status"]):
        return "Env: Docker/System Fail"
    if any(x in txt for x in ["modulenotfound", "importerror", "no module named"]):
        return "Env: Missing Dependency"

    if "keyerror" in txt: return "Data: Missing Column/Key"
    if "indexerror" in txt: return "Data: Index Out of Bounds"
    if "valueerror" in txt: return "Data: Value Error"
    if "typeerror" in txt: return "Data: Type Mismatch"
    if any(x in txt for x in ["empty", "none", "nan", "null"]): return "Data: Empty/Null Data"
    if "mergeerror" in txt: return "Data: Merge/Join Fail"
    
    if "syntaxerror" in txt: return "Code: Syntax Error"
    if "indentationerror" in txt: return "Code: Indentation Error"
    if "nameerror" in txt: return "Code: Undefined Variable"
    if "attributeerror" in txt: return "Code: Wrong Method/Attribute"
    if "assertionerror" in txt: return "Code: Assertion/Test Failed"
    if "notimplementederror" in txt: return "Code: Not Implemented"
    
    if any(x in txt for x in ["filenotfound", "no such file", "directory", "isdir", "isfile"]):
        return "System: File Not Found"
    
    if "pandas" in txt: return "Lib: Pandas Error"
    if "numpy" in txt: return "Lib: Numpy Error"
    if "matplotlib" in txt or "seaborn" in txt: return "Lib: Plotting Error"
    if "statsmodels" in txt: return "Lib: Statsmodels Error"

    return "Uncategorized Runtime"

def main():
    all_data = []
    print(f"Scanning {RESULTS_DIR}...")

    for mode in MODES:
        mode_path = os.path.join(RESULTS_DIR, mode)
        if not os.path.exists(mode_path):
            print(f"Warning: {mode_path} does not exist.")
            continue

        study_dirs = glob.glob(os.path.join(mode_path, "*"))
        
        for study_path in study_dirs:
            if not os.path.isdir(study_path): continue
            study_id = os.path.basename(study_path)

            model_dirs = glob.glob(os.path.join(study_path, "*"))
            
            for model_path in model_dirs:
                if not os.path.isdir(model_path): continue
                model_name = os.path.basename(model_path)
                
                log_dir = os.path.join(model_path, "_log")
                if not os.path.exists(log_dir):
                    continue

                log_files = glob.glob(os.path.join(log_dir, LOG_PATTERN))
                
                for log_file in log_files:
                    errors = parse_log_file(log_file)
                    for err in errors:
                        all_data.append({
                            "Mode": mode,
                            "Study": study_id,
                            "Model": model_name,
                            "Turn": err['turn'],
                            "Error_Type_Broad": err['type'],
                            "Category": simple_categorize(err['raw_error']),
                            "Raw_Error": err['raw_error']
                        })

    if not all_data:
        print("No errors found or no logs found!")
        return

    df = pd.DataFrame(all_data)
    print(f"\nTotal Errors Found: {len(df)}")
    
    output_file = "error_distribution_raw.csv"
    df.to_csv(output_file, index=False)
    print(f"Detailed log saved to {output_file}")

    print("\n--- Error Distribution by Model ---")
    pivot = pd.crosstab(df['Model'], df['Category'])
    print(pivot)
    
    pivot.to_csv("error_summary_pivot.csv")

if __name__ == "__main__":
    main()