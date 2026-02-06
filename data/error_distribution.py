import os
import re
import json
import pandas as pd
import glob

RESULTS_DIR = "./results"
MODES = ["native", "python"]
LOG_PATTERN = "execute*.log" 

def parse_log_file(file_path):
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

        # SIGNAL 1: JSON
        obs_match = re.search(r'\*\*\*Agent input: Observation: ({.*})', turn_content, re.DOTALL)
        if obs_match:
            try:
                json_str = obs_match.group(1).split('\n202')[0]
                data = json.loads(json_str)
                if "steps" in data and isinstance(data["steps"], list):
                    for step in data["steps"]:
                        if step.get("ok") is False:
                            raw_err = step.get("stderr") or step.get("error") or json.dumps(step)
                            errors_found.append({"type": "Tool Failure (Nested)", "raw_error": str(raw_err), "turn": turn_idx})
                elif data.get("ok") is False:
                    err_msg = (data.get("error") or data.get("message") or data.get("content") or data.get("failure") or json.dumps(data))
                    errors_found.append({"type": "Tool Failure (JSON)", "raw_error": str(err_msg)[:500], "turn": turn_idx})
            except: pass 

        # SIGNAL 2: Text
        error_lines = re.findall(r'^(Error(?: executing [\w_]+)?: .+)$', turn_content, re.MULTILINE)
        for err_line in error_lines:
            errors_found.append({"type": "Tool Failure (Text)", "raw_error": err_line.strip(), "turn": turn_idx})
        
        entry_lines = re.findall(r'^(Entry not found.+)$', turn_content, re.MULTILINE)
        for entry_line in entry_lines:
             errors_found.append({"type": "Tool Failure (Text)", "raw_error": entry_line.strip(), "turn": turn_idx})

        # SIGNAL 3: Traceback
        if "Traceback (most recent call last)" in turn_content:
            tb_match = re.findall(r'^([A-Z][a-zA-Z]*Error: .+)$', turn_content, re.MULTILINE)
            if tb_match:
                errors_found.append({"type": "Code Crash", "raw_error": tb_match[-1], "turn": turn_idx})

        # SIGNAL 4: Docker
        if "build_log" in turn_content and "error" in turn_content.lower():
             if "pip dependency mismatch" in turn_content:
                 errors_found.append({"type": "Build Failure", "raw_error": "Pip Dependency Mismatch", "turn": turn_idx})
             elif "executor failed running" in turn_content:
                 errors_found.append({"type": "Build Failure", "raw_error": "Docker Executor Failed", "turn": turn_idx})

    return errors_found

def simple_categorize(error_text):
    txt = error_text.lower()
    
    # CODE EDITING & AGENT
    if "old_text not found" in txt: return "Agent: Code Edit Failed"
    if "end_marker not found" in txt: return "Agent: Code Edit Failed"
    if "requires anchor" in txt: return "Agent: Tool Misuse"
    if "requires insert_text" in txt: return "Agent: Tool Misuse"
    if "requires old_text" in txt: return "Agent: Tool Misuse"
    if "unexpected keyword argument" in txt: return "Agent: Tool Misuse"
    if "missing required argument" in txt: return "Agent: Tool Misuse"
    if "missing 1 required positional argument" in txt: return "Agent: Tool Misuse"
    if "unknown tool error" in txt: return "Agent: Tool Failure (No Msg)"
    if "invalid json observation" in txt: return "Agent: JSON/Format Hallucination"

    # PERMISSIONS / SYSTEM
    if "access denied" in txt or "outside of the study directory" in txt: return "System: Permission/Security Error"
    if "entry not found" in txt: return "System: File Not Found"
    if "filenotfound" in txt or "no such file" in txt or "does not exist" in txt: return "System: File Not Found"
    if "cp: " in txt and "cannot stat" in txt: return "System: File Not Found"

    # ENVIRONMENT / NETWORK
    if "could not find function" in txt: return "Code: Undefined Variable (R)" # Borderline, but often missing lib
    if "there is no package called" in txt: return "Env: Missing Dependency"
    if "execution halted" in txt: return "Env: R Runtime Error"
    if "installing packages into" in txt: return "Env: R Package Install Log"
    if "oci runtime exec failed" in txt: return "Env: Docker/System Fail"
    if any(x in txt for x in ["timeout", "timed out", "connection", "httperror", "urlerror", "incomplete", "max retries"]):
        return "Env: Network/Timeout"
    if any(x in txt for x in ["docker", "build failed", "executor failed", "returned non-zero exit status"]):
        return "Env: Docker/System Fail"
    if any(x in txt for x in ["modulenotfound", "importerror", "no module named"]):
        return "Env: Missing Dependency"

    # DATA & TYPES
    if "cannot interpret" in txt and "dtype" in txt: return "Lib: Pandas/Numpy Error"
    if "not supported between instances" in txt: return "Data: Type Mismatch"
    if "keyerror" in txt: return "Data: Missing Column/Key"
    if "indexerror" in txt: return "Data: Index Out of Bounds"
    if "valueerror" in txt: return "Data: Value Error"
    if "typeerror" in txt: return "Data: Type Mismatch"
    if any(x in txt for x in ["empty", "none", "nan", "null"]): return "Data: Empty/Null Data"
    if "dataset not loaded" in txt: return "Data: State/Context Error"
    
    # CODE LOGIC
    if "syntaxerror" in txt: return "Code: Syntax Error"
    if "indentationerror" in txt: return "Code: Indentation Error"
    if "nameerror" in txt: return "Code: Undefined Variable"
    if "attributeerror" in txt: return "Code: Wrong Method/Attribute"
    if "valuewarning" in txt: return "Code: Warning (Non-Fatal)"

    return "Uncategorized Runtime"

def map_to_super_category(granular_cat):
    """Maps granular categories to your 5 specific definitions."""
    
    # SETUP ERRORS
    # "Run cannot be started due to environment/dependency/missing files"
    if any(x in granular_cat for x in [
        "System: File Not Found",
        "System: Permission/Security Error",
        "Env: Docker/System Fail",
        "Env: Missing Dependency",
        "Env: R Runtime Error", 
        "Env: R Package Install Log",
        "Build Failure"
    ]):
        return "Setup Errors"

    # INPUT DATA ERRORS
    # "Dataset cannot be loaded, corrupted, missing variables, type mismatch"
    if any(x in granular_cat for x in [
        "Data: Empty/Null Data",
        "Data: Index Out of Bounds",
        "Data: Missing Column/Key",
        "Data: Type Mismatch",
        "Data: Value Error",
        "Data: State/Context Error",
        "Lib: Pandas/Numpy Error"
    ]):
        return "Input Data Errors"

    # TIMEOUT ERRORS
    # "Run does not finish"
    if "Network/Timeout" in granular_cat:
        return "Timeout Errors"

    # RESULT EXTRACTION ERRORS
    # "Outputs missing" - Hard to verify in runtime logs, but if explicit:
    if "Output Missing" in granular_cat: 
        return "Result Extraction Errors"

    # IMPLEMENTATION ERRORS
    # "Run cannot carry out procedure (logic, syntax, tool misuse)"
    # This acts as the primary bucket for logic/code issues.
    return "Implementation Errors"

def main():
    all_data = []
    print(f"Scanning {RESULTS_DIR}...")

    # [Standard File Walking Loop]
    for mode in MODES:
        mode_path = os.path.join(RESULTS_DIR, mode)
        if not os.path.exists(mode_path): continue
        for study_path in glob.glob(os.path.join(mode_path, "*")):
            if not os.path.isdir(study_path): continue
            study_id = os.path.basename(study_path)
            for model_path in glob.glob(os.path.join(study_path, "*")):
                if not os.path.isdir(model_path): continue
                model_name = os.path.basename(model_path)
                log_dir = os.path.join(model_path, "_log")
                if not os.path.exists(log_dir): continue
                log_files = glob.glob(os.path.join(log_dir, LOG_PATTERN))
                for log_file in log_files:
                    errors = parse_log_file(log_file)
                    for err in errors:
                        granular = simple_categorize(err['raw_error'])
                        super_cat = map_to_super_category(granular)
                        if model_name == "gpt-5_nocode":
                        	continue
                        all_data.append({
                            "Model": model_name,
                            "Granular": granular,
                            "Category": super_cat
                        })

    if not all_data:
        print("No errors found.")
        return

    df = pd.DataFrame(all_data)
    
    # Pivot for the final table
    pivot = pd.crosstab(df['Model'], df['Category'])
    
    # Ensure all columns exist
    cols = ["Setup Errors", "Input Data Errors", "Implementation Errors", "Result Extraction Errors", "Timeout Errors"]
    for c in cols:
        if c not in pivot.columns: pivot[c] = 0
    pivot = pivot[cols]
    
    print("\n" + "="*50)
    print("FINAL ERROR DISTRIBUTION TABLE")
    print("="*50)
    print(pivot)
    pivot.to_csv("final_error_table.csv")

if __name__ == "__main__":
    main()