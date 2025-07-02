"""
LLM_Benchmarking__
|
info_extractor--|file_utils.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import os
import pymupdf
import json
import pandas as pd
from pathlib import Path


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(file_path):
    try:
        with pymupdf.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"[PDF read error: {e}]"


def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"[JSON read error: {e}]"


def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        return f"[CSV read error: {e}]"

FILE_READERS = {
    ".txt": read_txt,
    ".pdf": read_pdf,
    ".json": read_json,
    ".csv": read_csv,
}


def read_file_contents(folder, difficulty, selection_rules):
    folder = Path(folder)

    allowed_files = selection_rules['info_extractor'][difficulty]

    aggregated_content = []

    for file in folder.iterdir():
        if file.name in allowed_files and file.suffix in FILE_READERS:
            try:
                reader = FILE_READERS[file.suffix]
                content = reader(file)
                aggregated_content.append(f"\n---\n **{file.name}**\n{content}")
            except Exception as e:
                print(f"Skipping {file.name} due to reader error: {e}")

    if not aggregated_content:
        print(f"No matching readable files for difficulty '{difficulty}'")

    return "\n".join(aggregated_content)    
    

def save_output(extracted_json, study_path):
    output_path = os.path.join(study_path, "replication_info.json")
    with open(output_path, 'w') as f:
        json.dump(extracted_json, f, indent=2)

    print(f"[INFO] Combined output saved to {output_path}")


def save_prompt_log(study_path, stage, prompt, full_message):
    # Extract case study name from path
    case_name = os.path.basename(os.path.normpath(study_path))

    # Create log folder
    log_dir = "logs"
    # log_dir = os.path.join(study_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log file name
    log_file = os.path.join(log_dir, f"{case_name}_stage{stage}_log.txt")

    # Save content
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")
        f.write("=== GENERATED FULL MESSAGE ===\n")
        f.write(full_message + "\n")

    print(f"[INFO] Prompt and message logged to {log_file}")

