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
import pyreadr
import io
import re
import docx
from pathlib import Path
from core.utils import get_logger
from core.tools import read_and_summarize_pdf

logger, formatter = get_logger()

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(file_path):
    try:
        with pymupdf.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"[PDF read error: {e}]"
    
def read_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        # Extract text from each paragraph and join with a newline
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        # This can catch errors from corrupted or non-standard docx files
        return f"An error occurred while reading the docx file: {e}"


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
    ".pdf": read_and_summarize_pdf,
    ".json": read_json,
    ".csv": read_csv,
}
    
    
def summarize_dataset(file_path):
    ext = file_path.suffix.lower()
    
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame.from_dict(data)
            else:
                return {"columns": None, "info": "[JSON parsing error: Unsupported structure]", "describe": None}
        elif ext == ".rdata":
            result = pyreadr.read_r(file_path)
            if result:
                df = next(iter(result.values()))
            else:
                return {"columns": None, "info": "[RData parsing error: No data frames found]", "describe": None}
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return {"columns": None, "info": f"[Unsupported format: {ext}]", "describe": None}
        
        # Capture df.info()
        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()
        
        # Try describe
        try:
            describe = df.describe(include='all', datetime_is_numeric=True).to_string()
        except Exception as e:
            describe = f"[describe() failed: {e}]"

        return {
            "columns": df.columns.tolist(),
            "info": info,
            "describe": describe
        }

    except Exception as e:
        return {
            "columns": None,
            "info": f"[Data summary error: {e}]",
            "describe": None
        }



def read_file_contents(folder, difficulty, selection_rules, stage="stage_1"):
    folder = Path(folder)

    tier_rules = selection_rules['info_extractor'].get(difficulty, {})
    stage_rules = tier_rules.get(stage, {})
    allowed_files = stage_rules.get("files", [])
    allowed_folders = stage_rules.get("folders", {})

    aggregated_content = []
    code_section = ["\n=== CODE RELATED FILES ==="]
    dataset_section = ["\n=== DATASET FILES ==="]
    datasets_original = []
    datasets_replication = []
    code_file_descriptions = {}

    for file in folder.iterdir():

        if file.is_dir():
            # CODEBASE folder (only include if allowed via folder rules or conventionally named)
            if (file.name.lower() == "code") and ("code" in allowed_folders or not allowed_folders):
                for code_file in file.glob("*.*"):
									
                    try:
                        with open(code_file, "r", encoding="utf-8", errors="ignore") as f:
                            code_content = f.read(3000)
                        code_section.append(f"\n---\n**{code_file.name}**\n{code_content}")
                        code_file_descriptions[code_file.name] = code_content
                    except Exception as e:
                        code_section.append(f"\n---\n**{code_file.name}**\n[Error reading file: {e}]")
                        logger.warning(f"\n---\n**{code_file.name}**\n[Error reading file: {e}]")

            # DATASET folder
            if file.name.lower() in ["dataset_folder", "datasets", "data"] and \
               ("data" in allowed_folders or not allowed_folders):
                for data_file in file.glob("*.*"):
                    summary = summarize_dataset(data_file)
                    head = ""
                    try:
                        if data_file.suffix == ".csv":
                            df = pd.read_csv(data_file)
                            head = df.head().to_string()
                        elif data_file.suffix == ".xlsx":
                            df = pd.read_excel(data_file)
                            head = df.head().to_string()
                        elif data_file.suffix == ".json":
                            with open(data_file, "r") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                                head = df.head().to_string()
                            else:
                                head = json.dumps(data, indent=2)[:1000]
                        elif data_file.suffix == ".rdata":
                            result = pyreadr.read_r(data_file)
                            if result:
                                df = next(iter(result.values()))
                                head = df.head().to_string()
                    except Exception as e:
                        head = f"[Error reading dataset head: {e}]"
																			   
                        logger.exception(f"[Error reading dataset head: {e}]")

																		
                    dataset_section.append(f"\n---\n**{data_file.name}**\n"
                                           f"=== HEAD ===\n{head}\n"
                                           f"=== INFO ===\n{summary.get('info')}\n"
                                           f"=== DESCRIBE ===\n{summary.get('describe')}")

                    dataset_obj = {
                        "name": data_file.stem,
                        "filename": data_file.name,
                        "type": "original" if "replication" not in data_file.name.lower() else "replication",
                        "file_format": data_file.suffix[1:],
                        "columns": summary.get("columns"),
                        "summary_statistics": {
                            "info": summary.get("info"),
                            "describe": summary.get("describe")
                        },
                        "access": {
                            "url": None,
                            "restrictions": None
                        },
                        "notes": None
                    }

                    if dataset_obj["type"] == "replication":
                        datasets_replication.append(dataset_obj)
                    else:
                        datasets_original.append(dataset_obj)

        # non-directory core files
        if file.name not in allowed_files:
            continue
        																				 
        elif file.suffix in FILE_READERS:
            try:
                reader = FILE_READERS[file.suffix]
                content = reader(file)
                aggregated_content.append(f"\n---\n**{file.name}**\n{content}")
            except Exception as e:
                aggregated_content.append(f"\n---\n**{file.name}**\n[Error reading file: {e}]")
                logger.exception(f"\n---\n**{file.name}**\n[Error reading file: {e}]")
        
    original_study_data = None
    if stage == 'stage_2':                
        # Load existing original study output
        post_reg_path = os.path.join(folder, "post_registration.json")
        
        
        if os.path.exists(post_reg_path):
            try:
                with open(post_reg_path, "r", encoding="utf-8") as f:
                    original_study_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read existing post_registration.json: {e}")
        else:
            raise ValueError("post_registration.json not found")

													   
    file_context = "\n".join(aggregated_content + code_section + dataset_section)

    return file_context, datasets_original, datasets_replication, code_file_descriptions, original_study_data 

    

def save_output(extracted_json, study_path, stage):
    if stage == "stage_1":
        output_filename = "post_registration.json"
    elif stage == "stage_2":
        output_filename = "replication_info.json"


    output_path = os.path.join(study_path, output_filename)
    try:
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(extracted_json, f, indent=2)
        print(f"[INFO] Stage '{stage}' output saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save output for stage {stage}: {e}")
        logger.exception(f"Failed to save output for stage {stage}: {e}")
    

def save_prompt_log(study_path, stage, prompt, full_message):
									   
    case_name = os.path.basename(os.path.normpath(study_path))
    
    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", study_path)
        if match:
            case_name = match.group()

					   
    log_dir = "logs"
												
    os.makedirs(log_dir, exist_ok=True)

				   
    log_file = os.path.join(log_dir, f"{case_name}_{stage}_log.txt")

				  
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")
        f.write("=== GENERATED FULL MESSAGE ===\n")
        f.write(full_message + "\n")

    print(f"[INFO] Prompt and message logged to {log_file}")


