import os
import fitz  # PyMuPDF
import openai
import json
import re
from docx import Document
from openai import OpenAI

from constants import API_KEY, TEMPLATE_PATHS
from info_extractor.file_utils import read_json


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_report_text(path):
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.lower().endswith(".docx") or path.lower().endswith(".doc"):
        return extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported format")


def build_extract_evaluate_prompt(original_paper, preregistration, extraction_schema, extracted_json):
    return f"""
        You are an information verifier.
        You are given (1) a research paper, (2) a preregistraton document that attempts to replicate a claim in the paper and (3) a JSON object, your task is to verify whether the information (key, value pair) presented in the JSON object matches with the information presented in the preregistration and/or paper.
        A match is considered when they are logically equivalent or semantically similar statements.
        If the value of the field is null, treat it as "Information cannot be found". You have to evaluate all values in the JSON.
        
        === START OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===
        {extraction_schema}
        === END OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===
        
        
        Please return your evaluation as another JSON object:
        {{
            "matches": [
                {{
                    "key": [the key path leading to the value, e.g. key1.key2],
                    "predicted": [the information presented in the provided json],
                    "expected": [the information you extract from the provided documents],
                    "explanation": [explanation why you think the predict info matches the info in the doc.]
                    "
                }},
                ...
            ],
            "unmatches: [
                {{
                    "key": [the key path leading to the value, e.g. key1.key2],
                    "predicted": [the information presented in the provided json],
                    "expected" [the information you extract from the provided documents (not the json). If possible, use as much original wording from the original paper and preregistration.,
                    "explanation": [explanation why you think the predict info does NOT match the info in the doc.]
                    "
                }},
                ...
            ],
        }}
        
        === ORIGINAL PAPER START ===
        {original_paper}
        === ORIGINAL PAPER END ===
                
        === REPLICATION STUDY PRE-REGISTRATION DOCUMENT START ===
        {preregistration}
        === REPLICATION STUDY PRE-REGISTRATION DOCUMENT END ===
        
        === JSON TO BE EVALUATED START ===
        {extracted_json}
        === JSON TO BE EVALUATED END ===

        Output Requirements:\n- Return a valid JSON object only.\n- Do NOT wrap the output in markdown (no ```json).\n- Do NOT include extra text, commentary, or notes.\n\n Ensure accuracy and completeness.\n- Strictly use provided sources as specified.
        """

def save_prompt_log(study_path, prompt):
    case_name = os.path.basename(os.path.normpath(study_path))

    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", study_path)
        if match:
            case_name = match.group()
            
    log_dir = "logs"
    # log_dir = os.path.join(study_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{case_name}_validator_ehri_log.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")

    print(f"[INFO] Prompt logged to {log_file}")
    

def generate_expected_json(original_paper, preregistration, expected_schema, extracted_json, client, log_path):
    prompt = build_extract_evaluate_prompt(original_paper, preregistration, expected_schema, extracted_json)

    save_prompt_log(log_path, prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    content = response.choices[0].message.content
    return json.loads(content)


def save_json(data, path):
    with open(os.path.join(path, "llm_eval.json"), 'w') as f:
        json.dump(data, f, indent=2)
        print(f"extract_and_evaluate_from_human_rep.py output saved to {path}")


def extract_from_human_replication_study(original_paper_path, preregistration_path, extracted_json_path, output_path):
    client = OpenAI(api_key=API_KEY)
    
    expected_schema = read_json(TEMPLATE_PATHS['replication_info_template'])
    
    original_paper = extract_text_from_pdf(original_paper_path)
    preregistration = load_report_text(preregistration_path)
    
    extracted_json = read_json(extracted_json_path)
    
    log_path = os.path.dirname(output_path)
    expected_json = generate_expected_json(original_paper, preregistration, expected_schema, extracted_json, client, log_path)
    save_json(expected_json, output_path)