"""
R005_LLM_Benchmarking__
|
info_extractor--|prompt_builder.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import os
import json

def build_prompt(template, instruction):
    prompt = (
        instruction.get("description", "") + "\n\n"
        + "Here is the JSON template, and its values represent descriptions of what is expected to be stored in each key:\n"
        + json.dumps(template, indent=2)
        + "\n\nPlease return only a completed JSON."
    )
    return prompt


def build_context_and_message(stage, study_path, full_template, file_context):
    if stage == '1':
        template = full_template['original_study']
        context_message = "Extract stage 1 (original study) information."
        stage1_data = ""
    else:
        stage1_output_path = os.path.join(study_path, "replication_info_stage1.json")
        if not os.path.exists(stage1_output_path):
            raise FileNotFoundError("Stage 1 output not found. Please run Stage 1 first.")
        with open(stage1_output_path) as f:
            stage1_data = json.load(f)

        template = {k: v for k, v in full_template.items() if k != 'original_study'}
        context_message = (
            "Below are the extracted information from stage 1:\n"
            f"{json.dumps(stage1_data, indent=2)}"
        )

    full_message = (
        f"{context_message}\n\n"
        "You are tasked with extracting structured information from the following text "
        "based on the given instructions.\n\n"
        "=== START OF FILE CONTENT ===\n"
        f"{file_context}\n"
        "=== END OF FILE CONTENT ==="
    )

    return template, context_message, full_message, stage1_data
