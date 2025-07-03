"""
LLM_Benchmarking__
|
info_extractor--|prompt_builder.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import json
from logger import get_logger

logger = get_logger()


def build_prompt(template, instruction):
    prompt = (
        instruction.get("description", "") + "\n\n"
        + "Here is the JSON template, and its values represent descriptions of what is expected to be stored in each key:\n"
        + json.dumps(template, indent=2)
        + "\n\nPlease return only a completed JSON."
    )
    return prompt


def build_context_and_message(study_path, full_template, file_context):
    context_message = "Extract structured information for both the original study and the replication metadata."

    full_message = (
        f"{context_message}\n\n"
        "You are tasked with extracting structured information from the following text "
        "based on the given instructions.\n\n"
        "=== START OF FILE CONTENT ===\n"
        f"{file_context}\n"
        "=== END OF FILE CONTENT ==="
    )

    return context_message, full_message




