"""
LLM_Benchmarking__
|
info_extractor--|extractor.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import json
import time
import re
from openai import OpenAI
from openai.types.beta.threads import TextContentBlock
from core.utils import get_logger

from info_extractor.file_utils import read_file_contents, save_output
from info_extractor.prompt_builder import build_prompt, build_context_and_message
from core.constants import API_KEY, TEMPLATE_PATHS, FILE_SELECTION_RULES
from core.utils import configure_file_logging
from core.agent import update_metadata, messages_to_responses_input

client = OpenAI(api_key=API_KEY)
logger, formatter = get_logger()

def is_reasoning_model(model: str) -> bool:
    return model.startswith(("o1", "o3", "gpt-5"))

def run_stage_1(study_path, difficulty, show_prompt=False, model_name: str="gpt-4o"):
    """
    Extract original study information and save to post_registration.json
    """
    start_time = time.time()
    configure_file_logging(logger, study_path, f"extract.log")
    print(f"\n\nmodel name for extractor stage: {model_name}\n\n")

    logger.info("Running Stage 1: original study extraction")
    # Load post-registration template
    with open(TEMPLATE_PATHS['post_registration_template']) as f:
        template = json.load(f)

    # Load instructions for stage_1 / difficulty
    with open(TEMPLATE_PATHS['info_extractor_instructions']) as f:
        instructions = json.load(f).get(difficulty, {}).get("stage_1", {})

    file_context, datasets_original, datasets_replication, code_file_descriptions, original_study_data = read_file_contents(
        study_path, difficulty, FILE_SELECTION_RULES, stage="stage_1"
    )

    if not file_context:
        print(f"No content was read from {study_path}")

    context_message, full_message = build_context_and_message(
        study_path, template, file_context, stage="stage_1"
    )
    prompt = build_prompt(template, instructions, stage="stage_1")

    print("=== GENERATED PROMPT (Stage 1) ===")
    logger.info(f"=== GENERATED PROMPT (Stage 1) ===\n{prompt}")
    logger.info(f"\n\n=== GENERATED MESSAGE (Stage 1) ===\n{full_message}")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": full_message},
    ]

    response = client.responses.create(
        model=model_name,
        input=messages_to_responses_input(messages),
    )

    duration = time.time() - start_time
    usage = response.usage
    json_text = response.output_text.strip()

    # metric collection (unchanged logic, model-aware fields)
    metric_data = {
        "total_time_seconds": round(duration, 2),
        "total_tokens": usage.total_tokens if usage else 0,
        "prompt_tokens": (
            usage.input_tokens if is_reasoning_model(model_name) else usage.prompt_tokens
        ) if usage else 0,
        "completion_tokens": (
            usage.output_tokens if is_reasoning_model(model_name) else usage.completion_tokens
        ) if usage else 0,
        "total_turns": 1
    }
    update_metadata(study_path, "extract_stage_1", metric_data)

    extracted_json = None

    # Remove markdown-style code fences if present
    if json_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()

    try:
        extracted_json = json.loads(json_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw text was:", json_text)
        logger.info(f"\n\n=== RAW TEXT (Stage 1) ===\n{json_text}")
        extracted_json = None

    save_output(extracted_json, study_path, stage="stage_1")
    return extracted_json


def run_extraction(study_path, difficulty, stage, show_prompt=False, model_name:str="gpt-4o"):

    if stage == "stage_1":
        return run_stage_1(study_path, difficulty, show_prompt, model_name)
    else:
        raise ValueError(f"Unknown stage: {stage}")


