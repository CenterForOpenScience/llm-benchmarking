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

from info_extractor.file_utils import read_file_contents, save_output, save_prompt_log
from info_extractor.prompt_builder import build_prompt, build_context_and_message
from core.constants import API_KEY, TEMPLATE_PATHS, FILE_SELECTION_RULES

client = OpenAI(api_key=API_KEY)
logger, formatter = get_logger()


def run_stage_1(study_path, difficulty, show_prompt=False):
    """
    Extract original study information and save to post_registration.json
    """
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

    if show_prompt:
        print("=== GENERATED PROMPT (Stage 1) ===")
        print(prompt)
        logger.info(f"=== GENERATED PROMPT (Stage 1) ===\n{prompt}")
        logger.info(f"\n\n=== GENERATED MESSAGE (Stage 1) ===\n{full_message}")

    save_prompt_log(study_path, "stage_1", prompt, full_message)

    assistant = client.beta.assistants.create(
        name=f"Extractor-stage-1-{difficulty}",
        instructions=prompt,
        model="gpt-4o",
        tools=[]
    )

    run = client.beta.threads.create_and_run(
        assistant_id=assistant.id,
        thread={
            "messages": [
                {
                    "role": "user",
                    "content": full_message
                }
            ]
        }
    )

    # wait until complete
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        time.sleep(2)

    messages = client.beta.threads.messages.list(thread_id=run.thread_id)
    reply = next((msg for msg in messages.data if msg.role == "assistant"), None)

    extracted_json = None
    if reply and reply.content:
        for block in reply.content:
            if isinstance(block, TextContentBlock):
                json_text = block.text.value.strip()
                break
        else:
            raise ValueError("No TextContentBlock found in assistant reply.")

		# Remove markdown-style code fences if present (e.g., ```json ... ```)																	  
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
    else:
        print("No assistant reply found for Stage 1.")

    # Save to post_registration.json
    save_output(extracted_json, study_path, stage="stage_1")
    return extracted_json  # original study data


def run_stage_2(study_path, difficulty, show_prompt=False):
    """
    Extract replication-specific information, merge with existing original study data, and save final replication_info.json
    """
    logger.info("Running Stage 2: replication study extraction")
    # Load replication_info template
    with open(TEMPLATE_PATHS['pre_registration_template']) as f:
        template = json.load(f)

    # Load instructions for stage_2 / difficulty
    with open(TEMPLATE_PATHS['info_extractor_instructions']) as f:
        instructions = json.load(f).get(difficulty, {}).get("stage_2", {})

    
    file_context, datasets_original, datasets_replication, code_file_descriptions, original_study_data = read_file_contents(
        study_path, difficulty, FILE_SELECTION_RULES, stage="stage_2"
    )

    if not file_context:
        print(f"No content was read from {study_path}")

    context_message, full_message = build_context_and_message(
        study_path, template, file_context, stage="stage_2", original_study=original_study_data
    )
    prompt = build_prompt(template, instructions, stage="stage_2")

    if show_prompt:
        print("=== GENERATED PROMPT (Stage 2) ===")
        print(prompt)
        logger.info(f"=== GENERATED PROMPT (stage 2) ===\n{prompt}")
        logger.info(f"\n\n=== GENERATED MESSAGE (stage 2) ===\n{full_message}")

    save_prompt_log(study_path, "stage_2", prompt, full_message)

    assistant = client.beta.assistants.create(
        name=f"Extractor-Stage-2-{difficulty}",
        instructions=prompt,
        model="gpt-4o",
        tools=[]
    )

    run = client.beta.threads.create_and_run(
        assistant_id=assistant.id,
        thread={
            "messages": [
                {
                    "role": "user",
                    "content": full_message
                }
            ]
        }
    )

    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        time.sleep(2)

    messages = client.beta.threads.messages.list(thread_id=run.thread_id)
    reply = next((msg for msg in messages.data if msg.role == "assistant"), None)

    extracted_json = None
    if reply and reply.content:
        for block in reply.content:
            if isinstance(block, TextContentBlock):
                json_text = block.text.value.strip()
                break
        else:
            raise ValueError("No TextContentBlock found in assistant reply.")

        if json_text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()

        try:
            replication_part = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Raw text was:", json_text)
            logger.info(f"\n\n=== RAW TEXT (Stage 2) ===\n{json_text}")
            replication_part = None
    else:
        print("No assistant reply found for Stage 2.")
        replication_part = None

    # Merge original + replication into final structure
    final_output = {}
    if isinstance(replication_part, dict):
        final_output = original_study_data
        for key, value in replication_part.items():
            if key == "original_study":
                continue
            final_output[key] = value
        
        # else:
        #     final_output = {
        #         original_study_data,
        #         "replication_study": replication_part
        #     }

    save_output(final_output, study_path, stage="stage_2")
    return final_output


def run_extraction(study_path, difficulty, stage, show_prompt=False):

    if stage == "stage_1":
        return run_stage_1(study_path, difficulty, show_prompt)
    elif stage == "stage_2":
        return run_stage_2(study_path, difficulty, show_prompt)
    else:
        raise ValueError(f"Unknown stage: {stage}")


