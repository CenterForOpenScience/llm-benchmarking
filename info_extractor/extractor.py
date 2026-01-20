"""
LLM_Benchmarking__
|
info_extractor--|extractor.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""
import os
import re
import json
import time
from openai import OpenAI
from openai.types.beta.threads import TextContentBlock
from core.utils import get_logger

from info_extractor.file_utils import read_file_contents, save_output, find_required_file, call_search_model_once, parse_json_strict
from info_extractor.prompt_builder import build_prompt, build_context_and_message
from core.constants import API_KEY, TEMPLATE_PATHS, FILE_SELECTION_RULES
from core.utils import configure_file_logging
from core.agent import update_metadata, messages_to_responses_input

from core.tools import read_and_summarize_pdf

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

    # metric collection
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

def run_web_search(study_path,model_name,show_prompt=False):
    """
    Reads initial_details.txt + original_paper.pdf, then calls the paired search model once
    to return URLs needed for replication. Saves found_urls.json.
    """
    start_time = time.time()
    configure_file_logging(logger, study_path, "find_urls.log")

    details_path = find_required_file(study_path, "initial_details.txt")
    paper_path = find_required_file(study_path, "original_paper.pdf")

    # read claim
    with open(details_path, "r", encoding="utf-8", errors="ignore") as f:
        claim_text = f.read()
    
    paper_text = read_and_summarize_pdf(paper_path, summarizer_model=model_name, for_data=True)

    if model_name.startswith("gpt-5"):
        search_model = "gpt-5-search-api"
    elif model_name.startswith("o3"):
    	search_model = "o3-deep-research"
    else:
    	search_model = "gpt-4o-search-preview"
    print(f"[web-search] summarizer_model={model_name} -> search_model={search_model}")

    try:
    	raw = call_search_model_once(search_model, claim_text, paper_text)
    except Exception as e:
    	print(f"search model call failed: {search_model}")
    	
    parsed = parse_json_strict(raw)

    duration = time.time() - start_time

    # Save output
    out_obj = {
        "summarizer_model": model_name,
        "search_model": search_model,
        "details_path": details_path,
        "paper_path": paper_path,
        "result": parsed,
        "raw_response": raw,
    }

    out_path = os.path.join(study_path, f"found_urls_{search_model}.json")
    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    # Metadata
    metric_data = {
        "total_time_seconds": round(duration, 2),
        "total_turns": 1,
        # token usage maybe
    }
    update_metadata(study_path, "extract_stage_find_urls", metric_data)

    return out_obj



def run_extraction(study_path, difficulty, stage, model_name, show_prompt=False):

    if stage == "stage_1":
        return run_stage_1(study_path, difficulty, show_prompt, model_name)
    if stage == "web_search":
    	return run_web_search(study_path,model_name, show_prompt)
    else:
        raise ValueError(f"Unknown stage: {stage}")


