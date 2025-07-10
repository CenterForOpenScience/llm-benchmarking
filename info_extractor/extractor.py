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
from logger import get_logger

from info_extractor.file_utils import read_file_contents, save_output, save_prompt_log
from info_extractor.prompt_builder import build_prompt, build_context_and_message
from constants import API_KEY, TEMPLATE_PATHS, FILE_SELECTION_RULES

client = OpenAI(api_key=API_KEY)
logger = get_logger()

def run_extraction(study_path, difficulty, show_prompt=False):
    
    logger.info(f"inside run extraction")
    # Load replication_info template
    with open(TEMPLATE_PATHS['replication_info_template']) as f:
        full_template = json.load(f)

    # Load instructions for this difficulty
    with open(TEMPLATE_PATHS['info_extractor_instructions']) as f:
        instructions = json.load(f)[difficulty]


    file_context = read_file_contents(study_path, difficulty, FILE_SELECTION_RULES)
    print("FILE CONTEXT")
    print(file_context[:500])

    if not file_context:
        print(f"No content was read from {study_path}")

    context_message, full_message = build_context_and_message(study_path, full_template, file_context)
    prompt = build_prompt(full_template, instructions)

    if show_prompt:
        print("=== GENERATED PROMPT ===")
        print(prompt)
        logger.info(f"=== GENERATED PROMPT ===\n{prompt}")
        logger.info(f"\n\n=== GENERATED MESSAGE ===\n{full_message}")

    save_prompt_log(study_path, "combined", prompt, full_message)

    # return
    assistant = client.beta.assistants.create(
        name=f"Extractor-Combined-{difficulty}",
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
    reply = next(msg for msg in messages.data if msg.role == "assistant")


    if reply and reply.content:
        for block in reply.content:
            if isinstance(block, TextContentBlock):
                json_text = block.text.value.strip()
                break
        else:
            raise ValueError("No TextContentBlock found in assistant reply.")
    
        # Remove markdown-style code fences if present (e.g., ```json ... ```)
        if json_text.startswith("```"):
            # Use regex to extract content between the code fences
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
    
        try:
            extracted_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Raw text was:", json_text)
            logger.info(f"\n\n=== RAW TEXT ===\n{json_text}")
            extracted_json = None
    else:
        print("No assistant reply found.")

    # if reply and reply.content:
    #     for block in reply.content:
    #         if isinstance(block, TextContentBlock):
    #             json_text = block.text.value
    #             break
    #     else:
    #         raise ValueError("No TextContentBlock found in assistant reply.")

    #     try:
    #         extracted_json = json.loads(json_text)
    #     except json.JSONDecodeError as e:
    #         print("Failed to parse JSON:", e)
    #         print("Raw text was:", json_text)
    #         logger.info(f"\n\n=== RAW TEXT ===\n{json_text}")
    #         extracted_json = None
    # else:
    #     print("No assistant reply found.")

    save_output(extracted_json, study_path)
