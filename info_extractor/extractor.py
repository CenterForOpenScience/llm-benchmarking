"""
R005_LLM_Benchmarking__
|
info_extractor--|extractor.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI
from openai.types.beta.threads import TextContentBlock

from info_extractor.file_utils import read_file_contents, save_output, save_prompt_log
from info_extractor.prompt_builder import build_prompt, build_context_and_message
from R005_constants import API_KEY, TEMPLATE_PATHS, FILE_SELECTION_RULES

client = OpenAI(api_key=API_KEY)

def run_extraction(study_path, stage, difficulty, show_prompt=False):
    # Load main template
    with open(TEMPLATE_PATHS['replication_info_template']) as f:
        full_template = json.load(f)

    # Load instructions for this stage & difficulty
    instruction_file = TEMPLATE_PATHS[f'stage{stage}_instructions']
    with open(instruction_file) as f:
        all_instructions = json.load(f)
    instruction = all_instructions[difficulty]

    # Load file contents
    file_context = read_file_contents(study_path, stage, difficulty, FILE_SELECTION_RULES)
    print("FILE CONTEXT")
    print(file_context[:500])
    
    if not file_context:
        print(f"No content was read from {study_path}")
    
    # Get full message and final message using the builder
    template, context_message, full_message, stage1_data = build_context_and_message(stage, study_path, full_template, file_context)
    
    # Generate prompt text
    prompt = build_prompt(template, instruction)

    if show_prompt:
        print("=== GENERATED PROMPT ===")
        print(prompt)
        print("========================")
        # print("=== GENERATED FULL MESSAGE ===")
        # print(full_message)
        # print("========================")

        # Log prompts to a file
        save_prompt_log(study_path, stage, prompt, full_message)
        
    # Create assistant 
    assistant = client.beta.assistants.create(
        name=f"Extractor-Stage{stage}-{difficulty}",
        instructions=prompt, 
        model="gpt-4o",
        tools=[]
    )

    # Start the run
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

    # Poll until complete
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        time.sleep(2)

    # Get assistant reply messages
    messages = client.beta.threads.messages.list(thread_id=run.thread_id)

    # Find assistant's first reply
    reply = next(msg for msg in messages.data if msg.role == "assistant")


    # Extract the JSON content
    if reply and reply.content:
        # This will work for single block text reply
        for block in reply.content:
            if isinstance(block, TextContentBlock):
                json_text = block.text.value  
                break
        else:
            raise ValueError("No TextContentBlock found in assistant reply.")
    
        try:
            extracted_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Raw text was:", json_text)
            extracted_json = None
    else:
        print("No assistant reply found.")

    save_output(stage, extracted_json, study_path, stage1_data)
    