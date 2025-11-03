# interpreter/agent.py

import os
import json
import re
import logging
import sys
import tiktoken
import copy

from constants import API_KEY, INTERPRET_CONSTANTS
from openai import OpenAI

from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx
from generator.execute_react.execute_tools import (
    load_dataset, get_dataset_head, get_dataset_shape, get_dataset_description, get_dataset_info
)
from generator.execute_react.execute_tools import (
    read_image, list_files_in_folder, ask_human_input
)

from core.agent import run_react_loop, save_output
from core.prompts import PREAMBLE, INTERPRET, EXAMPLE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

client = OpenAI(api_key=API_KEY)

MAX_TOKENS = 20000

def _count_tokens(text: str, model_name="gpt-4o"):
    enc = tiktoken.encoding_for_model(model_name if model_name else "gpt-4")
    return len(enc.encode(text))

def read_log(file_path: str, model_name: str = "gpt-4o"):
    """
    Tool: read a potentially very long log. If too big, chunk and summarize progressively.
    Returns a string (full text or summarized).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
    except Exception as e:
        return f"[read_log error] {e}"

    if _count_tokens(full_log, model_name=model_name) <= MAX_TOKENS:
        return full_log

    # Chunk + summarize
    lines = full_log.splitlines(keepends=True)
    chunk_size = 800  # lines per chunk (tweakable)
    chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

    sys_prompt = (
        "You are an effective log reader. Summarize the provided log chunk. "
        "Focus on errors, exceptions, warnings, commands executed, and results."
    )

    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"CHUNK {idx}/{len(chunks)}:\n{chunk}"}
        ]
        try:
            out = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=messages
            )
            summaries.append(out.choices[0].message.content)
        except Exception as e:
            summaries.append(f"[summarization error on chunk {idx}] {e}")

    # One final synthesis pass (bounded)
    final_messages = [
        {"role": "system", "content": "Synthesize the chunk summaries into a concise but detailed overall summary."},
        {"role": "user", "content": "\n\n".join(summaries)}
    ]
    try:
        final = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=final_messages
        ).choices[0].message.content
    except Exception as e:
        final = "\n\n".join(summaries) + f"\n[final synthesis error] {e}"
    return final

system_prompt = "\n\n".join([PREAMBLE, INTERPRET, EXAMPLE])

# Map action names to their functions (KEEP NAME: known_actions) --------
known_actions = {
    "list_files_in_folder": list_files_in_folder,
    "read_txt": read_txt,
    "read_csv": read_csv,
    "read_pdf": read_pdf,
    "read_json": read_json,
    "read_docx": read_docx,
    "read_log": read_log,  # ← special interpret tool
    "read_image": read_image,

    "load_dataset": load_dataset,
    "get_dataset_head": get_dataset_head,
    "get_dataset_shape": get_dataset_shape,
    "get_dataset_description": get_dataset_description,
    "get_dataset_info": get_dataset_info,

    "ask_human_input": ask_human_input,
}

def build_file_description(available_files, file_path):
    return "".join(
        f"{i}. {os.path.join(file_path, name)}: {desc}\n"
        for i, (name, desc) in enumerate(available_files.items(), start=1)
    )

def _configure_file_logging(study_path: str):
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    log_file_full_path = os.path.join(study_path, 'interpret.log')
    os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True)
    fh = logging.FileHandler(log_file_full_path, mode='a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f"File logging configured to: '{log_file_full_path}'.")

def run_interpret(study_path, show_prompt=False):
    _configure_file_logging(study_path)
    logger.info(f"Starting execution evaluation for study path: {study_path}")

    eval_prompt_template = read_txt(INTERPRET_CONSTANTS['prompt_template'])
    json_schema = read_json(INTERPRET_CONSTANTS['json_template'])
    claim_docs_for_evaluator = build_file_description(INTERPRET_CONSTANTS['claim_files'], study_path)
    agent_docs_for_evaluator = build_file_description(INTERPRET_CONSTANTS['agent_files'], study_path)

    variables = {
        'interpret_json_schema': json_schema,
        'claim_docs_for_evaluator': claim_docs_for_evaluator,
        'agent_docs_for_evaluator': agent_docs_for_evaluator,
    }

    question = "Question: " + eval_prompt_template.format(**variables)

    return run_react_loop(
        system_prompt,
        known_actions,
        question,
        session_state={"analyzers": {}},
        on_final=lambda ans: save_output(
            ans,
            study_path=study_path,
            filename="interpret_results.json",
            stage_name="interpret"
        )
    )
