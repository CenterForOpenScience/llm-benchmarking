# generator/execute_react/agent.py

import os
import json
import re
import logging
import sys

from constants import API_KEY, GENERATE_EXECUTE_REACT_CONSTANTS

from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx
from generator.execute_react.execute_tools import (
    load_dataset, get_dataset_head, get_dataset_shape, get_dataset_description, get_dataset_info,
    run_shell_command, read_image, list_files_in_folder, ask_human_input, run_stata_do_file
)
from generator.execute_react.orchestrator_tool import (
    orchestrator_generate_dockerfile,
    orchestrator_build_image,
    orchestrator_run_container,
    orchestrator_plan,
    orchestrator_preview_entry,
    orchestrator_execute_entry,
    orchestrator_stop_container,
)

from core.agent import run_react_loop, save_output
from core.prompts import PREAMBLE, EXECUTE, EXAMPLE 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

system_prompt = "\n\n".join([PREAMBLE, EXECUTE, EXAMPLE])

# Map action names to their functions (keep name EXACTLY: known_actions) ----------
known_actions = {
    "list_files_in_folder": list_files_in_folder,
    "read_txt": read_txt,
    "read_csv": read_csv,
    "read_pdf": read_pdf,
    "read_json": read_json,
    "read_docx": read_docx,
    "read_image": read_image,

    "load_dataset": load_dataset,
    "get_dataset_head": get_dataset_head,
    "get_dataset_shape": get_dataset_shape,
    "get_dataset_description": get_dataset_description,
    "get_dataset_info": get_dataset_info,

    "ask_human_input": ask_human_input,
    "run_shell_command": run_shell_command,
    "run_stata_do_file": run_stata_do_file,

    # Orchestrator tools
    "orchestrator_generate_dockerfile": orchestrator_generate_dockerfile,
    "orchestrator_build_image": orchestrator_build_image,
    "orchestrator_run_container": orchestrator_run_container,
    "orchestrator_plan": orchestrator_plan,
    "orchestrator_preview_entry": orchestrator_preview_entry,
    "orchestrator_execute_entry": orchestrator_execute_entry,
    "orchestrator_stop_container": orchestrator_stop_container,
}

def build_file_description(available_files, file_path):
    return "".join(
        f"{i}. {os.path.join(file_path, name)}: {desc}\n"
        for i, (name, desc) in enumerate(available_files.items(), start=1)
    )

def _configure_file_logging(study_path: str):
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    log_file_full_path = os.path.join(study_path, 'agent_execute.log')
    os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_full_path, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"File logging configured to: '{log_file_full_path}'.")

def run_execute_with_human_confirm(study_path: str, show_prompt: bool = False, templates_dir: str = "./templates"):
    _configure_file_logging(study_path)
    logger.info(f"[agent] stepwise orchestrator run WITH confirmation for: {study_path}")

    schema_path = os.path.join(templates_dir, "execute_schema.json")
    prev_files = GENERATE_EXECUTE_REACT_CONSTANTS.get("files", {}).copy()
    prev_template = GENERATE_EXECUTE_REACT_CONSTANTS.get("json_template")

    try:
        GENERATE_EXECUTE_REACT_CONSTANTS["files"] = {
            "replication_info.json": "Spec with docker base image, packages, volumes, and declared code files.",
            "execution_result.json": "Raw execution output from the run (plan, steps, stdout/stderr, artifacts)."
        }
        GENERATE_EXECUTE_REACT_CONSTANTS["json_template"] = schema_path

        instruction = f"""
Follow these Actions IN ORDER. You MUST preview and get human approval before executing:

1) Action: orchestrator_generate_dockerfile: "{study_path}"
2) Action: orchestrator_build_image: "{study_path}"
3) Action: orchestrator_run_container: "{{\\"study_path\\": \\"{study_path}\\", \\"mem_limit\\": null, \\"cpus\\": null, \\"read_only\\": false, \\"network_disabled\\": false}}"
4) Action: orchestrator_plan: "{study_path}"
5) Action: orchestrator_preview_entry: "{study_path}"

You will receive a JSON that includes "command_pretty" (the exact command).
Now ask the human to approve:

6) Action: ask_human_input: "About to execute inside the container: {{command_pretty}}. Approve to execute? (yes/no)"

If and only if the human answers exactly "yes" (case-insensitive), continue:

7) Action: orchestrator_execute_entry: "{study_path}"

Then stop the container:

8) Action: orchestrator_stop_container: "{study_path}"

After step 7, {os.path.join(study_path, "execution_result.json")} will exist.
Use that (stdout/stderr, artifacts) to fill this schema:
{json.dumps(read_json(schema_path))}

FINAL OUTPUT FORMAT:
Answer: {{...}}   # a single JSON object conforming to the schema

If the human does NOT approve, still stop the container (step 8) and then output the schema with:
- execution_summary explaining that execution was cancelled by the human,
- code_executed showing the planned command with status "Not executed (cancelled)",
- results as empty/NA where appropriate.
""".strip()

        if show_prompt:
            logger.info("\n\n===== Agent Input (truncated) =====\n" + instruction[:2000])

        return run_react_loop(
            system_prompt,
            known_actions,
            instruction,
            session_state={"analyzers": {}},
            on_final=lambda ans: save_output(
                ans,
                study_path=study_path,
                filename="execution_results.json",
                stage_name="execute"
            )
        )
    finally:
        GENERATE_EXECUTE_REACT_CONSTANTS["files"] = prev_files
        if prev_template:
            GENERATE_EXECUTE_REACT_CONSTANTS["json_template"] = prev_template

def run_execute(study_path, show_prompt=False):
    _configure_file_logging(study_path)
    logger.info(f"Starting extraction for study path: {study_path}")
    template = read_json(GENERATE_EXECUTE_REACT_CONSTANTS['json_template'])

    question = f"""Question: You will have access to the following documents:
{build_file_description(GENERATE_EXECUTE_REACT_CONSTANTS['files'], study_path)}

Based on the provided documents, your goal is to execute the replication study.

Assume that you are working in an environment that supports Python and Stata.
You can use the available tools and documents given to you for the execution.
Once finished, inspect any outputs/logs and fill out this structured report:
{json.dumps(template)}
""".strip()

    return run_react_loop(
        system_prompt,
        known_actions,
        question,
        session_state={"analyzers": {}},
        on_final=lambda ans: save_output(
            ans,
            study_path=study_path,
            filename="execution_results.json",
            stage_name="execute"
        )
    )
