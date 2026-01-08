# generator/design-react.py
import os
import json
import re
from openai import OpenAI
from core.constants import API_KEY, GENERATE_REACT_CONSTANTS
import sys

from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx # Keep save_output here if the agent orchestrates saving
from core.tools import load_dataset, get_dataset_head, get_dataset_shape, get_dataset_description, get_dataset_info, read_image, list_files_in_folder, ask_human_input, write_file
from core.prompts import PREAMBLE, DESIGN, EXAMPLE, DESIGN_CODE_MODE_POLICY
from core.agent import Agent, run_react_loop, save_output
from core.utils import build_file_description, configure_file_logging, get_logger
from core.actions import base_known_actions

logger, formatter = get_logger()
action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE) # Use re.MULTILINE for multiline parsing
known_actions = base_known_actions()

def build_system_prompt(code_mode: str) -> str:
    code_policy = DESIGN_CODE_MODE_POLICY.get(code_mode, DESIGN_CODE_MODE_POLICY["python"])
    # Put the policy in SYSTEM prompt
    return "\n\n".join([PREAMBLE, DESIGN, code_policy, EXAMPLE])

def run_design(study_path, show_prompt: bool = False, tier: str = "easy", code_mode: str = "python"):
    configure_file_logging(logger, study_path, f"design_{tier}__{code_mode}.log")
    # Load json template
    logger.info(f"Starting extraction for study path: {study_path}")
    template =  read_json(GENERATE_REACT_CONSTANTS[f'json_template_{code_mode}'])
    code_policy = DESIGN_CODE_MODE_POLICY.get(code_mode, DESIGN_CODE_MODE_POLICY["python"])
    
    system_prompt = build_system_prompt(code_mode)
    
    question = f"""
    You will have access to the following documents:
    {build_file_description(GENERATE_REACT_CONSTANTS['files'], study_path)}
    
    Based on the provided documents, your goal is to plan for the replication study and fill out this JSON template:
    {json.dumps(template)}
    
    First, determine whether the provided data can be used for replicating the provided focal claim. 
    - Ensure that all necessary variables are available.
    - Ensure that the data qualify for replication criteria. Replication data achieves its purpose by being different data collected under similar/identical conditions, thus testing if the phenomenon is robust across independent instances.
    
    If you find issues with the provided data, follow-up with a human supervisor to ask for a different data source until appropriate data is given.
    
    Once you have determined the provided data are good for replication, explore the code to help fill out fields related to the codebase. This code will operate directly on the data files given to you.
    Find potential issues with the provided code such as a data file path that is different from the data files you have looked at.
    - If the code reads any data file, the file path must be in this directory "/app/data".
    - If the code dumps content or produce additional content, the file must also be in this directory "/app/data".
    
    {code_policy}

    If your code reads in any data file, ASSUME that the data will be in this directory: "/app/data".
    If you code produce any addtional files, the code must save the files in this directory: "/app/data".

    File operations policy:
    - To modify existing files: ALWAYS call read_file first, then use edit_file for targeted changes.
    - write_file is for creating new files. It will refuse to overwrite unless overwrite=True.
    - Only use write_file(overwrite=True) when you intend to replace the entire file contents.
    
    After all issues have been resolved, finish by complete by filling out the required JSON with all the updated/final information to prepare for replication execution.
    Rememeber, every response needs to have the the following one of the two formats:
    ----- FORMAT 1 (For when you need to call actions to help accomplish the given task) -------
    Thought: [Your thinking/planning process for completing the task based on interactions so far]
    Action: [call next action to help you solve the task]
    PAUSE
    ----- FORMAT 2 (For when you are ready to give a final response)-------
    Thought: [Your thinking/planning process for completing the task based on interactions so far]
    Answer: [Execute necessary next action to help you solve the task]
    """.strip()
    
    return run_react_loop(
    	system_prompt,
    	known_actions,
    	question,
    	session_state={"analyzers": {}},
    	study_path=study_path,
        stage_name="generate-design",
    	on_final=lambda ans: save_output(ans, study_path)
    )