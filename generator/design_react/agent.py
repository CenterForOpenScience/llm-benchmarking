# generator/design-react.py
import os
import json
import re
from openai import OpenAI
from constants import API_KEY, GENERATE_REACT_CONSTANTS
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG during development to see everything
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO) 
logger.addHandler(console_handler)

from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx # Keep save_output here if the agent orchestrates saving
from generator.design_react.design_tools import load_dataset, get_dataset_head, get_dataset_shape, get_dataset_description, get_dataset_info
from generator.design_react.design_tools import read_image, list_files_in_folder, ask_human_input, write_file

from core.prompts import PREAMBLE, DESIGN, EXAMPLE
from core.agent import Agent, run_react_loop, save_output

system_prompt = "\n\n".join([PREAMBLE, DESIGN, EXAMPLE])
action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE) # Use re.MULTILINE for multiline parsing

# Map action names to their functions
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
    "write_file": write_file
}

def build_file_description(available_files, file_path):
    desc = ""
    for file_id, (file_name, file_desc) in enumerate(available_files.items(), start=1):
        desc += f"{file_id}. {os.path.join(file_path, file_name)}: {file_desc}\n"
    return desc

def _configure_file_logging(study_path: str):
    """
    Configures a file handler for the logger, saving logs to the study_path.
    This function should be called once the study_path is known (e.g., at the start of run_extraction).
    It first removes any existing FileHandlers to avoid duplicate logging if called multiple times.
    """
    # Remove any existing FileHandlers attached to this logger
    # This prevents creating multiple log files or appending to old ones if run_extraction is called multiple times.
    for handler in list(logger.handlers): # Use list() to iterate over a copy, safe to modify
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close() # Important: close the file handle to release the file

    # Construct the log file path within the given study_path
    log_file_name = 'agent_design.log'
    log_directory = study_path # Assuming study_path is already the directory where you want the log
    log_file_full_path = os.path.join(log_directory, log_file_name)

    # Ensure the directory exists before trying to write the log file
    os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True)
    logger.setLevel(logging.DEBUG)

    # Create a new FileHandler
    file_handler = logging.FileHandler(log_file_full_path, mode='a') # 'a' for append
    file_handler.setFormatter(formatter) # Use the globally defined formatter
    file_handler.setLevel(logging.DEBUG) # File logs everything (DEBUG level)
    logger.addHandler(file_handler)
    logger.info(f"File logging configured to: '{log_file_full_path}'.")

def run_design(study_path, show_prompt=False):
    _configure_file_logging(study_path)
    # Load json template
    logger.info(f"Starting extraction for study path: {study_path}")
    template =  read_json(GENERATE_REACT_CONSTANTS['json_template'])
        
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
    If there are potential issues with the provided code such as a data file path that is different from the data files you have looked at, YOU MUST RESOLVE THEM and rewrite the code to a new file.
    
    After all issues have been resolved, finish by complete by filling out the required JSON with all the updated/final information to prepare for replication execution.        
    ".
    """.strip()
    
    return run_react_loop(
    	system_prompt,
    	known_actions,
    	question,
    	session_state={"analyzers": {}},
    	on_final=lambda ans: save_output(ans, study_path)
    )