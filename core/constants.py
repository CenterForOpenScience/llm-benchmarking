"""
LLM_Benchmarking__
|
constants.py
Created on Mon Jun  9 15:36:52 2025
@authors: Rochana Obadage, Bang Nguyen
"""

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")


TEMPLATE_PATHS = {
    "post_registration_template": "templates/post_registration_schema.json",
    "pre_registration_template": "templates/pre_registration_schema.json",
    "info_extractor_instructions": "templates/info_extractor_instructions.json",
    "extract_eval_prompt_template": "templates/prompts/extract_eval.txt",
    "generate_design_eval_prompt_template": "templates/prompts/generate_design_eval.txt"
}


FILE_SELECTION_RULES = {
    "info_extractor": {
        "easy": {
            "stage_1": {
                "files": ["initial_details_easy.txt", "original_paper.pdf"],
                "folders": {
                    "original_data": {},
                    "original_code": {}
                }
            },
            "stage_2": {
                "files": ["initial_details_easy.txt", "post_registration.json"],
                "folders": {
                    "data": {},
                    "replication_data": {},
                    "replication_code": {},
                    "execution_outputs": {}
                }
            }
        },
        "medium": {
            "stage_1": {
                "files": ["initial_details_medium_hard.txt", "original_paper.pdf"],
                "folders": {
                    "original_data": {},
                    "original_code": {}
                }
            },
            "stage_2": {
                "files": ["initial_details_easy.txt", "post_registration.json"],
                "folders": {
                    "data": {},
                    "replication_data": {},
                    "replication_code": {},
                    "execution_outputs": {}
                }
            }
        },
        "hard": {
            "stage_1": {
                "files": ["initial_details_medium_hard.txt", "original_paper.pdf"],
                "folders": {
                    "original_data": {},
                    "original_code": {}
                }
            },
            "stage_2": {
                "files": ["initial_details_easy.txt", "post_registration.json"],
                "folders": {
                    "data": {},
                    "replication_data": {},
                    "replication_code": {},
                    "execution_outputs": {}
                }
            }
        }
    }
}


GENERATE_REACT_CONSTANTS = {
    "files": {
        "original_paper.pdf": "The pdf file containing the full text of the original paper",
        "initial_details.txt": "Details about the claim from the original paper to be replicated",
        "post_registration.json": "A structured document with key extracted information about the original paper and the claim to be replicated.",
        "replication_data": "The folder containing the data that can potentially be used for the replication.",
    },
    "json_template": "templates/pre_registration_schema.json"
}

GENERATE_EXECUTE_REACT_CONSTANTS = {
    "files": {
        "original_paper.pdf": "The pdf file containing the full text of the original paper",
        "initial_details.txt": "Details about the claim from the original paper to be replicated",
        "post_registration.json": "A structured document with key extracted information about the original paper and the claim to be replicated.",
        "replication_preregistration.json": "A structured document with plans for your replication of the claim.",
        "replication_data_code": "The folder containing the data and code that can be used for the replication.",
    },
    "json_template": "templates/execute_schema.json"
}

EVALUATE_GENERATE_EXECUTE_CONSTANTS = {
    "prompt_template": "templates/prompts/execute_eval.txt",
    "claim_files": {
        "original_paper.pdf": "The pdf file containing the full text of the original paper",
        "initial_details.txt": "Details about the claim from the original paper to be replicated",
        "replication_data": "The folder containing the data and code that can be used for the replication.",
    },
    "agent_files": {
        "post_registration.json": "A structured document with key extracted information about the original paper and the claim to be replicated.",
        "design.log": "The entire process of the design stage, where the agent interacts with the environment to investigate data and replication code to fill out a structured document plan for the replication.",
        "replication_info.json": "Final structured report of the design stage by the agent.",
        "execute.log": "The entire proces of the execute stage, where the agent interacts with the environment to execute the plan from the design stage.",
        "execution_results.json": "Final strcuterd report of the execution stage by the agent."
    },
    "json_template": "templates/evaluate_execute_schema.json"
}


INTERPRET_CONSTANTS = {
    "prompt_template": "templates/prompts/interpret.txt",
    "claim_files": {
        "original_paper.pdf": "The pdf file containing the full text of the original paper",
        "initial_details.txt": "Details about the claim from the original paper to be replicated",
        "replication_data": "The folder containing the data and code that were used for the replication.",
    },
    "agent_files": {
        "post_registration.json": "A structured document with key extracted information about the original paper and the claim to be replicated.",
        "replication_info.json": "Structured report of the agent at the PLANNING stage for the replication of the given claim.",
        "execution_results.json": "Final structured report of the execution stage by the agent."
    },
    "json_template": "templates/interpret_schema.json"
}

EVALUATE_INTERPRET_CONSTANTS = {
    "prompt_template": "templates/prompts/interpret_eval.txt",
    "interpret_results": "interpret_results.json",
    "json_template": "templates/interpret_schema.json"
}

