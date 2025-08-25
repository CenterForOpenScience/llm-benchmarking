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
    "pre_registration_template": "templates/replication_info_schema.json",
    "info_extractor_instructions": "templates/info_extractor_instructions.json",
    "extract_eval_prompt_template": "templates/prompts/extract_eval.txt"
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


INTERPRET_CONSTANTS = {
    "files": {
        "original_paper.pdf": "The pdf file containing the full text of the original paper",
        "preregistration.pdf": "Preregistration that documents the research plan for replicating a focal claim from the original paper.",
        "replication_code": "The folder containing the code you used to run the replication study based on your preregistration.",
        "execution_outputs": "The folder containing output files after executing the code in the replication_code folder"
    },
    "json_template": "templates/interpret_schema.json"
}
