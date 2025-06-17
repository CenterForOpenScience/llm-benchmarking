"""
LLM_Benchmarking__
|
constants.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

API_KEY = ""

TEMPLATE_PATHS = {
    "replication_info_template": "templates/replication_info_schema.json",
    "stage1_instructions": "templates/info_extractor_stage1_instructions.json",
    "stage2_instructions": "templates/info_extractor_stage2_instructions.json"
}

FILE_SELECTION_RULES = {
    "1": {
        "easy": ["initial_details_easy.txt", "original_paper.pdf", "data_description.txt", "code"],
        "medium": ["initial_details_medium_hard.txt", "original_paper.pdf"],
        "hard": ["initial_details_medium_hard.txt", "original_paper.pdf"]
    },
    "2": {
        "easy": ["initial_details_easy.txt", "original_paper.pdf","replication_info_stage1.json", "replication_data.csv?"],
        "medium": ["initial_details_medium_hard.txt", "original_paper.pdf","replication_info_stage1.json"],
        "hard": ["initial_details_medium_hard.txt", "original_paper.pdf","replication_info_stage1.json"]
    }
}

