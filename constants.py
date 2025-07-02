"""
LLM_Benchmarking__
|
constants.py
Created on Mon Jun  9 15:36:52 2025
@authors: Rochana Obadage, Bang Nguyen
"""

API_KEY = ""

TEMPLATE_PATHS = {
    "replication_info_template": "templates/replication_info_schema.json",
    "info_extractor_instructions": "templates/info_extractor_instructions.json"
}

FILE_SELECTION_RULES = {
    "info_extractor": {
        "easy": ["initial_details_easy.txt", "original_paper.pdf", "data_description.txt", "code"],
        "medium": ["initial_details_medium_hard.txt", "original_paper.pdf"],
        "hard": ["initial_details_medium_hard.txt", "original_paper.pdf"]
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
