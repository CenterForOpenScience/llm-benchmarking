# core/actions.py
from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx
from core.tools import (
    list_files_in_folder,
    ask_human_input,
    read_image,
    write_file,
    load_dataset,
    get_dataset_head,
    get_dataset_shape,
    get_dataset_description,
    get_dataset_info,
    read_and_summarize_pdf,
)

def base_known_actions() -> dict:
    """
    Generic actions available to ALL agents.
    Stage-specific agents can extend this with their own entries.
    """
    return {
        "list_files_in_folder": list_files_in_folder,

        "read_txt": read_txt,
        "read_csv": read_csv,
        #"read_pdf": read_pdf,
        "read_pdf": read_and_summarize_pdf,
        "read_json": read_json,
        "read_docx": read_docx,

        "read_image": read_image,

        "load_dataset": load_dataset,
        "get_dataset_head": get_dataset_head,
        "get_dataset_shape": get_dataset_shape,
        "get_dataset_description": get_dataset_description,
        "get_dataset_info": get_dataset_info,

        "ask_human_input": ask_human_input,
        "write_file": write_file,
    }
