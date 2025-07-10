"""
LLM_Benchmarking__
|
main.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser(description="LLM-based Replication Info Extractor")
    parser.add_argument('--study_path', required=True, help="Path to case study folder")
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], required=True, help="Difficulty level")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    case_name = os.path.basename(os.path.normpath(args.study_path))
    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", args.study_path)
        if match:
            case_name = match.group()

    log_file_name = f"{case_name}_stage_combined_main_log.log"
    os.environ["LOG_FILE"] = log_file_name  

    from logger import get_logger
    logger = get_logger()
    logger.info(f"Running extraction for {args.study_path} at {args.difficulty} difficulty")

    from info_extractor.extractor import run_extraction
    try:
        run_extraction(study_path=args.study_path, difficulty=args.difficulty, show_prompt=args.show_prompt)
    except Exception as e:
        logger.exception(f"Fatal error during run_extraction: {e}")
        raise

if __name__ == "__main__":
    main()
