"""
evaluate_extracted_info.py
Extract and evaluate from human replication study at the same time. 
"""

import argparse
import json

from validator.extract_and_evaluate_from_human_rep import extract_from_human_replication_study


def extract_human_replication_info():
    parser = argparse.ArgumentParser(description="Validator: Extract expected replication info from SCORE reports")
    parser.add_argument('--original_paper', type=str, required=True, help='Path to PDF or DOCX original paper')
    parser.add_argument('--preregistration', type=str, required=True, help='Path to PDF or DOCX pre-registration document')
    parser.add_argument('--extracted_json_path', type=str, required=True, help='Path to the extracted replication_info.json')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the expected replication_info.json')
    args = parser.parse_args()

    extract_from_human_replication_study(args.original_paper, args.preregistration, args.extracted_json_path,  args.output_path)


if __name__ == "__main__":
    extract_human_replication_info()
