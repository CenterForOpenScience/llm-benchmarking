"""
R005_LLM_Benchmarking__
|
R005_driver_lemos.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

from info_extractor.extractor import run_extraction
import argparse

def main():
    parser = argparse.ArgumentParser(description="LLM-based Replication Info Extractor")
    parser.add_argument('--study_path', required=True, help="Path to case study folder")
    parser.add_argument('--stage', choices=['1', '2'], required=True, help="Stage: 1 or 2")
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], required=True, help="Difficulty level")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated prompt and exit")
    args = parser.parse_args()
   
    run_extraction(study_path=args.study_path, stage=args.stage, difficulty=args.difficulty, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()
