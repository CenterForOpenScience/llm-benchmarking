import os
import json
import re
import sys

from core.prompts import PREAMBLE, EXAMPLE, GENERATE_GOLD_ANALYSIS
from core.agent import Agent, run_react_loop, save_output
from core.utils import configure_file_logging, get_logger
from core.actions import base_known_actions, get_tool_definitions, read_file

logger, formatter = get_logger(name="robustness")
action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE)
known_actions = base_known_actions()

def build_system_prompt(code_mode: str) -> str:
    # Put the policy in SYSTEM prompt
    return "\n\n".join([PREAMBLE, GENERATE_GOLD_ANALYSIS, EXAMPLE])

def run_evaluate_execute_capability(study_path, tier: str = "easy", code_mode: str = "python", model_name: str = "gpt-5"):
    configure_file_logging(logger, os.path.join(study_path, f"evals/{model_name}"), f"evaluated_execute_capabilityevaluated_execute_capability.log")
    # Load json template
    logger.info(f"Starting evaluation of execution capability for study path: {study_path}")
    
    system_prompt = build_system_prompt(code_mode)
    
    question = f"""
You are a researcher specialized in evaluating the quality of research reproduction and validation attempts in the social sciences.
You will be given the following information:
1. original_paper.pdf: A published paper in the social and behaviorial sciences domain.
3. proposed_analysis.pdf: A proposed analysis, from an independent researcher different from the original authors of original_paper.pdf, to validate a focal claim in the paper. This analysis can have DIFFERENT design choices from those in the original paper, but can still validate the focal claim. This is an EXAMPLE of an ACCEPTABLE reanalysis where all reported code results (OF the reanalysis, NOT the original paper) are found to be reproduciable. 
4. data/: This folder contains original data file used in original_paper.pdf AND possibly a code file based on the proposed_analysis.pdf

You are also given this: 
5. _logs/execute_easy__python.log: This is the log file documenting an agent's attempt in executing the same reanalysis in proposed_analysis.pdf. YOUR TASK is to EVALUATE the execution performance of the agent based on this log.
Specifcally, you must classify the agent attempt into one of the following categories:
(1) Unsuccessful execution: No comparable results are produced to match against the results reported in proposed_analysis.pdf.
(2) Successful execution, results mismatches: Agent successfully executed the same reanalysis discussed in proposed_analysis.pdf, BUT obtain a significant different findings (Significant deviations in values of statistics or different conclusions regarding the validity of the focal claim in original_paper.pdf. Remember that you should use the results and conclusion reported in proposed_analysis.pdf as the reference.)
(3) Successful execution, no result mismatch: Agent successfully executed the same reanalysis discussed in proposed_analysis.pdf, AND obtain the same results and conclusion reported in proposed_analysis.pdf

You can use any tools and inspect any documents available to you to help you accomplish the task, if need be. 

After calling all necessary actions to accomplish the task, use this tempalte for your final response:
{{
    "evaluated_execute_performance":  // one of the following VERBATIM "(1) Unsuccessful execution", "(2) Successful execution, results mismatches", OR "(3) Successful execution, no result mismatch". //
}}
""".strip()
    print(f"starting design phase with {model_name}\n")
    tool_definitions = get_tool_definitions()
    return run_react_loop(
    	system_prompt,
    	known_actions,
    	tool_definitions,
    	question,
    	session_state={"analyzers": {}},
    	study_path=study_path,
        stage_name="evaluate-execute-capability",
    	on_final=lambda ans: save_output(ans, os.path.join(study_path, f"evals/{model_name}"), "evaluated_execute_capability.json", "evaluate-execute-capability"),
    	model_name=model_name,
        logger=logger
    )