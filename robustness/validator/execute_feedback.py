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

def run_evaluate_execute_feedback(study_path, tier: str = "easy", code_mode: str = "python", model_name: str = "gpt-5"):
    configure_file_logging(logger, os.path.join(study_path, f"evals/{model_name}"), f"evaluated_execute_capabilityevaluated_execute_capability.log")
    # Load json template
    logger.info(f"Starting evaluation of execution capability for study path: {study_path}")
    
    system_prompt = build_system_prompt(code_mode)
    
    question = f"""
You are a researcher specialized in evaluating the quality of research reproduction and validation attempts in the social sciences.
You will be given the following information:
1. original_paper.pdf: A published paper in the social and behaviorial sciences domain.
2. analysis_info.json: A proposed analysis, from an independent researcher different from the original authors of original_paper.pdf, to validate a focal claim in the paper. This analysis can have DIFFERENT design choices from those in the original paper, but can still validate the focal claim.

You are also given an agent attempt that follows the reanalysis plan proposed in analysis_info.json: 
3. data/: This folder contains original data file used in original_paper.pdf AND potentially relevant code used by the agent during its execution.
4. _logs/execute_easy__python.log: This is the log file documenting the agent's attempt in executing the reanalysis proposed in analysis_info.info.
4. execution_results.json: This a structured report of the execution process filled out the by agent itself at the end of execution.

YOUR TASK is to EVALUATE the execution performance of the agent based on the entire process and the reported outcome.
Specifically, you must classify the agent attempt into one of the following categories:
(1) Acceptable: The agent successfully executes the proposed analysis faithfully with no major or significant changes from the proposed plan in analysis_info.json.
(2) Unacceptable: The agent could not execute the proposed analysis, or has major deviations from the proposed analysis. For example, some models, variables, or methodologies listed in the plan were dropped during either the execution process, or during filling out the report (Agent only relies on ONE single model result instead of multiple results to make conclusion in exeuction_results). These are just example of potential significant deviations. You should think of other potential significant deviations issues that result in an unfaithful execution of the proposed analysis.

You can use any tools and inspect any documents available to you to help you accomplish the task, if need be. 
After calling all necessary actions to accomplish the task, use this tempalte for your final response:
{{
    "execute_feedback":  // one of the following VERBATIM "(1) Acceptable",OR  "(2) Unacceptable" //,
    "details" // provide detailed feedback for the agent in case your evaluation is "(2) Unacceptable", especially why and how can they improve their execution to become more faithful to the proposed analysis in analysis_info.info. //,
}}
""".strip()
    tool_definitions = get_tool_definitions()
    return run_react_loop(
    	system_prompt,
    	known_actions,
    	tool_definitions,
    	question,
    	session_state={"analyzers": {}},
    	study_path=study_path,
        stage_name="evaluate-execute-feedback",
    	on_final=lambda ans: save_output(ans, os.path.join(study_path, f"evals/{model_name}"), "execute_feedback.json", "evaluate-execute-feedback"),
    	model_name=model_name,
        logger=logger,
    )