import os
import json
import re
from core.constants import GEN_GOLD_ANALYSIS_CONSTANTS
import sys

from info_extractor.file_utils import read_json # Keep save_output here if the agent orchestrates saving
from core.prompts import PREAMBLE_ROBUSTNESS, EXAMPLE_ROBUSTNESS, GENERATE_GOLD_ANALYSIS, ROBUSTNESS_DESIGN_CODE_MODE_POLICY, ROBUSTNESS_EXTRACT_POLICY
from core.agent import Agent, run_react_loop, save_output
from core.utils import build_file_description, configure_file_logging, get_logger
from core.actions import base_known_actions, get_tool_definitions, read_file

logger, formatter = get_logger(name="robustness")
action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE) # Use re.MULTILINE for multiline parsing
known_actions = base_known_actions()

def build_system_prompt(code_mode: str) -> str:
    # Put the policy in SYSTEM prompt
    return "\n\n".join([PREAMBLE_ROBUSTNESS, GENERATE_GOLD_ANALYSIS, EXAMPLE_ROBUSTNESS])

def run_gen_gold_analysis(study_path, tier: str = "easy", code_mode: str = "python", model_name: str = "gpt-5"):
    configure_file_logging(logger, study_path, f"gen_gold_analysis.log")
    # Load json template
    logger.info(f"Starting gold analysis extraction for study path: {study_path}")
    analysis_schema =  read_file(GEN_GOLD_ANALYSIS_CONSTANTS['analysis_schema'])
    code_policy = ROBUSTNESS_DESIGN_CODE_MODE_POLICY.get(code_mode, ROBUSTNESS_DESIGN_CODE_MODE_POLICY["native"])
    extract_input_rules = ROBUSTNESS_EXTRACT_POLICY.get("input", "")
    
    system_prompt = build_system_prompt(code_mode)
    
    question = f"""
You are a researcher specialized in analytical robustness evaluation in the social and behavioral sciences.
You will be given the following information:
1. original_paper.pdf: A published paper containing the focal claim.
2. proposed_analysis.pdf: A planned robustness analysis path. This path comes from an independent reanalysis of the same focal claim using the same original data. It may use different reasonable analysis choices from the original paper, but it must still test the same focal claim.
3. data/: This folder contains the original data used for the robustness reanalysis. It may also contain analysis code or supporting files needed to execute the planned analysis.

Your task is to extract relevant information about proposed analysis following the input rules:
{extract_input_rules}

Then fill out this template:
=== START OF JSON OUTPUT===
{analysis_schema}
=== END OF JSON OUTPUT ===

You can use any tools and inspect any documents available to you to help you accomplish the task, if need be. 

{code_policy}

Output Requirements:\n- Return a valid JSON object only.\n- Do NOT wrap the output in markdown (no ```json).\n- Do NOT include extra text, commentary, or notes.\n\n Ensure accuracy and completeness.\n- Strictly use provided sources as specified.
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
        stage_name="execute-gen_gold_analysis",
    	on_final=lambda ans: save_output(ans, study_path, "analysis_info.json", "execute-gen_gold_analysis"),
    	model_name=model_name,
        logger=logger
    )