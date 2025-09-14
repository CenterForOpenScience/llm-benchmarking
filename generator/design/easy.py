# generate_design_easy.py
# Generate → Design stage (Easy tier), aligned to your existing schema
#
# Usage:
#   python generate_design_easy.py --study-path ./case_study_1 --templates-dir ./templates --show-prompt
#
# Output:
#   ./case_study_1/preregistration_design.json

import os
import re
import json
import time
import argparse
import logging
from typing import Any, Dict, Optional

# OpenAI client (reads .env)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

from openai import OpenAI
OPENAI_API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


from ..common.io import read_json, read_text, write_json, save_prompt_log
from ..common.schema_utils import build_combined_template, fallback_build, first_nonnull

# Inputs loader
def load_inputs(study_path: str) -> Dict[str, Any]:
    return {
        "post_registration": read_json(os.path.join(study_path, "post_registration.json")),
        "replication_info": read_json(os.path.join(study_path, "replication_info.json")),
        "validation": read_json(os.path.join(study_path, "info_exractor_validation_results.json")),
        "initial_easy": read_text(os.path.join(study_path, "inputs", "initial_details_easy.txt")),
        "initial_medhard": read_text(os.path.join(study_path, "inputs", "initial_details_medium_hard.txt")),
        "original_pdf_present": os.path.exists(os.path.join(study_path, "inputs", "original_paper.pdf"))
    }


# Prompt + Assistant runner
def build_prompt_design_easy(schema_json: Dict[str, Any]) -> str:
    return (
        "You are the Design-stage generator for a replication preregistration (Easy tier).\n"
        "STRICTLY output valid JSON (no code fences). Conform exactly to the provided JSON schema keys.\n"
        "If a field is unknown, use null (or [] where appropriate). Do not add extra keys.\n\n"
        "EXPECTED OUTPUT SHAPE:\n" + json.dumps(schema_json, indent=2) + "\n\n"
        "RULES:\n"
        "- Use post_registration.json as authoritative for 'original_study'.\n"
        "- For 'Replication', write a concrete plan on the available replication dataset(s) if any; "
        "otherwise set identifiers to null/TBD and include short notes.\n"
        "- Replication hypothesis must be operationalized: fraud ~ β1·violence + β2·violence² + controls; "
        "expected signs β1>0, β2<0.\n"
        "- Models: use logit (if fraud binary) or OLS (if continuous fraud index), mirroring original study's models.\n"
        "- Tools/software: if replication_info suggests Stata (e.g., .do files or docker base is stata-*), prefer Stata; "
        "else Python/R. If unknown, set a sensible default and note it.\n"
        "- Steps should be an ordered list of actions to execute.\n"
        "- Keep the output concise, strictly following the keys of the schema.\n"
    )

def build_message_context(study_path: str, inputs: Dict[str, Any]) -> str:
    context = {
        "study_path": study_path,
        "files_present": sorted(os.listdir(study_path)),
        "post_registration": inputs["post_registration"],
        "replication_info": inputs["replication_info"],
        "info_extractor_validation_results": inputs["validation"],
        "initial_details_easy.txt": inputs["initial_easy"],
        "initial_details_medium_hard.txt": inputs["initial_medhard"]
    }
    return json.dumps(context, indent=2, ensure_ascii=False)

def strip_code_fences(s: str) -> str:
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return s

def run_assistant(prompt: str, user_message: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    assistant = client.beta.assistants.create(
        name="Generate-Design-Easy",
        instructions=prompt,
        model="gpt-4o",
        tools=[]
    )

    run = client.beta.threads.create_and_run(
        assistant_id=assistant.id,
        thread={"messages": [{"role": "user", "content": user_message}]}
    )

    while True:
        status = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
        if status.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(1.5)

    if status.status != "completed":
        logger.warning(f"Assistant status: {status.status}")
        return None

    messages = client.beta.threads.messages.list(thread_id=run.thread_id)
    reply = next((m for m in messages.data if m.role == "assistant"), None)
    if not reply or not reply.content:
        return None

    text = None
    for block in reply.content:
        try:
            text = block.text.value.strip()
            break
        except Exception:
            continue
    if not text:
        return None

    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw head:\n{text[:1000]}")
        return None

# Orchestrator
def run_design_easy(study_path: str, templates_dir: str, show_prompt: bool, logger: logging.Logger) -> Dict[str, Any]:
    os.makedirs(study_path, exist_ok=True)
    inputs = load_inputs(study_path)

    # Build combined schema from your template
    combined_schema = build_combined_template(templates_dir)

    # Build prompt + message
    prompt = build_prompt_design_easy(combined_schema)
    message = build_message_context(study_path, inputs)
    if show_prompt:
        print("=== DESIGN PROMPT (Easy) ===")
        print(prompt)
        print("\n=== DESIGN MESSAGE CONTEXT ===")
        print(message)

    save_prompt_log(study_path, "design_easy", prompt, message)

    # LLM attempt
    prereg = None
    if OPENAI_API_KEY:
        logger.info("Calling OpenAI Assistant for Design (Easy)...")
        prereg = run_assistant(prompt, message, logger)
    else:
        logger.warning("OPENAI_API_KEY not found; skipping LLM call.")

    # Fallback
    if prereg is None:
        logger.info("Using deterministic fallback builder.")
        prereg = fallback_build(combined_schema, inputs)

    # Write output
    out_path = os.path.join(study_path, "preregistration_design.json")
    write_json(out_path, prereg)
    logger.info(f"Saved preregistration to {out_path}")
    return prereg

