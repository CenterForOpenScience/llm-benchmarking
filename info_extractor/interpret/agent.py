# main_agent.py

import os
import json
import re
from openai import OpenAI
from constants import API_KEY, INTERPRET_CONSTANTS
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG during development to see everything
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log_file_path = 'agent_process.log' # You can change this log file name
# file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append, 'w' for overwrite
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO) 
logger.addHandler(console_handler)

client = OpenAI(api_key=API_KEY) 
from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf # Keep save_output here if the agent orchestrates saving
from info_extractor.interpret.interpret_tools import read_image, list_files_in_folder

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                                model="gpt-4o",
                                temperature=0,
                                messages=self.messages)
        return completion.choices[0].message.content

# --- Agent System Prompt ---
agent_prompt = """
You are an advanced research assistant specialized in extracting structured information from replication studies of some focal claim in a research paper.
You operate in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you output an Answer in JSON format.

Use Thought to describe your reasoning about the question and what actions you need to take.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

1. list_files_in_folder:
    e.g. list_files_in_folder: "data/study_A/datasets"
    Description: Lists all files within a specified folder
    Returns: Names of all files within the specified folder with their names as a single string,
    with each file separated by a comma.

2.  read_txt:
    e.g. read_txt: "data/study_X/abstract.txt"
    Description: Reads the plain text content of a file with .txt or .do (Stata do-file) extensions. This is the default reader if a specific file type is not recognized.
    Returns: The content of the file as a string.

3.  read_pdf:
    e.g. read_pdf: "data/study_Y/methods.pdf"
    Description: Extracts and reads the text content from a PDF (.pdf) file.
    Returns: The extracted text content of the PDF as a string.

4.  read_json:
    e.g. read_json: "data/study_Z/config.json"
    Description: Reads and parses a JSON (.json) file.
    Returns: The content of the JSON file as a Python dictionary (which will be converted to string representation for observation).

5.  read_csv:
    e.g. read_csv: "data/study_W/data.csv"
    Description: Reads a CSV (.csv) file and returns its content as a string table.
    Returns: A string representation of the CSV data.

6. read_image:
   e.g read_image: "data/study_T/image.png"
   Description: Take in an input image of type .png, .jpeg, .jpg, .webp, or .gif and describe in natural language what the image is about.
   Returns: Textual description of the provided image

Important: When reading a file, you must choose the *specific* reader tool based on the file's extension. If the extension is not listed (e.g., .docx, .xlsx), you should use `read_txt` as a fallback. 
Remember, you don't have to read all provided files if you don't think they are necessary to fill out the required JSON.

Example Session:

Question: Extract information about the original paper and claim to be replicated from the provided files and fill out this JSON template
    {
      "statement": "The main claim made by the original study.",
      "hypothesis": "A testable hypothesis based on the claim.",
      "original_coefficient": "Numeric value indicating strength/direction of effect.",
      "original_p_value": "P-value for testing statistical significance.",
      "direction": "Positive, negative, or null effect.",
      "study_type": "Type of study (Experimental, Observational, Meta-Analysis)."
    }
You will have access to the following documents:
1. original_paper.pdf: The pdf file containing the full text of the original paper 
2. initial_details.txt: A document containing the following details: (1) the focal claim from the original that needs to be replicated.

Thought: The required JSON centers around the main claim. I need to determine what the claim is from initial_detailst.txt. I should use the 'read_txt' tool.
Action: read_txt: initial_details.txt
PAUSE

You will be called again with this:

Observation:[CLAIM]
The relationship between violence and election fraud follows an inverted U-shape: fraud increases with violence up to a certain level, then decreases.

You then output:

Thought: I now know about the claim to be replicated. I need to look for additional information about the claim from the full paper. I should use the 'read_pdf' tool.
Action: read_pdf: original.pdf
PAUSE

You will be called again with this:
Observation: [FULL PAPER PDF redacted here]

You then output:
Answer: {
    "statement": "The relationship between violence and election fraud follows an inverted U-shape: fraud increases with violence up to a certain level, then decreases.",
    "hypothesis": [
      "H1: The linear association between violence and election fraud will be positive.",
      "H* (SCORE focal test): The quadratic association between violence and election fraud will be negative."
    ],
    "original_coefficients": {
        "linear_term": 8.477,
        "squared_term": -13.748
    },
    "original_p_value": {
        "linear_term": "<0.05",
        "squared_term": "<0.01"
    },
    "direction": "Inverted U-shape effect",
    "study_type": "Observational"
  }
""".strip()

# Map action names to their functions
known_actions = {
    "list_files_in_folder": list_files_in_folder,
    "read_txt": read_txt,
    "read_csv": read_csv,
    "read_pdf": read_pdf,
    "read_json": read_json,
    "read_image": read_image,
}

action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE) # Use re.MULTILINE for multiline parsing
def save_output(extracted_json, study_path):
    final_output = {
        "stage": "interpret",
        **extracted_json
    }
    output_path = os.path.join(study_path, "extracted_replication_results.json")
    extracted_json = final_output
    with open(output_path, 'w') as f:
        json.dump(extracted_json, f, indent=2)

    logger.info(f"Interpret stage output saved to {output_path}")
    
def query_agent(question: str, max_turns: int = 10, study_path_for_saving=None):
    """
    Main function to query the agent and orchestrate the extraction process.
    """
    i = 0
    bot = Agent(agent_prompt)
    next_prompt = question

    final_extracted_data = {} # To accumulate results
    
    MAX_DISPLAY_PROMPT_LEN = 2000

    while i < max_turns:
        i += 1
        logger.info(f"\n--- Turn {i} ---")
        # print(f"Agent input: {next_prompt}")
        display_prompt = next_prompt
        if len(display_prompt) > MAX_DISPLAY_PROMPT_LEN:
            display_prompt = display_prompt[:MAX_DISPLAY_PROMPT_LEN] + "\n... (truncated for display)"
        logger.debug(f"Agent input: {display_prompt}")

        result = bot(next_prompt) # Get LLM's thought/action/answer
        logger.info(f"Agent output:\n{result}")

        # Check if the LLM provided a final answer
        if "Answer:" in result:
            try:
                answer_match = re.search(r'Answer:\s*(\{.*?\})\s*$', result, re.DOTALL)
                if answer_match:
                    json_answer_str = answer_match.group(1).strip()
                else:
                    json_answer_str = result.split("Answer:", 1)[1].strip()
                    if json_answer_str.strip().startswith('{') and json_answer_str.strip().endswith('}'):
                            pass # Looks like valid JSON, proceed
                    else:
                        logger.warning(f"Warning: Answer found but doesn't look like clean JSON: {json_answer_str[:200]}...")
                        # Try to find the JSON part more aggressively
                        json_start = json_answer_str.find('{')
                        json_end = json_answer_str.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            json_answer_str = json_answer_str[json_start : json_end + 1]
                        else:
                            raise ValueError("Could not find a valid JSON structure after 'Answer:'")
                json_start = json_answer_str.find('{')
                json_end = json_answer_str.rfind('}')
                if json_start == -1 or json_end == -1 or json_end < json_start:
                    raise ValueError("Could not find a valid JSON object (missing curly braces) after cleaning.")

                final_answer = json.loads(json_answer_str[json_start : json_end + 1])
                logger.info("\n--- Final Answer ---")
                logger.info(json.dumps(final_answer, indent=2))
                # Agent decides when to save the output now
                if study_path_for_saving:
                    save_output(final_answer, study_path_for_saving)
                logger.info("Process completed")
                return final_answer
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing final JSON answer: {e}")
                logger.error(f"Raw answer: {json_answer_str}")
                return {"error": "Failed to parse final answer JSON"}
            except Exception as e:
                logger.error(f"An error occurred processing final answer: {e}")
                return {"error": str(e)}
        else:
            actions_matches = [
                action_re.match(line)
                for line in result.split('\n')
                if action_re.match(line)
            ]

            if actions_matches:
                # There is an action to run
                match = actions_matches[0]
                action, action_input_str = match.groups()

                logger.info(f" -- Running Action: {action} with input: {action_input_str}")

                if action not in known_actions:
                    logger.error(f"Unknown action: {action}: {action_input_str}") 
                    raise Exception(f"Unknown action: {action}: {action_input_str}")

                # Parse action_input_str as JSON for tool arguments if it's a dict, otherwise keep as string
                parsed_action_input = None
                try:
                    # Attempt to parse as JSON. If it fails, assume it's a string argument.
                    parsed_action_input = json.loads(action_input_str)
                except json.JSONDecodeError:
                    parsed_action_input = action_input_str # It's a string, not JSON

                observation = None
                if isinstance(parsed_action_input, dict):
                    # If it's a dictionary, unpack it as keyword arguments
                    observation = known_actions[action](**parsed_action_input)
                else:
                    # Otherwise, pass it as a single positional argument
                    observation = known_actions[action](parsed_action_input)

                # print(f"Observation: {observation}")
                next_prompt = f"Observation: {observation}"
            else:
                logger.warning("Agent did not propose an action. Terminating.")
                # If the agent doesn't provide an action or an answer, something is wrong or it's stuck.
                return {"error": "Agent did not provide a recognized action or final answer."}

    print("Max turns reached. Agent terminated without a final answer.")
    return {"error": "Max turns reached without a final answer."}


def build_file_description(available_files, file_path):
    desc = ""
    for file_id, (file_name, file_desc) in enumerate(available_files.items(), start=1):
        desc += f"{file_id}. {os.path.join(file_path, file_name)}: {file_desc}\n"
    return desc

def _configure_file_logging(study_path: str):
    """
    Configures a file handler for the logger, saving logs to the study_path.
    This function should be called once the study_path is known (e.g., at the start of run_extraction).
    It first removes any existing FileHandlers to avoid duplicate logging if called multiple times.
    """
    # Remove any existing FileHandlers attached to this logger
    # This prevents creating multiple log files or appending to old ones if run_extraction is called multiple times.
    for handler in list(logger.handlers): # Use list() to iterate over a copy, safe to modify
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close() # Important: close the file handle to release the file

    # Construct the log file path within the given study_path
    log_file_name = 'agent_process.log'
    log_directory = study_path # Assuming study_path is already the directory where you want the log
    log_file_full_path = os.path.join(log_directory, log_file_name)

    # Ensure the directory exists before trying to write the log file
    os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True)

    # Create a new FileHandler
    file_handler = logging.FileHandler(log_file_full_path, mode='a') # 'a' for append
    file_handler.setFormatter(formatter) # Use the globally defined formatter
    file_handler.setLevel(logging.DEBUG) # File logs everything (DEBUG level)
    logger.addHandler(file_handler)

    logger.info(f"File logging configured to: '{log_file_full_path}'.")


def run_extraction(study_path, show_prompt=False):
    _configure_file_logging(study_path)
    # Load json template
    logger.info(f"Starting extraction for study path: {study_path}")
    with open(INTERPRET_CONSTANTS['json_template']) as f:
        template = json.load(f)  
    
    query_question = f"""
    Extract information from the provided files and fill out this JSON template
        {json.dumps(template)}
    You will have access to the following documents:
    {build_file_description(INTERPRET_CONSTANTS['files'], study_path)}
    """.strip()
    
    query_agent(
        query_question,
        study_path_for_saving=study_path,
    )