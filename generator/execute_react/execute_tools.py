import base64
from openai import OpenAI
import os
import json
import pandas as pd
from constants import API_KEY
from typing import Dict, Any, Optional, Tuple
import io # Add this import at the top of your file
import shlex
import subprocess

client = OpenAI(api_key=API_KEY)

class DataFrameAnalyzer:
    """
    A class to load and analyze a tabular dataset from a file.

    Loads a DataFrame once upon initialization and provides methods
    to perform common exploratory analysis tasks.
    """
    def __init__(self, file_path: str):
        """
        Initializes the analyzer and loads the data.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = self._load_data()

    def _load_data(self) -> Optional[pd.DataFrame]:
        """
        Private method to load data from the file_path.
        Handles both .csv and .xlsx files and potential errors.
        """
        # Get the file extension from the file path
        _, file_extension = os.path.splitext(self.file_path)
        
        try:
            print(f"Loading data from {self.file_path}...")
            
            # Choose the correct pandas function based on the extension
            if file_extension == '.csv':
                return pd.read_csv(self.file_path)
            elif file_extension in ['.xlsx', '.xls']:
                # You might need to install openpyxl: pip install openpyxl
                return pd.read_excel(self.file_path)
            else:
                print(f"Error: Unsupported file type '{file_extension}'.")
                return None

        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
            return None
        except (pd.errors.ParserError, ValueError, Exception) as e:
            # Catch pandas parsing errors and other potential issues
            print(f"An error occurred while reading the file: {e}")
            return None

    def get_head(self, n: int = 5) -> Optional[pd.DataFrame]:
        """Returns the first n rows of the DataFrame."""
        if self.df is not None:
            return self.df.head(n)
        return None

    def get_shape(self) -> Optional[Tuple[int, int]]:
        """Returns the shape (rows, columns) of the DataFrame."""
        if self.df is not None:
            # .shape is an attribute, not a function
            return self.df.shape
        return None

    def get_info(self) -> str: # Change the return type hint to str
        """
        Returns a concise summary of the DataFrame as a string.
        """
        if self.df is not None:
            # Create an in-memory text buffer
            buffer = io.StringIO()
            
            # Tell df.info() to write its output to the buffer instead of the console
            self.df.info(buf=buffer)
            
            # Get the string from the buffer and return it
            return buffer.getvalue()
        return "Error: DataFrame not loaded."

    def get_description(self) -> Optional[pd.DataFrame]:
        """Returns descriptive statistics of the DataFrame."""
        if self.df is not None:
            return self.df.describe()
        return None
    
def load_dataset(session_state: Dict[str, Any], file_path: str) -> str:
    """
    Loads a dataset and stores its analyzer in the session state.
    """
    analyzers = session_state["analyzers"]
    if file_path in analyzers:
        return f"Dataset '{file_path}' is already loaded."
    
    analyzer = DataFrameAnalyzer(file_path)
    if analyzer.df is not None:
        analyzers[file_path] = analyzer
        return f"Successfully loaded dataset '{file_path}'."
    else:
        return f"Failed to load dataset from '{file_path}'."

def get_dataset_shape(session_state: Dict[str, Any], file_path: str) -> str:
    """
    Gets the shape from an analyzer in the session state.
    """
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_shape())

def get_dataset_head(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_head())

def get_dataset_info(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_info())

def get_dataset_description(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_description())

def read_image(file_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    base64_image = encode_image(file_path)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Describe this image in details." },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ])
    return completion.choices[0].message.content


def run_shell_command(command: str) -> str:
    """
    Executes a shell command in the local terminal after receiving human confirmation.

    Args:
        command (str): The complete shell command to execute (e.g., "python3 my_script.py --arg value").

    Returns:
        str: The combined standard output and standard error from the command, or a rejection message.
    """
    # 1. Ask for human confirmation, showing the exact command
    print(f"\n🤔 [HUMAN CONFIRMATION REQUIRED] 🤔")
    print(f"Agent wants to execute the command: `{command}`")
    user_response = input("Do you approve? (yes/no): ")

    # 2. Check the user's response
    if user_response.lower().strip() != 'yes':
        print("❌ User denied execution.")
        return "Command execution denied by the user."

    print(f"✅ User approved. Executing command...")
    try:
        # 3. Execute the command securely
        # shlex.split handles arguments with spaces correctly
        args = shlex.split(command)
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False # Don't raise an exception on errors
        )
    
        # 4. Return the full output to the agent
        output = f"Exit Code: {result.returncode}\n---STDOUT---\n{result.stdout}\n---STDERR---\n{result.stderr}"
        return output.strip()

    except FileNotFoundError:
        return f"Error: The command '{args[0]}' was not found. Make sure it's installed and in your system's PATH."
    except Exception as e:
        return f"An error occurred while executing the command: {e}"

def run_stata_do_file(file_path: str) -> str:
    """
    Executes a Stata .do file in batch mode after human confirmation,
    captures the output from the corresponding .log file, and returns it as a string.

    Args:
        file_path (str): The path to the Stata .do file.

    Returns:
        str: The full content of the generated .log file, or an error message.
    """
    # 1. Determine the expected log file path from the do-file path
    base_name, _ = os.path.splitext(file_path)
    log_path = base_name.split("/")[-1] + ".log"

    # NOTE: The Stata executable might have a different name on your system (e.g., 'stata-se', 'stata')
    command = f"stata-mp -b do {file_path}"

    # 2. Get human confirmation before executing
    print(f"\n🤔 [HUMAN CONFIRMATION REQUIRED] 🤔")
    print(f"Agent wants to execute the Stata script: `{file_path}`")
    user_response = input(f"This will run the command: `{command}`\nDo you approve? (yes/no): ")

    if user_response.lower().strip() != 'yes':
        return "Command execution denied by the user."
    
    try:
        # 3. Execute the Stata command
        print(f"✅ User approved. Executing Stata script...")
        args = shlex.split(command)
        result = subprocess.run(args, capture_output=True, text=True, check=False)

        # If Stata itself threw an error, return that for debugging
        if result.returncode != 0:
            return f"Stata execution failed.\n---STDERR---\n{result.stderr}"

        # 4. Read the entire contents of the generated log file
        print(f"Execution finished. Reading output from '{log_path}'...")
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            log_content = log_file.read()

        # 5. (Good practice) Clean up the log file
        os.remove(log_path)
        
        return log_content

    except FileNotFoundError:
        return f"Error: Could not find the generated log file at '{log_path}'. Make sure Stata is installed and the do-file path is correct."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def ask_human_input(question_to_ask: str) -> str:
    """
    Prompts the human user for input in the terminal.

    Use this tool when you are stuck, need clarification, or require 
    information that you cannot find or deduce from the available files.

    Args:
        question_to_ask (str): The clear, specific question to ask the human user.

    Returns:
        str: The human's response from the terminal.
    """
    # Print a clear message to the user indicating the agent needs help
    print("\n🤔 [AGENT NEEDS HUMAN INPUT] 🤔")
    print(f"Agent's Question: {question_to_ask}")
    
    # Get input from the user
    human_response = input("Your Response: ")
    
    return human_response

    


def list_files_in_folder(folder_path: str) -> str:
    """
    Lists all files within a specified folder and returns their names as a single string,
    with each file separated by a comma.

    Args:
        folder_path: The absolute or relative path to the folder.

    Returns:
        A string containing the names of all files in the folder,
        each separated by a newline character. If the folder does not exist
        or is not a directory, an error message is returned.
    """
    # Check if the provided path exists
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' does not exist."

    # Check if the provided path is actually a directory
    if not os.path.isdir(folder_path):
        return f"Error: Path '{folder_path}' is not a directory."

    file_names = []
    # Iterate over all entries in the folder
    for entry in os.listdir(folder_path):
        # Construct the full path to the entry
        full_path = os.path.join(folder_path, entry)
        # Check if the entry is a file (and not a subdirectory)
        if os.path.isfile(full_path):
            file_names.append(entry)

    # Sort the file names alphabetically for consistent output
    file_names.sort()
    
    file_info = f"Folder path: {folder_path}\n"
    file_info += f"All files: {', '.join(file_names)}"
    return file_info
