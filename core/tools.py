import base64
from openai import OpenAI
import os
import json
import pandas as pd
import pyreadr
from core.constants import API_KEY
from typing import Dict, Any, Optional, Tuple
import io # Add this import at the top of your file
from pathlib import Path

from pypdf import PdfReader

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
        Handles both .csv, .xlsx, and .dta files and potential errors.
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
            elif file_extension == '.dta':
                return pd.read_stata(self.file_path)
            elif file_extension.lower() == '.rds':
                return pyreadr.read_r(self.file_path)[None]
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

def write_file(file_path: str, file_content: str) -> str:
    """
    Create a file at file_path and dump file_content into it.

    Use this tool when you need to write new code or modify existing code.

    Args:
        file_path (str): Path to where new file will be created

    Returns:
        str: A confirmation if the file is approved and has been created or a rejection message.
    """
     # Determine the full, absolute path for user confirmation
    full_path = Path.cwd() / file_path

    # Print a clear message to the user indicating the agent needs help
    print("\n🤔 [AGENT ASKS TO WRITE A NEW FILE 🤔")
    print(f"FULL PATH: {full_path}")
    print(f"FILE CONTENT:\n---\n{file_content}\n---")

    # Get input from the user
    user_response = input("Do you approve? (yes/no): ")

    # Check the user's response
    if user_response.lower().strip() != 'yes':
        print("❌ User denied execution.")
        return f"Command execution denied by the user:\n{user_response}"

    try:
        # Ensure the parent directory of the full path exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the specified full path
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
            
        success_message = f"✅ Successfully wrote content to {full_path}"
        print(success_message)
        return success_message
    except Exception as e:
        error_message = f"❌ Error writing file to {full_path}: {e}"
        print(error_message)
        return error_message

def read_and_summarize_pdf(file_path: str) -> str:
    """
    Reads a PDF file. If the PDF is short (<= 15 pages), it returns the full text.
    If the PDF is long (> 15 pages), it splits the text into chunks and uses the
    LLM to summarize each chunk, returning a consolidated summary to save context window.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Full text or a consolidated summary of the PDF.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    try:
        reader = PdfReader(file_path)
        number_of_pages = len(reader.pages)
        
        # Extract all text first
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # THRESHOLD: If 15 pages or less, just return the text as is.
        if number_of_pages <= 15:
            print(f"PDF is short ({number_of_pages} pages). Returning full text.")
            return f"--- START OF PDF CONTENT ({number_of_pages} pages) ---\n{full_text}\n--- END OF PDF CONTENT ---"

        # LOGIC FOR LONG PDFS
        print(f"PDF is long ({number_of_pages} pages). Summarizing content to prevent overflow...")
        
        # Split text into chunks of roughly 12,000 characters (approx 3-4k tokens)
        chunk_size = 12000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        summaries = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{total_chunks}...")
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful research assistant. Summarize the following text from a technical paper/document. Capture key methodologies, specific metrics, results, and conclusions. Do not lose specific data points."
                    },
                    {
                        "role": "user", 
                        "content": chunk
                    }
                ]
            )
            summaries.append(completion.choices[0].message.content)
            
        consolidated_summary = "\n\n".join(summaries)
        
        return (f"--- PDF SUMMARY (Document was {number_of_pages} pages long) ---\n"
                f"The document was too long to read directly, so here is a detailed summary of all sections:\n\n"
                f"{consolidated_summary}")

    except Exception as e:
        return f"Error reading or summarizing PDF: {e}"