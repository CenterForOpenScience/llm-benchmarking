import base64
from openai import OpenAI
import os
import json
from constants import API_KEY

client = OpenAI(api_key=API_KEY)

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
    file_info += f"All files: {", ".join(file_names)}"
    return file_info
