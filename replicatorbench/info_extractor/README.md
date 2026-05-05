# Information Extractor Module

## Overview
This module extracts structured information from research documents using LLMs (GPT-4) to populate a standardized JSON schema for study replication.

## Components

### `extractor.py`
Main extraction logic that:
- Loads templates and instructions
- Reads input files
- Creates and manages OpenAI Assistant interactions
- Handles stage 1 (original study) and stage 2 (replication) extraction
- Saves outputs

### `file_utils.py`
File handling utilities:
- Reads PDF, TXT, JSON, and CSV files
- Selects files based on difficulty rules
- Saves output JSON

### `prompt_builder.py`
Constructs LLM prompts by combining:
- JSON template schemas
- Difficulty level-specific instructions


## Usage
```python
from info_extractor.extractor import run_extraction

run_extraction(
    study_path="path/to/study_folder",
    stage="stage_1",  # or "stage_2"
    difficulty="easy",  # "medium" or "hard"
    show_prompt=False
)
```

## Dependencies
* Python 3.8+
* `openai` (>= 1.0)
* `pymupdf` (for PDF reading)
* `pandas` (for CSV reading)


## Configuration
Edit `R005_constants.py` to:
* Set your OpenAI API key
* Adjust template paths
* Modify file selection rules