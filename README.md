# LLM Benchmarking Project

Welcome to the official repository for the **LLM Benchmarking Project**, led by the Center for Open Science (COS). This project aims to evaluate the capabilities of large language model (LLM) agents across key components of the scientific research lifecycle, including **replication**, **peer review**, and **research design**.

## 🔍 What This Project Is About

We are developing a modular benchmark framework to assess whether and how LLM agents can:

* **Replicate published scientific findings**
* **Evaluate the quality and credibility of research outputs**
* **Generate valid and meaningful research designs**

This work builds on the conceptual structure outlined in our Open Philanthropy grant, emphasizing real-world relevance, task diversity, and community participation.

## 🚧 Current Status

This repository is in **active development**. Right now, it hosts internal work on:

* Task definitions for replication benchmarking
* Agent development and evaluation pipelines
* Experimental scaffolding for testing and refining agent performance

Over time, we will open up parts of this repo for **community use and feedback**, including:

* Evaluation harnesses
* Benchmarks and datasets
* Contribution guidelines for task submissions and agent evaluation strategies


## Project Structure
```
llm-benchmarking/
│
├── info_extractor/
│   ├── extractor.py
│   ├── file_utils.py
│   ├── prompt_builder.py
│   └── README.md
│
├── validator/
│   ├── extract_from_human_replication_study.py
│   ├── compare_outputs.py                        
│   └── README.md                                 
│
├── templates/
│   ├── info_extractor_instructions.json
│   ├── interpret_schema.json
│   ├── post_registration_schema.json
│   └── replication_info_schema.json
│
├── samples/
│   ├── initial_details_easy.txt
│   └── initial_details_medium_hard.txt             
│
├── constants.py
├── extract_human_replication_info.py
├── main.py
├── README.md
└── validate_info_extractor.py


```

## 🧰 Installation
1. Clone repository:
   ```bash
   git clone https://github.com/CenterForOpenScience/llm-benchmarking.git
   cd llm-benchmarking
   ```

2. Install dependencies:
   ```bash
   pip install openai pymupdf pandas python-docx dotenv pyreadr
   ```

3. Configure API key in `constants.py`

## 🔧 Usage

### Info Extractor Module

This module runs LLM-based extraction of structured metadata from original and replication studies (based on the difficulty level).

```bash
# Stage 1: Extract from original study
python main.py --study_path ./studies/case_study_1 --stage stage_1 --difficulty easy

# Stage 2: Extract from replication study
python main.py --study_path ./studies/case_study_1 --stage stage_2 --difficulty easy
```

**Arguments:**
* `--study_path`: Path to the study folder
* `--stage`: `"stage_1"` for original, `"stage_2"` for replication
* `--difficulty`: `"easy"`, `"medium"`, or `"hard"`
* `--show-prompt`: Print the constructed LLM prompt for debugging

#### Output Files
* Stage 1 → `post_registration.json`
* Stage 2 → `replication_info.json`

#### Input File Requirements

##### Stage 1

| Difficulty | Required Files |
|------------|----------------|
| Easy | `initial_details_easy.txt`, `original_paper.pdf` |
| Medium | `initial_details_medium_hard.txt`, `original_paper.pdf` |
| Hard | `initial_details_medium_hard.txt`, `original_paper.pdf` |

##### Stage 2

| Difficulty | Required Files |
|------------|----------------|
| Easy | `initial_details_easy.txt`, `original_paper.pdf`, `post_registration.json`, `replication_data.csv` |
| Medium | `initial_details_medium_hard.txt`, `original_paper.pdf`, `post_registration.json` |
| Hard | `initial_details_medium_hard.txt`, `original_paper.pdf`, `post_registration.json` |

### Validator Module

This module validates whether the metadata extracted by the info extractor matches what is expected based on human-annotated metadata.
- We use an LLM (GPT4o) to compare the extracted info (`extracted_json.json`) against the human-annotated ground-truth (`expected_json.json`).
- We use the same proposed evaluation rubrics in the task design as prompt to the LLM-as-judge and ask it to assign a score for the extracted info (can be found under `templates/prompts/extract_eval.txt`.

```bash
python evaluate_extract_info.py \
  --extracted_json_path path/to/extracted_json.json \
  --expected_json_path path/to/expected_json.json \
  --output_path path/to/study_dir/llm_eval.json
```
##### Output
* JSON formatted evaluation of the extracted metadata
* prompt for traceability (`logs/` directory)

## 🔐 Access and Permissions

This repository is managed under the **COS GitHub organization**, with:

* **Admin access** retained by COS staff
* **Write or maintain access** granted to approved external collaborators


## 📄 License

All content in this repository is shared under the [Apache License 2.0](LICENSE)

## 👥 Contributors

Core team members from COS, plus external partners from Old Dominion University, Pennsylvania State University, and University of Notre Dame  specializing in:

* Agent development
* Benchmark design
* Open Science Research

## 📬 Contact

For questions please contact:

**Shakhlo Nematova**
Research Scientist
[shakhlo@cos.io](mailto:shakhlo@cos.io)

---
