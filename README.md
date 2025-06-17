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
├── templates/
│   ├── replication_info_schema.json
│   ├── info_extractor_stage1_instructions.json
│   └── info_extractor_stage2_instructions.json
│
├── samples/
│   ├── initial_details_easy.txt
│   └── initial_details_medium_hard.txt
│
├── constants.py
├── main.py
└── README.md

```

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/CenterForOpenScience/llm-benchmarking.git
   cd llm-benchmarking
   ```

2. Install dependencies:
   ```bash
   pip install openai pymupdf pandas
   ```

3. Configure API key in `constants.py`

## Usage
```bash
# Run extraction phase only
python main.py --study_path ./studies/example --stage 1 --difficulty medium
python main.py --study_path ./studies/example --stage 2 --difficulty medium
```

**Arguments:**
* `--study_path`: Path to study folder
* `--stage`: "1" (original) or "2" (replication)
* `--difficulty`: "easy", "medium", or "hard"
* `--show-prompt`: Print prompt

## Output Files
* Stage 1: `replication_info_stage1.json`
* Stage 2: `replication_info.json`

## File Requirements

### Stage 1
| Difficulty | Required Files |
|------------|----------------|
| Easy       | `initial_details_easy.txt`, `original_paper.pdf`, `data_description.txt`|
| Medium     | `initial_details_medium_hard.txt`, `original_paper.pdf` |
| Hard       | `initial_details_medium_hard.txt`, `original_paper.pdf` |

### Stage2
| Difficulty | Required Files |
|------------|----------------|
| Easy       | `initial_details_easy.txt`, `original_paper.pdf`, `replication_info_stage1.json`, `replication_data.csv` |
| Medium     | `initial_details_medium_hard.txt`, `original_paper.pdf`, `replication_info_stage1.json` |
| Hard       | `initial_details_medium_hard.txt`, `original_paper.pdf`, `replication_info_stage1.json` |


## 🔐 Access and Permissions

This repository is managed under the **COS GitHub organization**, with:

* **Admin access** retained by COS staff
* **Write or maintain access** granted to approved external collaborators


## 📄 License

All content in this repository is shared under <placeholder for LICENCE>

## 👥 Contributors

Core team members from COS, plus external partners specializing in:

* Agent development
* Benchmark design
* Reproducibility and evaluation

## 📬 Contact

For questions please contact:

**Shakhlo Nematova**
Research Scientist
[shakhlo@cos.io](mailto:shakhlo@cos.io)

---
