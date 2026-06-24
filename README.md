# LLM Benchmarking Project

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Center for Open Science](https://img.shields.io/badge/Organization-COS-green)](https://cos.io)

Welcome to the official repository for the **LLM Benchmarking Project**, led by the Center for Open Science (COS). This project provides a modular framework to evaluate the capabilities of large language model (LLM) agents across key components of the scientific research lifecycle, including **replication**, **robustness**, and **research design**.
It contains the **ReplicatorBench** **ReplicatorAgent** published preprint on [arxiv](https://arxiv.org/abs/2602.11354) (to appear in KDD 2026 AI4Sciences track).

## 🔍 ReplicatorBench Overview

### Core Capabilities
* **Information Extraction:** Automated extraction of structured metadata from PDFs and data files.
* **Research Design:** LLM-driven generation of replication plans and analysis scripts.
* **Execution & Sandboxing:** Secure execution of generated code within Docker environments.
* **Scientific Interpretation:** Synthesis of statistical results into human-readable research reports.
* **Automated Validation:** An **LLM-as-judge** system that benchmarks agent performance against expert-annotated ground truths.

This work builds on the conceptual structure outlined in our Open Philanthropy grant, emphasizing real-world relevance, task diversity, and community participation.

---


## 📂 Project Structure (ongoing)
```
llm-benchmarking/
├── replicatorbench/
|   ├── core/            # Central logic containing autonomous agent, tools, prompts, and actions.
|   ├── info_extractor/  # PDF parsing and metadata extraction
|   ├── generator/       # Research design and code generation
|   ├── interpreter/     # Result analysis and report generation
|   ├── validator/       # CLI tools for LLM-based evaluation
|   ├── templates/       # JSON schemas and prompt templates
|   ├── data/            # Benchmark datasets and ground truth
|   ├── README.md
|   └── Makefile            # Project automation
├── robustness
|   ├── Makefile
|   └── requirements-dev.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md (this file)

```

--- 

## 📄 License

All content in this repository is shared under the [Apache License 2.0](LICENSE)

## 👥 Contributors

Core team members from COS, plus external partners from Old Dominion University, Pennsylvania State University, and University of Notre Dame  specializing in:

* Agent development
* Benchmark design
* Open Science Research

## Acknowledgement
This project is funded by Coefficient Giving as part of its 'Benchmarking LLM Agents on Consequential Real-World Tasks' program. We thank Anna Szabelska, Adam	Gill, and Ahana Biswas for their annotation of the ground-truth post-registrations for the extraction stage.

## 📬 Contact

For questions please contact:

**Shakhlo Nematova**
Research Scientist
[shakhlo@cos.io](mailto:shakhlo@cos.io)

---
