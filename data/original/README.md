---
pretty_name: "LLM Benchmarking Project — Scientific Replication Benchmark Data"
license: apache-2.0
language:
  - en
tags:
  - open-science
  - replication
  - benchmark
  - llm
  - information-extraction
  - research-design
  - evaluation
---

# LLM Benchmarking Project — Dataset (Scientific Replication Benchmark)

This repository contains the **data-only** portion of the Center for Open Science (COS) **LLM Benchmarking Project**. The dataset supports benchmarking LLM agents on core parts of the scientific research lifecycle—especially **replication**—including:

- **Information extraction** from scientific papers into structured JSON  
- **Research design** and analysis planning  
- **(Optional) execution support** using provided replication datasets and code  
- **Scientific interpretation** using human reference materials and expected outputs  

Each numbered folder corresponds to **one study instance** in the benchmark.

## Dataset contents (per study)

Each study folder typically contains:

- `original_paper.pdf`  
  The published paper used as the primary input.

- `initial_details.txt`  
  Brief notes to orient the replication attempt (e.g., key outcomes, hints, pointers).

- `replication_data/`  
  Data and scripts required to reproduce analyses (common formats: `.csv`, `.dta`, `.rds`, `.R`, `.do`, etc.).

- `human_preregistration.(pdf|docx)`  
  Human-created preregistration describing the replication plan.

- `human_report.(pdf|docx)`  
  Human-created replication report describing analyses and findings.

- `expected_post_registration*.json`  
  Expert-annotated ground truth structured outputs used for evaluation.  
  - `expected_post_registration.json` is the primary reference.  
  - `expected_post_registration_2.json`, `_3.json`, etc. are acceptable alternative variants where applicable.

Some studies include multiple acceptable ground-truth variants to capture permissible differences in annotation or representation.

## Repository structure

At the dataset root, folders like `1/`, `2/`, `10/`, `11/`, etc. are **study IDs**.

Example:

```
text
.
├── 1/
│   ├── expected_post_registration.json
│   ├── expected_post_registration_2.json
│   ├── human_preregistration.pdf
│   ├── human_report.pdf
│   ├── initial_details.txt
│   ├── original_paper.pdf
│   └── replication_data/
│       ├── <data files>
│       └── <analysis scripts>
```

## Intended uses

This dataset is intended for:

- Benchmarking LLM agents that **extract structured study metadata** from papers  
- Evaluating LLM systems that generate **replication plans** and analysis specifications  
- Comparing model outputs against **expert-annotated expected JSON** and human reference docs  

## Not intended for

- Clinical or other high-stakes decision-making  
- Producing definitive judgments about the original papers  
- Training models to reproduce copyrighted texts verbatim  

## Quickstart (local)

### Iterate over studies and load ground truth

```
python
from pathlib import Path
import json

root = Path(".")
study_dirs = sorted(
    [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)

for study in study_dirs:
    gt = study / "expected_post_registration.json"
    if gt.exists():
        data = json.loads(gt.read_text(encoding="utf-8"))
        print(study.name, "ground truth keys:", list(data.keys())[:10])
```

## Using with the main pipeline repository (recommended)

If you are using the **LLM Benchmarking Project** codebase, point the pipeline/evaluators at a given study directory:

```
bash
make evaluate-extract STUDY=/path/to/llm-benchmarking-data/1
```

The expected JSON format is defined by the main repository’s templates/schemas. Use those schemas to validate or format model outputs.

## Notes on multiple expected JSON variants

Some studies include `expected_post_registration_2.json`, `expected_post_registration_3.json`, etc. This is intentional:

- Some fields allow multiple equivalent representations  
- Annotation may vary slightly without changing meaning  
- Evaluators may accept any variant depending on scoring rules  

If you implement your own scorer, consider:
- Exact matching for strictly defined fields  
- More tolerant matching for lists, notes, or fields with legitimate paraphrase/format variation  

## File formats

You may encounter:

- R artifacts: `.R`, `.rds`  
- Stata artifacts: `.do`, `.dta`  
- CSV/tabular data: `.csv`  
- Documents: `.pdf`, `.docx`  
- Structured evaluation targets: `.json`  

Reproducing analyses may require R and/or Stata depending on the study.

## Licensing, copyright, and redistribution (important)

This repository is released under **Apache 2.0** for **COS-authored materials and annotations** (for example: benchmark scaffolding, expected JSON outputs, and other COS-created files).

However, some contents may be **third-party materials**, including (but not limited to):

- `original_paper.pdf` (publisher copyright may apply)  
- `replication_data/` (may have its own license/terms from the original authors)  
- external scripts or datasets included for replication  

**You are responsible for ensuring you have the right to redistribute third-party files publicly** (e.g., GitHub / Hugging Face).

Common options if redistribution is restricted:
- Remove third-party PDFs and provide **DOI/URL references** instead  
- Keep restricted files in a private location and publish only COS-authored annotations  
- Add per-study `LICENSE` / `NOTICE` files inside each study folder where terms are known  

## Large files (Git LFS recommendation)

If hosting on GitHub, consider Git LFS for PDFs and large datasets:

```
bash
git lfs install
git lfs track "*.pdf" "*.dta" "*.rds"
git add .gitattributes
```

## Citation

If you use this dataset in academic work, please cite it as:

```
bibtex
@dataset{cos_llm_benchmarking_data_2026,
  author    = {Center for Open Science},
  title     = {LLM Benchmarking Project: Scientific Replication Benchmark Data},
  year      = {2026},
  publisher = {Center for Open Science},
  note      = {Benchmark dataset for evaluating LLM agents on scientific replication tasks}
}
```

## Acknowledgements

This project is funded by Coefficient Giving as part of its “Benchmarking LLM Agents on Consequential Real-World Tasks” program. We thank the annotators who contributed to the ground-truth post-registrations for the extraction stage.

## Contact

For questions about this dataset:

**Shakhlo Nematova**  
Research Scientist, Center for Open Science  
shakhlo@cos.io
