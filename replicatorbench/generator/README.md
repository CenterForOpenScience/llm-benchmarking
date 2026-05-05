# Generate Module

Implements the two-stage **Generate** phase for the Replicator:
1) **Design**: builds a preregistration from Stage-1 extractions.
2) **Execute**: runs the preregistration on the replication dataset/code.

## Repository Structure
```bash
llm-benchmarking/
├── case_studies/
│ ├── case_study_1/
│ ├── case_study_3/
│ └── ...
├── templates/
│ └── post_registration_schema.json
├── generate/
│ ├── README.md
│ ├── __init__.py
│ ├── __main__.py # unified CLI
│ ├── common/
│ │ ├── __init__.py
│ │ ├── io.py # small helpers for read_json/write_json/read_text
│ │ ├── logging.py # setup_logger(...)
│ │ └── schema_utils.py # load/blank schema, fallback_build, etc.
│ ├── design/
│ │ ├── __init__.py
│ │ └── easy.py 
│ └── execute/
│   ├── __init__.py
│   └── easy.py # placeholder stub
```

## Inputs (Design/Easy)
- `post_registration.json` (Stage-1 output — authoritative for `original_study`)
- `replication_info.json` (hints about replication dataset/tools)
- `info_exractor_validation_results.json`
- `inputs/initial_details_easy.txt`
- `templates/post_registration_schema.json` (schema for `original_study`)

## Outputs (Design/Easy)
- `preregistration_design.json` with:
  - `original_study` (exact keys from the template)
  - `Replication` (hypothesis, data_plan, planned_method, planned_estimation_and_test)

## Run
```bash
make design-easy
make execute-easy
```

### What You Should See
```bash
case_study_1/
  preregistration_design.json
  _logs/
    design_easy.log
    design_easy_prompt.txt
    design_easy_message.txt
```
