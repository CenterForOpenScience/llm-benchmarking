# Validator Module

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
