# Validator Module

The **Validator** module supports automated verification of metadata extracted by LLMs agents via info_extractor module. Currenltly, it consists of two stages:

1. **Extract Expected Values**: Extract the *expected* information (`replication_info_expected.json`) by prompting a language model (GPT-4o) using human-written replication documents (pre-registration and SCORE reports).
2. **Validate info_extractor Output**: Compare the expected output (`replication_info_expected.json`) with the actual output from the info_extractor module (`replication_info.json`), and report matching/mismatching fields (`info_exractor_validation_results.json`).

---

## Structure

```
validator/
├── extract_from_human_replication_study.py  # Core logic for stage 1: Extract Expected Values
├── compare_outputs.py                       # Core logic for stage 2: Validate info_extractor Output
├── __init__.py                              # (optional)
```

---

## Stage 1: Extract Expected Values

Generates a replication_info_expected.json from two human-written documents:

### Inputs:
- `--preregistration`: Path to the pre-registration document (PDF or DOCX)
- `--score_report`: Path to the SCORE or replication report (PDF or DOCX)
- `--output_path`: Output file path for the expected JSON

### Command:
```bash
python extract_human_replication_info.py \
  --preregistration data/prereg.pdf \
  --score_report data/score_report.docx \
  --output_path study_dir/replication_info_expected.json
```

## Stage 2: Validate info_extractor Output

Compares `replication_info.json` (from info_extractor module) with `replication_info_expected.json` (from validator module Stage 1).

### Input:
* `--study_dir`: Directory containing both JSON files (`replication_info_expected.json` and `replication_info_expected.json`)
* `--results_file`: File name for json formatted summary of matched and mismatched fields (ex: `info_exractor_validation_results.json`)

### Command:
```bash
python validate_info_extractor.py --study_dir "data/study_dir" --results_file "info_exractor_validation_results.json"
```

### Output:
* JSON formatted report listing:
   * Matching fields
   * Mismatched fields

## Notes

* Prompts and completions are logged to `logs/` directory automatically.
