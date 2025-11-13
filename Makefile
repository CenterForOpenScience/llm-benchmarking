SHELL := /bin/bash
export PYTHONPATH := .
PYTHON ?= python3

# libs we need before running anything
REQ := pytest pytest_cov openai dotenv pymupdf pyreadr pandas numpy docker docx

STUDY ?= ./data/processed/1

.PHONY: check-deps install-dev test test-extractor test-generator test-all design-easy execute-easy

check-deps:
	$(PYTHON) core/check_deps.py $(REQ)

install-deps:
	pip install -r requirements-dev.txt

check-docker:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not in PATH"; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable (is it running? do you have permissions?)"; exit 1; }
	@echo "Docker: OK"

# extractor
extract-stage1: check-deps
	python -m info_extractor --stage stage_1 --difficulty easy --study-path $(STUDY)

extract-stage2: check-deps
	python -m info_extractor --stage stage_2 --difficulty easy --study-path $(STUDY)

# validator module
validate-info: check-deps
	python -m validator.cli.validate_info_extractor --study_dir $(STUDY) --results_file info_extractor_validation_results.json --show-mismatches

extract-human-info: check-deps
	python -m validator.cli.extract_human_replication_info --preregistration $(STUDY)/prereg.docx --score_report $(STUDY)/report.pdf --output_path $(STUDY)/replication_info_expected.json

evaluate-human-info: check-deps
	python -m validator.cli.evaluate_extracted_info --extracted_json_path $(STUDY)/replication_info.json --expected_json_path $(STUDY)/replication_info_expected.json --output_path $(STUDY)
	
extract-results: check-deps
	python -m validator.cli.extract_replication_results --study_path $(STUDY)

evaluate-execute: check-deps
	python -m validator.cli.evaluate_execute_cli  --study_path $(STUDY)

evaluate-interpret: check-deps
	python -m validator.cli.evaluate_interpret_cli  --study_path $(STUDY) --reference_report_path $(STUDY_HUMAN_REPORT)

# generator module
design-easy: check-deps
	python -m generator --stage design --tier easy --study-path $(STUDY) --templates-dir ./templates
execute-easy: check-deps check-docker
	python -m generator --stage execute --tier easy --study-path $(STUDY)

generate: design-easy execute-easy

# interpreter module
interpret-easy: check-deps
	python -m interpreter --tier easy --study-path $(STUDY)

# full pipeline (extract -> design -> execute -> interpret)
pipeline-easy: extract-stage1 design-easy execute-easy interpret-easy

# test suite
test: check-deps
	pytest -q

test-extractor: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=info_extractor.extractor --cov-report=term-missing

test-generator: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=generator.design.easy --cov-report=term-missing

test-cov: check-deps
	pytest -q --cov=.

# run the whole suite (extractor + validator + generator) with test coverage
test-all: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=term-missing
