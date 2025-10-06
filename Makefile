SHELL := /bin/bash
export PYTHONPATH := .
PYTHON ?= python3

# libs we need before running anything
REQ := pytest pytest_cov openai dotenv pymupdf pyreadr pandas numpy docker docx

.PHONY: check-deps install-dev test test-extractor test-generator test-all design-easy execute-easy

check-deps:
	$(PYTHON) check_deps.py $(REQ)

install-deps:
	pip install -r requirements-dev.txt

check-docker:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not in PATH"; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable (is it running? do you have permissions?)"; exit 1; }
	@echo "Docker: OK"

# extractor
extract-stage1: check-deps
	python -m info_extractor --stage stage_1 --difficulty easy --study-path ./case_studies/case_study_1

extract-stage2: check-deps
	python -m info_extractor --stage stage_2 --difficulty easy --study-path ./case_studies/case_study_1

# validator module
validate-info: check-deps
	python -m validator.cli.validate_info_extractor --study_dir ./case_studies/case_study_3 --results_file info_extractor_validation_results.json --show-mismatches

extract-human-info: check-deps
	python -m validator.cli.extract_human_replication_info --preregistration ./case_studies/case_study_5/prereg.docx --score_report ./case_studies/case_study_5/report.pdf --output_path ./case_studies/case_study_5/replication_info_expected.json

evaluate-human-info: check-deps
	python -m validator.cli.evaluate_extracted_info --extracted_json_path ./case_studies/case_study_5/replication_info.json --expected_json_path ./case_studies/case_study_5/replication_info_expected.json --output_path ./case_studies/case_study_5

extract-results: check-deps
	python -m validator.cli.extract_replication_results --study_path ./case_studies/case_study_5

# generator module
design-easy: check-deps
	python -m generator --stage design --tier easy --study-path ./case_studies/case_study_4 --templates-dir ./templates
execute-easy: check-deps check-docker
	python -m generator --stage execute --tier easy --study-path ./case_studies/case_study_4

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
