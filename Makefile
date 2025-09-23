SHELL := /bin/bash
export PYTHONPATH := .
PYTHON ?= python3

# libs we need before running anything
REQ := pytest pytest_cov openai dotenv pymupdf pyreadr pandas numpy

.PHONY: check-deps install-dev test test-extractor test-generator test-all design-easy execute-easy

check-deps:
	$(PYTHON) check_deps.py $(REQ)

install-deps:
	pip install -r requirements-dev.txt

extract-stage1: check-deps
	python -m info_extractor --stage stage_1 --difficulty easy --study-path ./case_studies/case_study_3

extract-stage2: check-deps
	python -m info_extractor --stage stage_2 --difficulty easy --study-path ./case_studies/case_study_3

design-easy: check-deps
	python -m generator --stage design --tier easy --study-path ./case_studies/case_study_12 --templates-dir ./templates
execute-easy: check-deps
	python -m generator --stage execute --tier easy --study-path ./case_studies/case_study_1

# tests
test: check-deps
	pytest -q

test-extractor: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=info_extractor.extractor --cov-report=term-missing

test-generator: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=generator.design.easy --cov-report=term-missing

# Run the whole suite (extractor + generator) with coverage
test-all: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=info_extractor.extractor --cov=generator.design.easy --cov-report=term-missing
