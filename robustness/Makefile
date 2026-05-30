SHELL := /bin/bash
export PYTHONPATH := .
PYTHON ?= python3

# libs we need before running anything
REQ := pytest pytest_cov openai dotenv pymupdf pyreadr pandas numpy docker docx

.PHONY: check-deps install-dev test test-extractor test-generator test-all design-easy execute-easy

check-deps:
	$(PYTHON) ../replicatorbench/core/check_deps.py $(REQ)

install-deps:
	pip install -r requirements-dev.txt


check-docker:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not in PATH"; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable (is it running? do you have permissions?)"; exit 1; }
	@echo "Docker: OK"

execute:
	$(PYTHON) 
