import os
import pathlib

from logger import get_logger


def test_get_logger_creates_log_file_and_reuses_instance(tmp_path, monkeypatch):
    # Work in isolated temp dir so repo files aren’t touched
    monkeypatch.chdir(tmp_path)
    # Direct log file name via env
    monkeypatch.setenv("LOG_FILE", "unit.log")

    # First retrieval should create handler and file on first write
    logger1 = get_logger("unit-test")
    assert logger1.name == "unit-test"

    logger1.info("hello")

    log_path = pathlib.Path("logs") / "unit.log"
    assert log_path.exists(), "Expected log file to be created in logs/"

    # Subsequent calls return the same instance (no duplicate handlers)
    logger2 = get_logger("unit-test")
    assert logger1 is logger2
    # Ensure only one handler is attached
    assert len(logger2.handlers) == 1

