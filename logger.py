# -*- coding: utf-8 -*-
"""
LLM_Benchmarking__
|
logger.py
Created on Thu Jul  3 00:22:30 2025
@author: Rochana Obadage
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_loggers = {}

def get_logger(name="info_extractor"):
    if name in _loggers:
        return _loggers[name]

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Read log file name from ENV, fallback to run.log
    log_file = os.environ.get("LOG_FILE", "run.log")
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _loggers[name] = logger
    return logger

