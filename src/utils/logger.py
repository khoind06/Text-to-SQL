# Centralized Logging Setup

import sys
from loguru import logger

def setup_logger(log_file: str = "logs/app.log"):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    return logger
