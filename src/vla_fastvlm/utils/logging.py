
import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> None:
    logger = logging.getLogger(name)
    handler_exists = any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    if not handler_exists:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
