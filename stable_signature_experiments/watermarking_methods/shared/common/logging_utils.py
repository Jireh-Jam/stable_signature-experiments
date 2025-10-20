"""Lightweight logging helpers.

Usage:
    from common.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""
from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """Create a configured logger.

    Args:
        name: Logger name (usually __name__). If None, root logger.
        level: Optional level name (e.g., "INFO", "DEBUG"). If not provided,
               reads LOG_LEVEL env var or defaults to INFO.

    Returns:
        A configured logger with a simple, concise formatter.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
