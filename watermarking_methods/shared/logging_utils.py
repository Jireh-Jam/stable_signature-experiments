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


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
