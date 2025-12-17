"""
Logging configuration module for the RAG Retrieval Verification system.

This module sets up logging for the verification application.
"""
import logging
import sys
from typing import Optional
from .config import get_config


def setup_logging(level: Optional[str] = None) -> None:
    """
    Setup logging configuration for the verification application.

    Args:
        level: Logging level to use (defaults to LOG_LEVEL from config)
    """
    config = get_config()
    log_level = level or config.log_level or 'INFO'

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Create file handler (optional, can be configured via environment)
    log_file = None  # This could be configured via environment variable
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
    else:
        file_handler = None

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    if file_handler:
        file_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    logging.info("Logging configuration completed")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Initialize logging when module is imported
setup_logging()