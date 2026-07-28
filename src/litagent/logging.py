"""Configure and retrieve namespaced LitAgent loggers."""

import logging

from litagent.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure the process-wide logging level and format."""
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the LitAgent namespace."""
    return logging.getLogger(f"litagent.{name}")
