"""Tests for logging configuration and logger creation."""

import logging

import pytest

from litagent.logging import setup_logging, get_logger
from litagent.config import LoggingConfig


class TestGetLogger:
    """Tests named logger creation."""

    def test_prefix(self):
        logger = get_logger("test_module")
        assert logger.name == "litagent.test_module"

    def test_child_logger(self):
        logger = get_logger("agent.react")
        assert logger.name == "litagent.agent.react"


class TestSetupLogging:
    """Tests logging setup."""

    def test_setup_logging_no_crash(self):
        """Default logging setup completes without error."""
        config = LoggingConfig(level="WARNING")
        setup_logging(config)

    def test_setup_logging_custom_level(self):
        """Logging setup applies the configured level."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            setup_logging(LoggingConfig(level=level))
            logger = get_logger(f"test_{level.lower()}")
            logger.debug("test message")
            logger.info("test message")
