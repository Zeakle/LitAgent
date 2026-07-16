import logging
import pytest
from litagent.logging import setup_logging, get_logger
from litagent.config import LoggingConfig


class TestGetLogger:
    def test_prefix(self):
        logger = get_logger("test_module")
        assert logger.name == "litagent.test_module"

    def test_child_logger(self):
        logger = get_logger("agent.react")
        assert logger.name == "litagent.agent.react"


class TestSetupLogging:
    def test_setup_logging_no_crash(self):
        """setup_logging 正常执行不报错"""
        config = LoggingConfig(level="WARNING")
        setup_logging(config)  # 不抛异常即通过

    def test_setup_logging_custom_level(self):
        """传入不同 level 不报错"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            setup_logging(LoggingConfig(level=level))
            logger = get_logger(f"test_{level.lower()}")
            logger.debug("test message")
            logger.info("test message")
