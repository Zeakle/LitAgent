"""LitAgent — Multi-agent adversarial literature review framework."""

from litagent.config import load_config, AppConfig
from litagent.runner import LitAgent

__version__ = "0.1.0"
__all__ = ["LitAgent", "load_config", "AppConfig"]