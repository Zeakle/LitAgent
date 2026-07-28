"""Tests for the LitAgent exception hierarchy."""

import pytest

from litagent.exceptions import LitAgentError, ConfigError, MCPError, SafetyError


class TestExceptionHierarchy:
    """Tests inheritance and messages for project exceptions."""

    def test_all_inherit_from_base(self):
        assert issubclass(ConfigError, LitAgentError)
        assert issubclass(MCPError, LitAgentError)
        assert issubclass(SafetyError, LitAgentError)

    def test_catch_by_parent(self):
        with pytest.raises(LitAgentError):
            raise ConfigError("bad config")
        with pytest.raises(ConfigError):
            raise ConfigError("bad config")

    def test_error_message(self):
        err = ConfigError("missing field: agent.max_loops")
        assert "missing field" in str(err)
