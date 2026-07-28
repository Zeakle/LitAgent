"""Define the LitAgent exception hierarchy."""


class LitAgentError(Exception):
    """Base exception for LitAgent failures."""

    pass


class ConfigError(LitAgentError):
    """Report configuration loading or validation failures."""

    pass


class MCPError(LitAgentError):
    """Report MCP connection or invocation failures."""

    pass


class SafetyError(LitAgentError):
    """Report safety-policy violations."""

    pass
