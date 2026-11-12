"""Validate tool calls before graph execution."""

from litagent.logging import get_logger
from litagent.tools.registry import ToolRegistry

logger = get_logger("agent_validation")

_REQUIRED_FIELDS = {"name", "args"}


def validate_tool_call(
    action: dict,
    graph_tool_names: set[str] | None = None,
    registry: ToolRegistry | None = None,
) -> dict:
    """Validate a pending tool call against its shape and available tools."""
    if registry is None and graph_tool_names is None:
        # Compatibility path for callers that have not supplied graph-local tools.
        from litagent.tools.registry import get_registry

        registry = get_registry()
    errors = []

    for field in _REQUIRED_FIELDS:
        if field not in action:
            errors.append(f"Missing required field: {field}")

    if action.get("name", "") == "":
        errors.append(f"Tool name must not be empty")

    if "args" in action and not isinstance(action["args"], dict):
        errors.append("Tool args must be a dict")

    name = action.get("name", "")
    in_registry = registry is not None and len(registry) > 0 and name in registry
    in_graph = graph_tool_names is not None and name in graph_tool_names
    # An explicitly supplied empty graph set is still a policy: no tools are
    # available. Only the legacy no-policy call may fall back to permissive shape
    # validation when the global compatibility registry is empty.
    has_known_tools = registry is not None or graph_tool_names is not None
    if has_known_tools and not in_registry and not in_graph:
        errors.append(f"Tool {action.get('name', '')} not registered")

    if errors:
        logger.warning(f"ToolCall validation failed: {errors}")
        return {
            "_validation_result": {
                "valid": False,
                "errors": errors,
            }
        }

    return {"_validation_result": {"valid": True, "errors": []}}
