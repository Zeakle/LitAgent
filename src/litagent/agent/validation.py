"""ToolCall 参数校验。

Phase 2 用规则做校验（检查必须字段），Phase 3 迁移到 Pydantic。
"""


from litagent.logging import get_logger


logger = get_logger('agent_validation')

_REQUIRED_FIELDS = {"name", "args"}


def validate_tool_call(action: dict, graph_tool_names: set[str] | None = None) -> dict:
    """校验 tool_call 的格式是否合法。"""
    from litagent.tools.registry import get_registry
    registry = get_registry()
    errors = []

    # 检查必须字段
    for field in _REQUIRED_FIELDS:
        if field not in action:
            errors.append(f'Missing required field: {field}')
        
    # 检查name是否非空
    if action.get('name', "") == "":
        errors.append(f'Tool name must not be empty')

    # 检查args是否为dict
    if 'args' in action and not isinstance(action['args'], dict):
        errors.append('Tool args must be a dict')

    # 检测工具是否注册 (Registry or ReAct graph ToolNode)
    name = action.get('name', "")
    in_registry = len(registry) > 0 and name in registry
    in_graph = graph_tool_names is not None and name in graph_tool_names
    if len(registry) > 0 and not in_registry and not in_graph:
        errors.append(f"Tool {action.get('name', '')} not registered")

    if errors:
        logger.warning(f"ToolCall validation failed: {errors}")
        return {
            "_validation_result": {
                'valid': False,
                'errors': errors,
            }
        }

    return {'_validation_result': {"valid": True, "errors": []}}