"""ToolCall 参数校验。

Phase 2 用规则做校验（检查必须字段），Phase 3 迁移到 Pydantic。
"""


from litagent.logging import get_logger

logger = get_logger('agent_validation')

_REQUIRED_FIELDS = {"name", "args"}


def validate_tool_call(action: dict) -> dict:
    """校验 tool_call 的格式是否合法。

    Args:
        action: 待校验的 tool_call，格式 {"name": str, "args": dict, "id": str}

    Returns:
        {"_validation_result": {"valid": bool, "retry_count": int, "errors": list[str]}}
    """
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

    if errors:
        logger.warning(f"ToolCall validation failed: {errors}")
        # _retry_count > 3放弃工具执行
        action.setdefault("_retry_count", 0)
        return {
            "_validation_result": {
                'valid': False,
                'retry_count': action["_retry_count"],
                'errors': errors,
            }
        }

    return {'_validation_result': {"valid": True}}