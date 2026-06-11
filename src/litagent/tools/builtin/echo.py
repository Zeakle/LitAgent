"""Echo 工具——用于测试 Tool System 集成。Phase 6 替换为真实搜索工具。"""


from litagent.tools.base import ToolDefinition, ToolCategory


def echo_tool(message: str) -> str:
    """返回输入的message并加上前缀"""
    return f"Echo: {message}"


echo_definition = ToolDefinition(
    name='echo',
    description='Echo back the input message',
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to echo back"
            }
        },
        "required": ["message"]
    },
    category=ToolCategory.READ,
    timeout_ms=5000,
    max_retries=1,
    cache_ttl_ms=0,
    version='1.0.0',
)