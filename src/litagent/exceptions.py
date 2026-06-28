class LitAgentError(Exception):
    """LitAgent 所有异常的基类。

    继承链的设计原因:
    - 调用方只关心 LitAgentError → catch 一个就行
    - 需要精确处理时 catch 子类 → try: ... except ConfigError: ...
    """
    pass


class ConfigError(LitAgentError):
    """配置加载或校验失败。"""
    pass


class ToolError(LitAgentError):
    """工具执行失败（Phase 3 开始使用）。"""
    pass


class AgentError(LitAgentError):
    """Agent 执行失败（Phase 2 开始使用）。"""
    pass


class MemoryStoreError(LitAgentError):
    """Memory 层操作失败（Phase 4 开始使用）。不用 MemoryError 因为会覆盖 Python 内置 OOM 异常。"""
    pass


class MCPError(LitAgentError):
    """MCP 连接或调用失败"""
    pass