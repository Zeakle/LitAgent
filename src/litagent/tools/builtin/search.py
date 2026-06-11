"""学术搜索工具——Phase 6 RAG 完成后实现真正的 arxiv/Semantic Scholar 搜索。"""

from litagent.tools.base import ToolDefinition, ToolCategory


def search_arxiv(query: str, max_results: int = 10) -> str:
    """[Placeholder] Search arxiv."""
    return f"[arxiv] Found 0 results for '{query}' (not implemented yet)"


search_arxiv_definition = ToolDefinition(
    name="search_arxiv",
    description="Search arxiv for papers matching the query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results", "default": 10},
        },
        "required": ["query"]
    },
    category=ToolCategory.READ,
    timeout_ms=15000,
    max_retries=2,
    version="1.0.0",
)