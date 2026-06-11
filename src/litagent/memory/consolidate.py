"""Consolidate: Working Memory → Episodic Memory 提升。

会话结束时调用。Phase 4 用规则提取结构化摘要。
Phase 5 升级为 LLM 驱动的完整摘要（key_findings / tools_used / importance）。
"""

from langchain_core.messages import HumanMessage

from litagent.memory.models import Episode
from litagent.logging import get_logger


logger = get_logger('memory.consolidate')


async def consolidate_session(state: dict, session_id: str) -> Episode | None:
    """从 session state 提取 Episode"""
    messages = state.get('messages', [])
    if len(messages) < 2:
        return None

    user_query = _extract_user_query(messages)
    msg_count = len(messages)

    return Episode(
        summary=f"User asked: '{user_query}' ({msg_count} messages exchanged)",
        intent=_classify_intent(user_query),
        session_id=session_id,
    )


def _extract_user_query(messages: list) -> str:
    for msg in messages:
        # 处理两种形态: LangChain Message 对象 / Redis 反序列化的 dict
        if isinstance(msg, dict):
            if msg.get("type") == "human":
                return msg.get("content", "")
        elif isinstance(msg, HumanMessage):
            return msg.content
        elif hasattr(msg, "type") and msg.type == "human":
            return getattr(msg, "content", "")
    return "unknown query"


def _classify_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["survey", "review", "综述", "文献"]):
        return "literature_review"
    if any(w in q for w in ["compare", "对比"]):
        return "comparison"
    if any(w in q for w in ["find", "search", "找"]):
        return "search"
    return "general"