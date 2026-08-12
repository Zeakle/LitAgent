"""Consolidate session state into reusable episodic memory."""

import json

from langchain_core.messages import HumanMessage

from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger
from litagent.memory.models import Episode

logger = get_logger("memory.consolidate")


async def consolidate_session(
    state: dict, session_id: str, llm: BaseLLMClient | None = None
) -> Episode | None:
    """Convert a session into an episode, falling back to deterministic rules."""
    messages = state.get("messages", [])
    if len(messages) < 2:
        return None

    if llm:
        try:
            return await _llm_consolidate(llm, messages, session_id)
        except Exception as e:
            # Keep consolidation available when the LLM or its JSON output fails.
            logger.warning(f"LLM consolidate failed: {e}, falling back to rule-based")
    return _rule_consolidate(messages, session_id)


async def _llm_consolidate(
    llm: BaseLLMClient, messages: list, session_id: str
) -> Episode:
    """Build a structured episode from session messages with an LLM."""
    conversation = _format_messages(messages)

    prompt = f"""Analyze this research session and return a JSON object in the \
format shown below.

Example format:
{{
    "summary": "User did a literature review on few-shot learning, \
finding 47 papers and identifying ProtoNet as SOTA.",
    "intent": "literature_review",
    "key_findings": ["ProtoNet is SOTA on miniImageNet at 93.2%", \
"MAML dominates 1-shot scenarios"],
    "tools_used": ["search_arxiv", "search_semantic_scholar", "extract_claims"],
    "errors_encountered": ["PapersWithCode API timed out"],
    "importance_score": 0.8,
    "extracted_facts": [
        {{"key": "few_shot_sota", "value": \
{{"model": "ProtoNet", "accuracy": "93.2%"}}, "type": \
"domain_knowledge", "confidence": 0.85}}
    ]
}}

Return ONLY valid JSON, no other text.

Conversation:
{conversation}"""

    resp = await llm.chat(
        [
            {
                "role": "system",
                "content": "You are a memory consolidation system. Output only JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed = json.loads(resp.content)

    return Episode(
        summary=parsed.get("summary", ""),
        intent=parsed.get("intent", "general"),
        key_findings=parsed.get("key_findings", []),
        tools_used=parsed.get("tools_used", []),
        errors_encountered=parsed.get("errors_encountered", []),
        importance_score=float(parsed.get("importance_score", 0.5)),
        extracted_facts=parsed.get("extracted_facts", []),
        session_id=session_id,
    )


def _rule_consolidate(messages: list, session_id: str) -> Episode:
    """Build a minimal episode from session messages without an LLM."""
    user_query = _extract_user_query(messages)
    msg_count = len(messages)
    return Episode(
        summary=f"User asked: '{user_query}' ({msg_count} messages exchanged)",
        intent=_classify_intent(user_query),
        session_id=session_id,
    )


def _format_messages(messages: list) -> str:
    """Render supported message objects as a role-prefixed transcript."""
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("type", "unknown")
            content = msg.get("content", "")
        elif isinstance(msg, HumanMessage):
            role, content = "user", msg.content
        elif hasattr(msg, "type"):
            role, content = msg.type, getattr(msg, "content", "")
        else:
            continue
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _extract_user_query(messages: list) -> str:
    """Extract the first user query from session messages."""
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("type") == "human":
                return msg.get("content", "")
        elif isinstance(msg, HumanMessage):
            return msg.content
        elif hasattr(msg, "type") and msg.type == "human":
            return getattr(msg, "content", "")
    return "unknown query"


def _classify_intent(query: str) -> str:
    """Classify the query into a stable intent category."""
    q = query.lower()
    if any(w in q for w in ["survey", "review", "综述", "文献"]):
        return "literature_review"
    if any(w in q for w in ["compare", "对比"]):
        return "comparison"
    if any(w in q for w in ["find", "search", "找"]):
        return "search"
    return "general"
