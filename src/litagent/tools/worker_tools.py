"""Build LangChain tools backed by optional memory and skill services."""

from langchain_core.tools import tool

from litagent.memory.manager import MemoryManager
from litagent.logging import get_logger
from litagent.skills.manager import SkillManager

logger = get_logger("tools.worker")


def make_recall_memory_tool(memory: MemoryManager | None):
    """Create a memory-recall tool bound to the supplied manager."""

    @tool
    async def recall_memory(query: str) -> str:
        """Recall relevant domain knowledge and past episodes from memory.

        Use this when you need to reference previously gathered knowledge
        or past survey findings about a topic.

        Args:
            query: The search query for memory recall.
        """
        if not memory:
            return "Memory not available (Redis/Qdrant not connected)."

        recalled = await memory.recall(query)
        facts = recalled.get("facts", [])
        episodes = recalled.get("episodes", [])

        parts = []
        if episodes:
            parts.append("## Past Episodes")
            for ep in episodes[:3]:
                parts.append(f"- {ep.summary}")

        if facts:
            parts.append("## Domain Knowledge")
            for f in facts[:5]:
                parts.append(f"- {f.get('key', '?')}: {f.get('value', {})}")

        return "\n".join(parts) if parts else "No relevant knowledge found"

    return recall_memory


def make_load_skill_tool(skill_manager: SkillManager | None):
    """Create a skill-loading tool bound to the supplied manager."""

    @tool
    async def load_skill(name: str) -> str:
        """Load a skill's full methodology by name when you need domain guidance.

        Available skill names are listed in your system prompt under 'Available Skills'.

        Args:
            name: The skill name to load (from the Available Skills list).
        """
        if skill_manager is None:
            return "Skills not available"

        try:
            return skill_manager.get_body(name)
        except KeyError:
            return f"Skill {name} not found"

    return load_skill
