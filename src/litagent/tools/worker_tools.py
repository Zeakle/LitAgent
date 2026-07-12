from langchain_core.tools import tool

from litagent.memory.manager import MemoryManager
from litagent.rag.claims_index import ClaimsIndex
from litagent.logging import get_logger
from litagent.skills.manager import SkillManager


logger = get_logger('tools.worker')


def make_recall_memory_tool(memory: MemoryManager | None):

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
        facts = recalled.get('facts', [])
        episodes = recalled.get('episodes', [])

        parts = []
        if episodes:
            parts.append("## Past Episodes")
            for ep in episodes[:3]:
                parts.append(f"- {ep.summary}")
        
        if facts:
            parts.append("## Domain Knowledge")
            for f in facts[:5]:
                parts.append(f"- {f.get('key', '?')}: {f.get('value', {})}")

        return '\n'.join(parts) if parts else 'No relevant knowledge found'

    return recall_memory


def make_lookup_claims_tool(claims_index: ClaimsIndex | None):

    @tool
    async def lookup_claims(query: str) -> str:
        """Search the claims index for related claims to cross-reference.

        Use this to verify factual claims against previously extracted
        claims from other papers. Returns matching claims with source papers.

        Args:
            query: Keywords from the claim to verify.
        """
        if claims_index is None:
            return "Claims index not available"

        results = await claims_index.search(query, top_k=5)
        if not results:
            return 'No matching claims found'
        
        parts = []
        for c in results:
            parts.append(
                f"- [{c.confidence:.2f}] {c.text} (source: {c.source_paper})"
            )
        
        return '\n'.join(parts)

    return lookup_claims


def make_load_skill_tool(skill_manager: SkillManager | None):

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