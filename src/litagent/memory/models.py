"""Define serializable records shared by memory backends."""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Episode:
    """Summarize a completed session for episodic storage and recall."""

    summary: str
    intent: str
    session_id: str
    user_id: str = "default"

    key_findings: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    # Normalized importance in the inclusive range from zero to one.
    importance_score: float = 0.5
    episode_id: str = ""
    created_at: float = 0.0
    last_recalled_at: float = 0.0
    recall_count: int = 0
    extracted_facts: list[dict] = field(default_factory=list)

    def decay_score(self) -> float:
        """Apply 30-day decay and a bounded recall bonus to importance."""
        import time

        if self.created_at <= 0:
            return self.importance_score

        days = (time.time() - self.created_at) / 86400.0
        recall_bonus = 1.0 + min(self.recall_count / 5.0, 1.0)
        return self.importance_score * (0.5 ** (days / 30.0)) * recall_bonus

    def to_dict(self) -> dict:
        """Serialize this episode to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Build an episode while ignoring unknown persisted fields."""
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
