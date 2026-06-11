"""Episodic Memory 数据模型。"""


from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Episode:
    """一次对话中提取出的关键事件——Episodic Memory 的存储单元。

    由 Consolidate 流程从 Working Memory 中提取，LLM 生成结构化摘要后写入 Qdrant。
    """
    summary: str  # 人类可读摘要
    intent: str  # 用户意图
    session_id: str  # 来源会话
    user_id: str = 'default'

    key_findings: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    importance_score: float = 0.5  # 0-1
    episode_id: str = ""
    created_at: float = 0.0
    last_recalled_at: float = 0.0
    recall_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)  # 创建实例
