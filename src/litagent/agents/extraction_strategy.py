"""提取策略——regex / LLM / 降级。Strategy Pattern。"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod

from litagent.llm.client import BaseLLMClient
from litagent.tools.executor import ToolExecutor
from litagent.skills.manager import SkillManager
from litagent.context.templates import wrap_xml
from litagent.logging import get_logger


logger = get_logger('agents.extraction_strategy')


class ExtractionStrategy(ABC):
    """一篇论文进，结构化字段 dict 出。"""
    @abstractmethod
    async def extract(self, paper: dict) -> dict: ...


class RegexStrategy(ExtractionStrategy):
    """regex 提取——调 extract.py 的 4 个工具。免费、即时。做降级兜底。"""

    def __init__(self, executor: ToolExecutor):
        self._executor = executor

    async def extract(self, paper: dict) -> dict:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {abstract}".lower()
        claims_r = await self._executor.execute("extract_claims", {"abstract": abstract})
        metrics_r = await self._executor.execute("extract_metrics", {"abstract": abstract})
        methods_r = await self._executor.execute("extract_methods", {"text": text})
        datasets_r = await self._executor.execute("extract_datasets", {"text": text})
        return {
            "claims": claims_r.output if not claims_r.error else [],
            "metrics": metrics_r.output if not metrics_r.error else {},
            "methods": methods_r.output if not methods_r.error else [],
            "datasets": datasets_r.output if not datasets_r.error else [],
        }


_REQUIRED_KEYS_INSTRUCTION = """Output a JSON object with EXACTLY these keys \
(same as the rule-based extractor, so downstream stays consistent):
- "claims": list of key claim sentences
- "metrics": object mapping metric name to value, e.g. {"accuracy": "85.7%"}
- "methods": list of method names
- "datasets": list of dataset names
PLUS any domain-specific fields from the chosen skill \
(e.g. model_architecture, backbone, training_strategy).
Choose the most appropriate skill based on the paper, then extract."""


class LLMStrategy(ExtractionStrategy):
    """LLM提取--注入 SKILL.md, 按领域模板抽字段。主力。"""

    def __init__(self, llm: BaseLLMClient, skill_manager: SkillManager):
        self._llm = llm
        self._skill_manager = skill_manager

    
    async def extract(self, paper: dict) -> dict:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        system = (
            "You are an academic paper extractor.\n"
            f"{self._skill_manager.to_metadata_text_for('extracting structured fields from a paper', top_k=2)}\n\n"
            f"{_REQUIRED_KEYS_INSTRUCTION}"
        )
        
        resp = await self._llm.chat(
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': wrap_xml('paper', f'{title}\n{abstract}')},
            ],
            response_format={'type': 'json_object'},
        )

        data = json.loads(resp.content)
        data.setdefault('claims', [])
        data.setdefault("metrics", {})
        data.setdefault("methods", [])
        data.setdefault("datasets", [])
        return data


class ResilientExtractionStrategy(ExtractionStrategy):
    """LLM extract -> 单篇降级 regex"""

    def __init__(self, llm_strategy: LLMStrategy, regex_strategy: RegexStrategy):
        self._llm = llm_strategy
        self._regex_strategy = regex_strategy

    
    async def extract(self, paper: dict) -> dict:
        try:
            return await self._llm.extract(paper)
        except Exception as e:
            pid = paper.get('paper_id', '?')
            logger.warning(f'LLM extract failed for {pid}: {e}, falling back to regex')
            return await self._regex_strategy.extract(paper)