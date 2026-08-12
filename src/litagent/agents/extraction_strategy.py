"""Extract structured paper data with LLM and rule-based strategies."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Mapping

from litagent.context.templates import wrap_xml
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger
from litagent.skills.manager import SkillManager
from litagent.tools.executor import ToolExecutor

logger = get_logger("agents.extraction_strategy")


def build_extraction_context(
    paper: Mapping[str, Any],
    *,
    max_chunks: int = 4,
    max_chars: int = 12_000,
) -> tuple[str, list[str]]:
    """Render bounded, locator-marked evidence for claim extraction."""
    if max_chunks <= 0 or max_chars <= 0:
        raise ValueError("max_chunks and max_chars must be positive")

    raw_chunks = paper.get("chunks") if isinstance(paper, Mapping) else None
    chunks = [chunk for chunk in (raw_chunks or []) if isinstance(chunk, Mapping)]
    ranked = sorted(
        enumerate(chunks),
        key=lambda pair: (
            0 if pair[1].get("content_scope") == "selected_fulltext" else 1,
            pair[0],
        ),
    )
    selected = [chunk for _, chunk in ranked[:max_chunks] if chunk.get("text")]
    title = str(paper.get("title") or "")[:500]
    lines = [f"Title: {title}"[:max_chars]]
    keys: list[str] = []
    for chunk in selected:
        key = str(chunk.get("chunk_key") or "")
        if not key:
            continue
        # A per-chunk cap prevents one malformed PDF block from consuming the
        # complete extraction budget while preserving stable chunk markers.
        prefix = f"[CHUNK:{key}] section={chunk.get('section', 'unknown')}\n"
        used = len("\n\n".join(lines))
        available = max_chars - used - 2 - len(prefix)
        if available <= 0:
            break
        text = str(chunk["text"])[: min(2_800, available)]
        lines.append(f"{prefix}{text}")
        keys.append(key)

    if not keys and paper.get("abstract"):
        used = len("\n\n".join(lines))
        prefix = "[ABSTRACT]\n"
        available = max_chars - used - 2 - len(prefix)
        if available > 0:
            lines.append(f"{prefix}{str(paper['abstract'])[: min(2_800, available)]}")

    return "\n\n".join(lines), keys


def _normalize_claim_records(
    value: Any,
    *,
    allowed_chunk_keys: set[str],
) -> list[dict[str, Any]]:
    """Keep only well-formed claims anchored to supplied extraction chunks."""
    if not isinstance(value, list):
        return []

    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        chunk_key = str(item.get("source_chunk_key") or "").strip()
        if not text or chunk_key not in allowed_chunk_keys:
            continue
        confidence = item.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue
            if not 0.0 <= confidence <= 1.0:
                continue
        records.append(
            {
                "text": text,
                "source_chunk_key": chunk_key,
                "confidence": confidence,
            }
        )
    return records


class ExtractionStrategy(ABC):
    """Define the asynchronous paper-extraction contract."""

    @abstractmethod
    async def extract(self, paper: dict) -> dict:
        """Extract structured data from the supplied paper."""
        ...


class RegexStrategy(ExtractionStrategy):
    """Extract paper fields with registered rule-based tools."""

    def __init__(self, executor: ToolExecutor):
        """Initialize the regex strategy."""
        self._executor = executor

    async def extract(self, paper: dict) -> dict:
        """Extract claims, metrics, methods, and datasets with local tools."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {abstract}".lower()
        claims_r = await self._executor.execute(
            "extract_claims", {"abstract": abstract}
        )
        metrics_r = await self._executor.execute(
            "extract_metrics", {"abstract": abstract}
        )
        methods_r = await self._executor.execute("extract_methods", {"text": text})
        datasets_r = await self._executor.execute("extract_datasets", {"text": text})
        raw_claims = claims_r.output if not claims_r.error else []
        claims = (
            [
                claim.strip()
                for claim in raw_claims
                if isinstance(claim, str) and claim.strip()
            ]
            if isinstance(raw_claims, list)
            else []
        )
        abstract_chunk_key = next(
            (
                str(chunk.get("chunk_key"))
                for chunk in paper.get("chunks", [])
                if isinstance(chunk, Mapping)
                and chunk.get("chunk_key")
                and (
                    chunk.get("content_scope") == "abstract"
                    or chunk.get("section") == "abstract"
                )
            ),
            None,
        )
        claim_records = (
            [
                {
                    "text": claim.strip(),
                    "source_chunk_key": abstract_chunk_key,
                    "confidence": None,
                }
                for claim in claims
            ]
            if abstract_chunk_key
            else []
        )
        return {
            "claims": claims,
            "claim_records": claim_records,
            "metrics": metrics_r.output if not metrics_r.error else {},
            "methods": methods_r.output if not methods_r.error else [],
            "datasets": datasets_r.output if not datasets_r.error else [],
        }


_REQUIRED_KEYS_INSTRUCTION = """Output a JSON object with these required keys:
- "claim_records": list of objects with "text", "source_chunk_key", and optional
  "confidence" in [0, 1]. source_chunk_key MUST exactly match a supplied CHUNK marker.
- "metrics": object mapping metric name to value, e.g. {"accuracy": "85.7%"}
- "methods": list of method names
- "datasets": list of dataset names
You may add domain-specific fields from the chosen skill \
(e.g. model_architecture, backbone, training_strategy).
Do not create a claim when no supplied chunk directly supports it."""


class LLMStrategy(ExtractionStrategy):
    """Extract structured paper fields with an LLM and skill metadata."""

    def __init__(self, llm: BaseLLMClient, skill_manager: SkillManager):
        """Initialize the LLM strategy."""
        self._llm = llm
        self._skill_manager = skill_manager

    async def extract(self, paper: dict) -> dict:
        """Return normalized structured fields from an LLM JSON response."""
        context, selected_keys = build_extraction_context(paper)
        system = (
            "You are an academic paper extractor.\n"
            f"""{self._skill_manager.to_metadata_text_for(
                'extracting structured fields from a paper',
                top_k=2,
            )}\n\n"""
            f"{_REQUIRED_KEYS_INSTRUCTION}"
        )

        resp = await self._llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": wrap_xml("paper", context)},
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(resp.content)
        if not isinstance(data, dict):
            raise ValueError("LLM extraction output must be a JSON object")
        claim_records = _normalize_claim_records(
            data.get("claim_records"),
            allowed_chunk_keys=set(selected_keys),
        )
        data["claim_records"] = claim_records
        data["claims"] = [record["text"] for record in claim_records]
        data.setdefault("metrics", {})
        data.setdefault("methods", [])
        data.setdefault("datasets", [])
        return data


class ResilientExtractionStrategy(ExtractionStrategy):
    """Use bounded LLM extraction with a rule-based fallback."""

    def __init__(
        self,
        llm_strategy: LLMStrategy,
        regex_strategy: RegexStrategy,
        per_paper_timeout_ms: int = 20000,
    ):
        """Initialize the resilient extraction strategy."""
        self._llm = llm_strategy
        self._regex_strategy = regex_strategy

        if per_paper_timeout_ms <= 0:
            raise ValueError("per_paper_timeout_ms must be positive")
        self._timeout_seconds = per_paper_timeout_ms / 1000.0

    async def extract(self, paper: dict) -> dict:
        """Extract one paper and degrade to rules on timeout or LLM failure."""
        paper_id = paper.get("paper_id", "?")
        try:
            result = await asyncio.wait_for(
                self._llm.extract(paper), timeout=self._timeout_seconds
            )

            result["extraction_mode"] = "llm"
            return result
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return await self._fallback(paper, paper_id, "llm_timeout", "TimeoutError")
        except json.JSONDecodeError as exc:
            return await self._fallback(
                paper, paper_id, "invalid_llm_output", type(exc).__name__
            )
        except Exception as exc:
            return await self._fallback(
                paper, paper_id, "llm_error", type(exc).__name__
            )

    async def _fallback(
        self, paper: dict, paper_id: str, reason: str, error_type: str
    ) -> dict:
        """Run the fallback extraction strategy."""
        logger.warning(
            "LLM extract degraded for paper=%s reason=%s error_type=%s",
            paper_id,
            reason,
            error_type,
        )

        result = await self._regex_strategy.extract(paper)
        result["extraction_mode"] = "regex_fallback"
        result["degradation_reason"] = reason
        return result
