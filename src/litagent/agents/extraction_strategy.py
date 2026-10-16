"""Extract structured paper data with LLM and rule-based strategies."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Mapping

from litagent.context.templates import wrap_xml
from litagent.llm.client import BaseLLMClient, LLMResponse
from litagent.logging import get_logger
from litagent.safety.budget import STRICT_EXTRACTION_OUTPUT_TOKENS
from litagent.skills.manager import SkillManager
from litagent.tools.executor import ToolExecutor
from litagent.workbench import is_strict_policy

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
    strict = is_strict_policy(paper.get("quality_policy"))

    def priority(chunk):
        """Prefer recorded scientific sections without assigning semantic trust."""
        section = str(chunk.get("section") or "unknown").casefold()
        if section == "abstract" or chunk.get("content_scope") == "abstract":
            return 0
        if section in {
            "introduction",
            "background",
            "method",
            "methods",
            "methodology",
            "results",
            "experiments",
            "discussion",
            "conclusion",
            "conclusions",
        }:
            return 1
        return 2 if len(str(chunk.get("text") or "")) >= 160 else 3

    if strict:
        chunks = [
            chunk
            for chunk in chunks
            if str(chunk.get("section") or "").casefold()
            not in {
                "title",
                "authors",
                "author",
                "affiliation",
                "affiliations",
                "references",
                "bibliography",
                "acknowledgments",
                "acknowledgements",
            }
        ]
    ranked = sorted(
        enumerate(chunks),
        key=lambda pair: (
            (
                priority(pair[1])
                if strict
                else (0 if pair[1].get("content_scope") == "selected_fulltext" else 1)
            ),
            pair[0],
        ),
    )
    selected = [
        chunk
        for _, chunk in (ranked if strict else ranked[:max_chunks])
        if chunk.get("text")
    ]
    title = str(paper.get("title") or "")[:500]
    lines = [f"Title: {title}"[:max_chars]]
    keys: list[str] = []
    for chunk in selected:
        if len(keys) >= max_chunks:
            break
        key = str(chunk.get("chunk_key") or "")
        if not key:
            continue
        # A per-chunk cap prevents one malformed PDF block from consuming the
        # complete extraction budget while preserving stable chunk markers.
        prefix = f"[CHUNK:{key}] section={chunk.get('section', 'unknown')}\n"
        used = len("\n\n".join(lines))
        available = max_chars - used - 2 - len(prefix)
        if available <= 0:
            if strict:
                continue
            break
        text = str(chunk["text"])
        if strict:
            if len(text) > available:
                continue
        else:
            text = text[: min(2_800, available)]
        lines.append(f"{prefix}{text}")
        keys.append(key)

    if not keys and paper.get("abstract"):
        used = len("\n\n".join(lines))
        prefix = "[ABSTRACT]\n"
        available = max_chars - used - 2 - len(prefix)
        if available > 0:
            lines.append(f"{prefix}{str(paper['abstract'])[: min(2_800, available)]}")

    return "\n\n".join(lines), keys


def _decode_source_chunk_key(value: Any, selected_keys: list[str]) -> str:
    """Resolve only an unambiguous, exact host ID or its display wrapper."""
    if not isinstance(value, str):
        raise ValueError("source_key_type")
    matches = {
        key for key in selected_keys if value in (key, f"CHUNK:{key}", f"[CHUNK:{key}]")
    }
    if len(matches) != 1:
        raise ValueError("source_key_ambiguous" if matches else "source_key_unknown")
    return matches.pop()


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
        fields = item.get("fact_fields")
        if isinstance(fields, Mapping):
            records[-1]["fact_fields"] = {
                key: value.strip()
                for key, value in fields.items()
                if key
                in {
                    "method",
                    "conditions",
                    "dataset",
                    "metric",
                    "value",
                    "unit",
                    "qualifiers",
                }
                and isinstance(value, str)
                and value.strip()
            }
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

_FACT_FIELD_INSTRUCTION = """\nFor strict fact-card-v1 also include fact_fields per
claim_record: method, conditions, dataset, metric, value, unit, qualifiers.
Copy each populated field verbatim from its actual source chunk. Missing or ambiguous
fields are null. EVERY fact_fields value must be a single string or null, never an
array, number, boolean or object. When a finding spans several methods or datasets,
copy one continuous source phrase into the field or leave it null; do not build a
list of inferred names. For example: {"method": "Prototypical Networks",
"conditions": null, "dataset": "miniImageNet", "metric": null, "value": null,
"unit": null, "qualifiers": null}.
Do not guess aliases, settings, causality or paper identity. These
are unverified semantic extractions, not scientific judgments."""


def _parse_extraction_response(
    response: LLMResponse, *, max_tokens: int, strict: bool
) -> dict:
    """Reject incomplete structured responses without salvaging partial evidence."""
    reason = response.finish_reason
    if strict and reason == "length":
        raise ValueError("strict_extraction_output_truncated")
    if strict and reason not in {None, "stop"}:
        raise ValueError("strict_extraction_output_incomplete")
    completion = response.usage.get("completion_tokens")
    at_limit = type(completion) is int and completion >= max_tokens
    try:
        data = json.loads(response.content)
    except json.JSONDecodeError as exc:
        if not strict:
            raise
        # Historical responses lack a provider reason. Label inference explicitly.
        code = (
            "strict_extraction_output_truncated_inferred"
            if reason is None and at_limit
            else "strict_extraction_json_invalid"
        )
        raise ValueError(code) from exc
    if strict and reason is None and at_limit:
        raise ValueError("strict_extraction_completion_unverified")
    if not isinstance(data, dict):
        raise ValueError("LLM extraction output must be a JSON object")
    return data


class LLMStrategy(ExtractionStrategy):
    """Extract structured paper fields with an LLM and skill metadata."""

    def __init__(
        self,
        llm: BaseLLMClient,
        skill_manager: SkillManager,
        *,
        max_tokens: int = 2048,
        thinking_mode: str = "disabled",
    ):
        """Initialize the LLM strategy."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if thinking_mode not in {"provider_default", "enabled", "disabled"}:
            raise ValueError("unsupported thinking_mode")
        self._llm = llm
        self._skill_manager = skill_manager
        self._max_tokens = max_tokens
        self._thinking_mode = thinking_mode

    async def extract(self, paper: dict) -> dict:
        """Return normalized structured fields from an LLM JSON response."""
        context, selected_keys = build_extraction_context(paper)
        automatic = paper.get("quality_policy") in {"strict_auto_v1", "strict_auto_v2"}
        strict = is_strict_policy(paper.get("quality_policy"))
        required = _REQUIRED_KEYS_INSTRUCTION
        if automatic:
            required = required.replace(
                "source_chunk_key MUST exactly match a supplied CHUNK marker.",
                "source_chunk_key MUST be one of the bare host IDs listed below."
                " The display marker [CHUNK:ID] is a wrapper; copy ID only,"
                " without CHUNK: or brackets. If no IDs are supplied, return"
                " an empty claim_records list.",
            )
            required += "\nAllowed source_chunk_key IDs: " + json.dumps(
                selected_keys, ensure_ascii=False
            )
            if selected_keys:
                required += "\nExample source_chunk_key: " + json.dumps(
                    selected_keys[0], ensure_ascii=False
                )
        system = (
            "You are an academic paper extractor.\n"
            f"""{self._skill_manager.to_metadata_text_for(
                'extracting structured fields from a paper',
                top_k=2,
            )}\n\n"""
            f"{required}" + (_FACT_FIELD_INSTRUCTION if strict else "")
        )

        kwargs: dict[str, Any] = {
            "response_format": {"type": "json_object"},
            "max_tokens": (
                STRICT_EXTRACTION_OUTPUT_TOKENS if strict else self._max_tokens
            ),
        }
        if self._thinking_mode != "provider_default":
            kwargs["extra_body"] = {"thinking": {"type": self._thinking_mode}}

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": wrap_xml("paper", context)},
        ]
        if strict:
            from litagent.eval.scoped import check_request_budget

            check_request_budget(
                self._llm, messages, kwargs["max_tokens"], int(131072 * 0.9)
            )
            kwargs["transport_max_attempts"] = 1
        resp = await self._llm.chat(messages, **kwargs)

        data = _parse_extraction_response(
            resp, max_tokens=kwargs["max_tokens"], strict=strict
        )
        if automatic:
            if (
                not isinstance(data.get("claim_records"), list)
                or not isinstance(data.get("metrics"), dict)
                or any(
                    not isinstance(data.get(key), list)
                    or any(not isinstance(value, str) for value in data[key])
                    for key in ("methods", "datasets")
                )
            ):
                raise ValueError("extraction_contract_invalid")
            seen = set()
            normalizations = []
            for index, record in enumerate(data["claim_records"]):
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("text"), str)
                    or not record["text"].strip()
                    or (
                        record.get("confidence") is not None
                        and (
                            type(record["confidence"]) not in (int, float)
                            or not 0 <= record["confidence"] <= 1
                        )
                    )
                ):
                    raise ValueError("extraction_contract_invalid")
                raw_key = record.get("source_chunk_key")
                try:
                    key = _decode_source_chunk_key(raw_key, selected_keys)
                except ValueError as exc:
                    raise ValueError(
                        f"extraction_contract_invalid:{exc}:record:{index}"
                    ) from exc
                if key != raw_key:
                    normalizations.append(
                        {
                            "record_index": index,
                            "raw_source_chunk_key": raw_key,
                            "source_chunk_key": key,
                        }
                    )
                # Canonicalize before duplicate checking and evidence construction;
                # display aliases must not create a second source or fact card.
                record["source_chunk_key"] = key
                identity = (record["source_chunk_key"], record["text"].strip())
                fields = record.get("fact_fields")
                if identity in seen:
                    raise ValueError("extraction_contract_invalid:duplicate_record")
                if fields is not None:
                    if not isinstance(fields, dict):
                        raise ValueError("extraction_contract_invalid:fact_fields_type")
                    for key, value in fields.items():
                        if value is not None and not isinstance(value, str):
                            field = (
                                key
                                if key
                                in {
                                    "method",
                                    "conditions",
                                    "dataset",
                                    "metric",
                                    "value",
                                    "unit",
                                    "qualifiers",
                                }
                                else "unknown"
                            )
                            raise ValueError(
                                f"extraction_contract_invalid:fact_field_type:{field}"
                            )
                seen.add(identity)
            data["source_key_normalizations"] = normalizations
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
        if paper.get("quality_policy") in {"strict_auto_v1", "strict_auto_v2"}:
            # The client owns a 180s bounded transport. Do not cancel a valid
            # in-flight request earlier merely to substitute a regex result.
            result = await asyncio.wait_for(self._llm.extract(paper), timeout=185)
            result["extraction_mode"] = "llm"
            return result
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
