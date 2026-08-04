"""Define stable run configuration and survey-result contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from litagent.config import AppConfig

CONFIG_SUMMARY_SCHEMA_VERSION = 1
SURVEY_RESULT_SCHEMA_VERSION = 1


def _safe_origin(value: str | None) -> str | None:
    """Return scheme/host/port while removing credentials, path, and query."""
    if not value:
        return None

    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return None

    try:
        port_value = parsed.port
    except ValueError:
        return None

    port = f":{port_value}" if port_value is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _mcp_summary(config: AppConfig) -> list[dict[str, Any]]:
    """Project only non-secret MCP behavior into the run identity."""
    projected: list[dict[str, Any]] = []
    for name, server in sorted(config.mcp_servers.items()):
        projected.append(
            {
                "name": name,
                "transport": server.transport,
                "enabled": server.enabled,
                "sandboxed": server.sandboxed,
                "sandbox_network": server.sandbox_network,
                "command": (Path(server.command).name if server.command else None),
                "origin": _safe_origin(server.url),
            }
        )
    return projected


def _rag_summary(config: AppConfig) -> dict[str, Any] | None:
    """Project the RAG model without accepting unknown fields."""
    rag = getattr(config, "rag", None)
    if rag is None:
        return None
    allowed = (
        "enabled",
        "paper_collection",
        "claims_collection",
        "corpus_version",
        "schema_version",
        "parser_version",
        "chunking_version",
        "embedding_model",
        "content_mode",
        "candidate_k",
        "top_k",
        "reranker_enabled",
        "writeback_enabled",
        "benchmark_collection",
    )

    return {field: getattr(rag, field) for field in allowed if hasattr(rag, field)}


def build_config_summary(config: AppConfig) -> dict[str, Any]:
    """Build a deterministic, non-secret projection of effective settings."""
    effective: dict[str, Any] = {
        "agent": config.agent.model_dump(),
        "llm": {
            "origin": _safe_origin(config.llm.base_url),
            "model": config.llm.model,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
        },
        "memory": {
            # Connection URLs are deployment details and may contain secrets.
            "working_ttl_seconds": config.memory.working_ttl_seconds,
        },
        "context": config.context.model_dump(),
        "orchestrator": config.orchestrator.model_dump(),
        "adversarial": config.adversarial.model_dump(),
        "safety": config.safety.model_dump(),
        "resilience": config.resilience.model_dump(),
        "extractor": config.extractor.model_dump(),
        "relevance": config.relevance.model_dump(),
        "eval": config.eval.model_dump(),
        "planner": config.planner.model_dump(),
        "observability": {
            "enabled": config.observability.enabled,
            "payload_mode": config.observability.payload_mode,
        },
        "mcp_servers": _mcp_summary(config),
    }
    rag = _rag_summary(config)
    if rag is not None:
        effective["rag"] = rag

    identity = {"schema_version": CONFIG_SUMMARY_SCHEMA_VERSION, "effective": effective}

    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        **identity,
        "fingerprint": f"sha256:{digest}",
    }


class SurveyResult(BaseModel):
    """Validate the one report payload shared by every delivery surface."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SURVEY_RESULT_SCHEMA_VERSION
    config_fingerprint: str = "unknown"
    survey: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    graph_data: dict[str, Any] = Field(default_factory=dict)
    partial: bool = False
    evaluation: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(
        default_factory=lambda: {
            "status": "unverified",
            "failed_metrics": [],
            "unverified_metrics": [],
        }
    )
    delivery: dict[str, Any] = Field(default_factory=dict)


def normalize_survey_result(
    payload: Mapping[str, Any],
    *,
    config_fingerprint: str,
) -> dict[str, Any]:
    """Validate and serialize the runner's sole final-report contract."""
    value = dict(payload)
    value["schema_version"] = SURVEY_RESULT_SCHEMA_VERSION
    value["config_fingerprint"] = config_fingerprint
    return SurveyResult.model_validate(value).model_dump()
