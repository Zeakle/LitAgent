"""Define stable run configuration and survey-result contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from litagent.config import AppConfig
from litagent.quality import WRITING_VERSION
from litagent.safety.budget import STRICT_EXTRACTION_OUTPUT_TOKENS

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
                "allowed_tools": [
                    {
                        "name": tool_name,
                        "enabled": capability.enabled,
                        "category": capability.category.value,
                    }
                    for tool_name, capability in sorted(server.allowed_tools.items())
                ],
            }
        )
    return projected


def _rag_summary(config: AppConfig) -> dict[str, Any] | None:
    """Project non-secret RAG behavior into the run fingerprint."""
    rag = getattr(config, "rag", None)
    if rag is None:
        return None
    allowed = (
        "enabled",
        "paper_collection",
        "claims_collection",
        "benchmark_collection",
        "corpus_version",
        "schema_version",
        "parser_version",
        "chunking_version",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "embedding_backend",
        "embedding_model",
        "embedding_document_adapter",
        "embedding_query_adapter",
        "content_mode",
        "retrieval_mode",
        "candidate_k",
        "top_k",
        "max_representative_chunks",
        "reranker_enabled",
        "reranker_model",
        "writeback_enabled",
        "max_pdf_bytes",
        "download_timeout_seconds",
        "parser_timeout_seconds",
        "max_concurrent_parsers",
        "allowed_pdf_content_types",
    )
    serialized = rag.model_dump(mode="json")
    return {field: serialized[field] for field in allowed}


def build_config_summary(
    config: AppConfig, quality_policy: str = "legacy"
) -> dict[str, Any]:
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
        "safety": config.safety.model_dump(mode="json"),
        "api": config.api.model_dump(mode="json"),
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
    strict_fields = {
        k: v for k, v in effective["eval"].items() if k.startswith("strict_")
    }
    if quality_policy == "legacy":
        for key in strict_fields:
            effective["eval"].pop(key)
    elif quality_policy in {"strict_v1", "strict_auto_v1", "strict_auto_v2"}:
        from litagent.context.evidence_selector import SHARED_SCOPE_CONTRACT
        from litagent.eval.batched import STRICT_DECISION_CONTRACT
        from litagent.eval.scoped import (
            STRICT_BINDING_CONTRACT,
            STRICT_INVENTORY_CONTRACT,
            STRICT_QUANTITY_CONTRACT,
            STRICT_QUANTITY_ROLE_CONTRACT,
            STRICT_SUPPORT_CONTRACT,
        )
        from litagent.quality import (
            DOCUMENT_SUBJECT_CONTRACT,
            FRAGMENT_REPAIR_CONTRACT,
            REFERENCE_PRESERVATION_CONTRACT,
            REPAIR_ENTRY_CONTRACT,
            REQUEST_REFERENCE_CONTRACT,
            WRITING_BUDGET_CONTRACT,
            WRITING_QUOTE_CONTRACT,
        )

        effective["quality_policy"] = {
            "policy": quality_policy,
            "contracts": [
                "fact-card-v1",
                "survey-evaluation-v3",
                STRICT_DECISION_CONTRACT,
                "staged-budget-v1",
                "original-support-only-v1",
                STRICT_BINDING_CONTRACT,
                STRICT_INVENTORY_CONTRACT,
                STRICT_QUANTITY_CONTRACT,
                STRICT_QUANTITY_ROLE_CONTRACT,
                STRICT_SUPPORT_CONTRACT,
                REPAIR_ENTRY_CONTRACT,
                WRITING_BUDGET_CONTRACT,
                WRITING_QUOTE_CONTRACT,
                DOCUMENT_SUBJECT_CONTRACT,
                REQUEST_REFERENCE_CONTRACT,
                REFERENCE_PRESERVATION_CONTRACT,
                FRAGMENT_REPAIR_CONTRACT,
                WRITING_VERSION,
                "source-selection-v1",
                SHARED_SCOPE_CONTRACT,
                "scientific-classification-v2",
                "strict-extraction-output-v1",
                "complete-review-materials-v2",
                "grounded-request-encoding-v1",
            ],
            "request_limit": 64,
            "stage_tokens": [
                int(config.safety.max_cost_tokens * ratio) for ratio in (0.4, 0.2, 0.4)
            ],
            "strict_eval": strict_fields,
        }
        effective["extractor"][
            "strict_max_output_tokens"
        ] = STRICT_EXTRACTION_OUTPUT_TOKENS
        if quality_policy in {"strict_auto_v1", "strict_auto_v2"}:
            effective["extractor"]["automatic_transport_timeout_ms"] = 180000
            effective["extractor"]["automatic_strategy_timeout_ms"] = 185000
            effective["extractor"][
                "automatic_source_key_protocol"
            ] = "host-source-key-v1"
            effective["quality_policy"]["contracts"].extend(
                [
                    "automatic-acceptance-v1",
                    "failed-acceptance-binding-v1",
                    "frozen-original-audit-v1",
                    "automatic-stop-v1",
                ]
            )
            effective["automatic_audit"] = {
                "judge": {
                    **config.report_audit.model_dump(),
                    "judge_base_url": _safe_origin(config.report_audit.judge_base_url),
                },
                "judge_total_tokens": 100000,
                "judge_requests": 2,
                "jev_model": "jev-1.13.0",
                "jev_input_tokens": 200000,
                "jev_requests": 20,
                "confidence_threshold": 0.9,
            }
        if quality_policy == "strict_auto_v2":
            from litagent.eval.batched import (
                GLOBAL_CONSISTENCY_CONTRACT,
                SOURCE_FIDELITY_CONTRACT,
            )
            from litagent.eval.scoped import LOCAL_NUMERIC_CONTRACT
            from litagent.extraction_verification import LEDGER_CONTRACT
            from litagent.extraction_verification import VERSION as EXTRACTION_VERSION
            from litagent.quality import (
                NATIVE_REPAIR_CONTRACT,
                TARGETED_REPAIR_CONTRACT,
            )
            from litagent.writing_passages import FULL_CONTRACT as CONTRACT
            from litagent.writing_passages import FULL_VERSION as VERSION
            from litagent.writing_passages import NOTICE_VERSION, SCOPE_NOTICE_CONTRACT

            effective["quality_policy"]["contracts"].extend(
                [
                    EXTRACTION_VERSION,
                    VERSION,
                    CONTRACT,
                    SOURCE_FIDELITY_CONTRACT,
                    LEDGER_CONTRACT,
                    LOCAL_NUMERIC_CONTRACT,
                    GLOBAL_CONSISTENCY_CONTRACT,
                    NATIVE_REPAIR_CONTRACT,
                    TARGETED_REPAIR_CONTRACT,
                    NOTICE_VERSION,
                    SCOPE_NOTICE_CONTRACT,
                ]
            )
            effective["automatic_audit"]["jev_stage"] = "extraction"
    else:
        raise ValueError("unsupported_quality_policy")

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
    metadata = dict(value.get("metadata", {}))
    acceptance = metadata.get("automatic_acceptance")
    if (
        metadata.get("quality_policy") in {"strict_auto_v1", "strict_auto_v2"}
        and acceptance
    ):
        from litagent.quality import digest

        if (
            acceptance.get("report_sha256")
            != hashlib.sha256(value["survey"].encode()).hexdigest()
            or acceptance.get("fact_cards_sha256")
            != digest(metadata.get("fact_cards", {}))
            or acceptance.get("evaluation_sha256")
            != digest(value.get("evaluation", {}))
            or acceptance.get("methodology_sha256")
            != digest(metadata.get("review_methodology", {}))
            or (
                (
                    metadata.get("verification") is not None
                    or acceptance.get("version") == "automatic-acceptance-v4"
                )
                and acceptance.get("verification_sha256")
                != digest(metadata.get("verification"))
            )
            or acceptance.get("body_quality", "legacy")
            != metadata.get("body_quality", "legacy")
            or (
                metadata.get("quality_policy") == "strict_auto_v2"
                and acceptance.get("extraction_verification_sha256")
                != digest(metadata.get("extraction_verification", {}))
            )
        ):
            metadata["automatic_acceptance"] = {
                **acceptance,
                "status": "invalidated",
                "publishable": False,
                "failure_codes": ["automatic_binding_changed"],
            }
            value["metadata"] = metadata
            value["delivery"] = {
                "status": "blocked",
                "publishable": False,
                "reason_codes": ["automatic_binding_changed"],
            }
    value["schema_version"] = SURVEY_RESULT_SCHEMA_VERSION
    value["config_fingerprint"] = config_fingerprint
    return SurveyResult.model_validate(value).model_dump()
