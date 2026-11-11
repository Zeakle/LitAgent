"""Describe the explicit twenty-percent resource profile and its audit surface."""

from __future__ import annotations

import copy
from fractions import Fraction

PROFILE = "operational-plus20-v1"
FACTOR = Fraction(6, 5)
# These are workload or resource ceilings, not scientific acceptance thresholds.
RESOURCE_PATHS = (
    "agent.max_loops",
    "memory.working_ttl_seconds",
    "context.max_tokens",
    "context.synthesis_evidence_max_tokens",
    "context.synthesis_papers_max_tokens",
    "context.evidence_top_k_per_section",
    "context.evidence_max_items",
    "context.evidence_per_paper_cap",
    "context.review_evidence_max_tokens",
    "context.review_draft_max_tokens",
    "context.review_feedback_max_tokens",
    "extractor.max_concurrent",
    "extractor.max_papers",
    "extractor.max_tokens",
    "extractor.per_paper_timeout_ms",
    "extractor.task_timeout_ms",
    "extractor.automatic_timeout_seconds",
    "orchestrator.timeout_ms",
    "orchestrator.max_concurrent",
    "llm.max_tokens",
    "relevance.max_papers",
    "rag.candidate_k",
    "rag.top_k",
    "rag.max_pdf_bytes",
    "rag.download_timeout_seconds",
    "rag.parser_timeout_seconds",
    "rag.max_concurrent_parsers",
    "rag.max_representative_chunks",
    "rag.trusted_claim_recall_top_k",
    "rag.trusted_claim_context_max_chars",
    "adversarial.max_rounds",
    "adversarial.review_max_tokens",
    "adversarial.task_timeout_ms",
    "safety.max_cost_tokens",
    "api.max_concurrent_surveys",
    "api.cancellation_grace_seconds",
    "eval.context_tokens",
    "eval.max_tokens",
    "eval.timeout_seconds",
    "eval.total_timeout_seconds",
    "eval.faithfulness_max_evidence_items",
    "eval.faithfulness_max_context_chars",
    "eval.strict_batch_size",
    "eval.strict_max_batches",
    "eval.strict_max_claims",
    "eval.rewrite.max_items",
    "eval.rewrite.max_chars",
    "planner.max_sub_queries",
    "planner.max_tokens",
    "report_audit.judge_context_tokens",
    "report_audit.judge_input_tokens",
)
CAPACITY_PATHS = (
    "max_body_units",
    "max_summary_units",
    "max_unit_characters",
    "initial_assertions",
    "batch_size",
    "body_json_bytes",
    "summary_json_bytes",
    "claim_json_bytes",
    "claim_max_json_bytes",
    "diagnostic_json_bytes",
)
EXCEPTIONS = {
    "scientific_thresholds": "Acceptance and confidence thresholds are not resource ceilings.",
    "transport_retries": "Zero retries remain zero; unknown responses are not replayed.",
    "writing_targets": "The 48-unit target is unchanged; only maximum capacity grows.",
    "unbounded_operational_limits": "Absent assertion/batch ceilings remain budget-derived.",
    "parser_safety": "Implementation safety bounds remain separate from active workload ceilings.",
    "provider_capabilities": "Configured requests must still fit verified provider capabilities.",
    "chunk_geometry": "Chunk size/overlap are retrieval design parameters, not run ceilings.",
    "circuit_breaker": "Failure thresholds remain safeguards, not workload allowances.",
}


def increase(value):
    """Increase a positive resource ceiling without floating-point under-rounding."""
    if type(value) not in (int, float) or value <= 0:
        raise ValueError("resource_ceiling_invalid")
    result = Fraction(str(value)) * FACTOR
    return (
        (result.numerator + result.denominator - 1) // result.denominator
        if type(value) is int
        else float(result)
    )


def uplift_configuration(baseline, capacity):
    """Return explicit effective YAML data and a complete old/new resource inventory."""
    result = copy.deepcopy(baseline)
    inventory = {}
    paths = list(RESOURCE_PATHS)
    for parent in ("orchestrator.task_timeouts_ms", "llm.request_timeouts_seconds"):
        section, key = parent.split(".")
        paths.extend(f"{parent}.{name}" for name in baseline[section][key])
    for path in paths:
        keys = path.split(".")
        node = result
        for key in keys[:-1]:
            node = node[key]
        old = node[keys[-1]]
        node[keys[-1]] = increase(old)
        inventory[path] = {"old": old, "new": node[keys[-1]]}
    result["rag"]["resource_revision"] = PROFILE
    result["llm"]["resource_revision"] = PROFILE
    for key, old in {"fulltext_papers": 8, "fulltext_concurrency": 2}.items():
        inventory["rag." + key] = {"old": old, "new": increase(old)}
    result["capacity"] = {**copy.deepcopy(capacity), "revision": PROFILE}
    for key in CAPACITY_PATHS:
        old = capacity[key]
        result["capacity"][key] = increase(old)
        inventory["capacity." + key] = {"old": old, "new": result["capacity"][key]}
    result["safety"]["planned_run_limits"] = {
        "profile": PROFILE,
        "max_tokens": 12_000_000,
        "max_requests": 77,
        "max_repairs": 3,
        "context_tokens": 235930,
        "max_output_tokens": 39322,
    }
    for key, old in {
        "max_tokens": 10_000_000,
        "max_requests": 64,
        "max_repairs": 2,
    }.items():
        inventory["safety.planned_run_limits." + key] = {
            "old": old,
            "new": increase(old),
        }
    for key, old in {
        "transport.preparation_context_tokens": 131072,
        "transport.maximum_request_timeout_seconds": 600,
        "safety.planned_run_limits.context_tokens": 196608,
        "safety.planned_run_limits.max_output_tokens": 32768,
        "extraction_jev.max_questions_per_batch": 32,
        "extraction_jev.concurrency": 4,
        "extraction_jev.max_requests": 200,
        "extraction_jev.max_input_tokens": 4000000,
        "extraction_jev.max_request_bytes": 24000,
        "extraction_jev.timeout_seconds": 30,
        "report_audit.concurrency": 1,
        "report_audit.judge_max_output_tokens": 16384,
        "report_audit.judge_timeout_seconds": 180,
        "report_audit.judge_total_tokens": 100000,
        "report_audit.jev.max_questions_per_batch": 16,
        "report_audit.jev.concurrency": 2,
        "report_audit.jev.max_requests": 20,
        "report_audit.jev.max_input_tokens": 200000,
        "report_audit.jev.max_request_bytes": 24000,
        "report_audit.jev.timeout_seconds": 30,
    }.items():
        inventory[key] = {"old": old, "new": increase(old)}
    return result, inventory


def is_uplifted(config):
    """Select the versioned profile from server configuration only."""
    if config is None:
        return False
    limits = getattr(getattr(config, "safety", None), "planned_run_limits", None)
    from litagent.run_limits import EXTENDED_PROFILE, FORMAT_PROFILE

    return getattr(limits, "profile", None) in (
        PROFILE,
        EXTENDED_PROFILE,
        FORMAT_PROFILE,
    )


def configured_ceiling(config, value):
    """Leave old explicit configurations unchanged while expanding new resources."""
    return increase(value) if is_uplifted(config) else value


def budget_resource_profile(budget):
    """Resolve a resource revision from the host owner or its authenticated plan."""
    from litagent.run_limits import EXTENDED_PROFILE, FORMAT_PROFILE

    if getattr(budget, "resource_profile", None) in (
        PROFILE,
        EXTENDED_PROFILE,
        FORMAT_PROFILE,
    ):
        return PROFILE
    backend = getattr(budget, "_planned_backend", None)
    plan = getattr(backend if backend is not None else budget, "plan", None)
    if isinstance(plan, dict) and (
        plan.get("contract") == "planned-preparation-v2"
        or plan.get("envelope", {}).get("run_limits", {}).get("profile")
        in (PROFILE, EXTENDED_PROFILE, FORMAT_PROFILE)
    ):
        return PROFILE
    return None
