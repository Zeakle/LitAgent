"""Project reviewed business fields from Survey archives for the public workbench."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit

from litagent.citation_formats import REF_PATTERN
from litagent.demo import extract_report_evidence
from litagent.observability.recorder import _redact_trace_value
from litagent.report_render import (
    build_reader_report,
    paper_catalog_from_nodes,
    render_report_html,
)
from litagent.workbench import ReportVersion

PUBLIC_FIELDS = set(
    """
    accepted accepted_count affected_sections agent_type analysis analysis_plan
    approved assertion assertion_count assertions attempt authors body_quality candidate_count
    candidate_k candidate_paper_ids canonical_id canonical_paper_id category
    checked_claims citation citation_accuracy citation_count citation_id
    citation_ids citations cited_count claim claim_contract claim_id claim_records
    claim_text claims conflict_count conflicts
    complete completed completion_tokens confidence constraint constraints content_scope
    contradiction contradiction_count contradictions contradicted_count
    count coverage coverage_constraints coverage_matrix coverage_requirements criteria
    datasets decision decision_code decomposition degradation_reason degradation_reasons
    delivery_status description details dimensions display_name doi draft elapsed_ms embedding_backend
    empty_index error_code error_type estimated_tokens evaluation evaluation_checks evidence_id
    evidence_ids evidence_items evidence_selection evidence_snapshot_sha256
    exclusion_reason explanation extraction_batch extraction_count extraction_mode
    extraction_verification extractions factual_claim_count
    failed failed_count faithfulness fact_fields feedback final_draft final_score finish_reason
    first_author from_cache from_fallback fulltext_acquisition fulltext_count
    generation_strategy global_issues graph_analysis has_fulltext host_processed_units
    implementation initial_chunk_candidates internal_consistency inventory_check
    invalid_citations issue issue_type issues iteration kind label language
    limitations max_items max_representative_chunks max_results max_sub_queries
    malformed_evidence_refs max_tokens method method_by_section methods metrics
    missing missing_paper_ids mode note
    model model_processed_units model_source_units name named_methods
    negative_evidence omitted_count output_limit_parameter
    output_size paper_count paper_id paper_ids paper_title papers partial penalty
    passed phase previous_draft previous_review priority processed_assertions processed_units
    prompt_tokens provider_api provider_status quality quality_policy quality_status
    query query_index questions ragas_score rank raw_hit_count real_count reason
    reason_code reason_codes reasoning_effort recovered relevance_method relevance_rank relevance_score
    replayed report_language report_sha256 requested_chunk_candidates
    requested_methods required required_paper_ids rerank_applied reranker_model
    result_count results retrieval retrieval_mode review review_feedback
    review_methodology review_round review_structure reviews revision round rounds
    schema_version scope score section section_evidence_ids section_key section_keys
    selected_count selected_ids selected_items selected_paper_ids seminal_papers
    severity skipped source source_paper_id source_report_sha256 source_units stages
    state status strict sub_queries
    suggestions summary support support_status supported supported_count
    supporting_claims supporting_evidence survey target targets task_id
    temperature threshold tier tier_counts title top_k top_k_per_section
    topic total total_assertions total_rounds total_tokens type unit_id unknown_citations
    unknown_evidence_ids unsupported unsupported_claims unsupported_count unverified
    unverified_count usage_available valid valid_citations validation_semantics verified
    verification version warnings writing_scope_policy year
    """.split()
)
KEYED_FIELDS = {
    "coverage_matrix",
    "method_by_section",
    "section_evidence_ids",
    "selected_items",
    "tier_counts",
    "upstream_results",
}
PUBLIC_FIELDS.update(KEYED_FIELDS)
ASSERTION_FIELDS = {
    "alignment_status",
    "atomic",
    "citation_scope",
    "classification_valid",
    "correct_kind",
    "source_end",
    "source_faithful",
    "source_quote",
    "source_start",
    "statement_id",
    "support_basis",
}
SCOPED_FIELDS = {
    "assertions": ASSERTION_FIELDS,
    "claim_records": {"text"},
    "unsupported_claims": ASSERTION_FIELDS,
}
SAFE_MAP_KEY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,119}")
PRIVATE_LOCATION = re.compile(
    r"(?i)(?:file://|/(?:Users|home|private/tmp|var/folders)/|"
    r"\b[a-z]:[\\/](?:Users|Documents)|\blocalhost\b|\b127\.0\.0\.1\b)"
)
URL = re.compile(r"https?://[^\s<>\[\])]+", re.IGNORECASE)
NONPUBLIC_HOST_TLDS = {"internal", "local", "localhost", "test"}


def public_text(value: str) -> bool:
    """Reject text that includes credentials, private locations or private links."""
    if PRIVATE_LOCATION.search(value) or _redact_trace_value(value) != value:
        return False
    for match in URL.finditer(value):
        try:
            url = urlsplit(match[0])
            host = (url.hostname or "").lower()
        except ValueError:
            return False
        if (
            "." not in host
            or host.rsplit(".", 1)[-1] in NONPUBLIC_HOST_TLDS
            or host.rsplit(".", 1)[-1].isdigit()
            or url.username is not None
            or url.password is not None
            or url.query
        ):
            return False
    return True


def _project(value, *, keyed=False, extra_fields=()):
    """Keep structured research values while dropping unapproved archive fields."""
    if isinstance(value, dict):
        result, omitted = {}, False
        for key, item in value.items():
            permitted = (
                SAFE_MAP_KEY.fullmatch(key) and public_text(key)
                if keyed
                else key in PUBLIC_FIELDS or key in extra_fields
            )
            if not permitted:
                omitted = True
                continue
            projected, reduced = _project(
                item, keyed=key in KEYED_FIELDS, extra_fields=SCOPED_FIELDS.get(key, ())
            )
            omitted |= reduced
            if projected is not None or item is None:
                result[key] = projected
        return result, omitted
    if isinstance(value, list):
        result, omitted = [], False
        for item in value:
            projected, reduced = _project(item, extra_fields=extra_fields)
            omitted |= reduced
            if projected is not None or item is None:
                result.append(projected)
        return result, omitted
    if isinstance(value, str) and not public_text(value):
        return None, True
    return value, False


def project_step(
    *, input=None, output=None, status="recorded", withhold_output=False
) -> dict | None:
    """Retain each observed side and explicitly identify reduced payloads."""
    entry = {"status": status}
    omitted = []
    for side, raw in (("input", input), ("output", output)):
        if raw is None:
            continue
        if not isinstance(raw, (dict, list)) or (side == "output" and withhold_output):
            omitted.append(side)
            continue
        value, reduced = _project(raw)
        if value is not None and (value or not reduced):
            entry[side] = value
        if reduced:
            omitted.append(side)
    if omitted:
        entry["omitted"] = omitted
    return entry if len(entry) > 1 else None


def project_report(archive, expected_sha256) -> dict | None:
    """Keep the exact manuscript and render an IEEE citation view from the archive."""
    report = archive.get("report")
    if not isinstance(report, dict):
        return None
    markdown = report.get("survey")
    if not isinstance(markdown, str) or not markdown.strip():
        return None
    faithfulness = (report.get("evaluation") or {}).get("faithfulness") or {}
    if (faithfulness.get("details") or {}).get("skipped_reason") == (
        "final_report_unavailable"
    ):
        # This archive contains a failure banner in the report field.
        return None
    digest = hashlib.sha256(markdown.encode()).hexdigest()
    if digest != expected_sha256 or not public_text(markdown):
        raise ValueError("Public report text failed its archive or publication check")

    display_markdown = markdown
    bibliography_span = None
    if REF_PATTERN.search(markdown):
        nodes = archive["nodes"]
        metadata = report.get("metadata") or {}
        reader = build_reader_report(
            ReportVersion(
                run_id=archive["run_id"],
                revision=1,
                report_language=metadata.get("report_language") or "en",
                survey=markdown,
            ),
            evidence_ledger={
                item["evidence_id"]: item
                for item in extract_report_evidence(nodes, report=report)
            },
            paper_catalog=paper_catalog_from_nodes(nodes),
            citation_style="ieee",
            writing_bindings=metadata.get("writing_bindings"),
        )
        display_markdown = reader.markdown
        bibliography_span = reader.bibliography_span
        if not public_text(display_markdown):
            raise ValueError("Public IEEE reference text failed its publication check")
    return {
        "markdown": markdown,
        "source_sha256": digest,
        "state": "partial" if report.get("partial") is True else "complete",
        "html": render_report_html(
            display_markdown, bibliography_span=bibliography_span
        ).html,
    }
