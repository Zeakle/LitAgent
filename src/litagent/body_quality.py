"""Share evidence preparation, analytical duties and body-only diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping

ANALYSIS_CONTRACT = "body-analysis-v1"
ANALYSIS_VERSION = "grounded-writing-v22"
ANALYSIS_OWNERSHIP_CONTRACT = "host-analysis-claim-ownership-v1"
ANALYSIS_CHAPTER_CONTRACT = "host-analysis-parent-chapter-v1"
UNIT_CITATION_SCOPE_CONTRACT = "unit-citation-scope-v1"
PAPER_BATCH_CONTRACT = "source-field-paper-batch-v1"
EVALUATION_ENCODING_CONTRACT = "native-schema-only-v1"
EXTRACTION_FIELD_SPAN_CONTRACT = "host-extraction-field-spans-v2"
EXTRACTION_SOURCE_LAYOUT_CONTRACT = "host-source-token-blocks-v1:size=8"
EXTRACTION_CLAIM_CONTRACT = "host-grounded-claims-v1"
METRIC_INFLECTION_CONTRACT = "accuracy-inflection-v1"
SUMMARY_ARRAY_WIRE_CONTRACT = "summary-array-items-v1"
NATIVE_ARRAY_WIRE_CONTRACT = "native-array-wire-v1"
NATIVE_SCHEMA_ENCODING_CONTRACT = "inline-local-schema-v1"
UNIT_CITATION_SCOPE_INSTRUCTION = """
SCIENTIFIC UNIT CITATION SCOPE unit-citation-scope-v1:
The host renders references only at the end of each native unit. Do not join
independent facts with an exposed semicolon in one unit: the earlier clause would
have no actual citation of its own. Use additional_sentences for separate findings,
each with its own exact supporting cards and complete conditions. Preserve all
relevant findings and core questions; never delete a fact, hide it as a notice,
weaken a qualifier or merely replace a semicolon with a comma to evade this rule.
Commas in conditions/lists and punctuation inside quotations are allowed. This
format check does not prove atomicity or scientific support. Comparison premises
still need their own sources and independently verified method/result attribution.
"""
APPLICABILITY_CONTRACT = "source-applicability-v1"
ANALYSIS_REPAIR_CAUSE_CONTRACT = "analysis-diagnostic-cause-v1"
APPLICABILITY_INSTRUCTION = """
SOURCE APPLICABILITY CONTRACT source-applicability-v1:
Read each actual cited original for the task, domain, population, experimental setup
and qualifiers of its finding. Preserve these boundaries in the scientific sentence,
including summaries and repairs. A task-specific system's result is not a general
property of its component technique. For example, improved cardiac MRI reconstruction
using adapted visual encoders does not alone establish improved general visual-task
performance from parameter-efficient fine-tuning. Conversely, do not reject a result
because it concerns a specialized domain when the sentence preserves that domain.
Only explicit evidence can establish transfer to another task or domain. A method
name, a source-located fact card or an extracted field is not proof of transfer.
Do not force an available paper to answer the user's broader question. Retain a
relevant bounded finding or disclose the unanswered requirement; do not hide a
scientific premise in a scope notice or omit a core question to pass evaluation.
During assessment, the required conditions check includes these applicability
boundaries even if the report omitted them. A broader conclusion or a changed task
is unsupported unless its own actual citations establish that broader claim.
"""
ROLES = {
    "introduction": "Define the user's problem and scope with sourced scientific background.",
    "taxonomy": "Explain classification criteria and relationships; distinguish related but different efficiency or task objectives.",
    "methods": "Explain each requested method's mechanism, contribution, evidence and applicability boundary. Naming a method is not an analysis.",
    "experiments": "Compare methods along requested dimensions. Retain datasets, baselines, training settings, metrics, units and qualifiers. Explicitly mark unmatched conditions as incomparable; do not invent a ranking.",
    "open_problems": "Explain concrete source-supported limitations or trade-offs and their implications. A missing selected source does not establish a field-wide research gap.",
    "abstract": "Concisely synthesize the research question, central findings and a key limitation from existing scientific body units.",
    "conclusion": "Answer the question through evidence-backed choices and trade-offs. Explain remaining boundaries; use a different emphasis from the abstract.",
}
ANALYSIS_INSTRUCTION = (
    """
BODY ANALYSIS CONTRACT body-analysis-v1:
Use the preparation table as a source index, never as a semantic verdict. Literal
match misses do not prove absent material; read the originals and preserve explicit
requirements. Answer
core questions with explanations grounded in original sources. Organize meaningful
paragraphs around mechanisms, comparisons and limitations; do not produce a chapter
with only method names or unrelated facts. More words, citations or assertions do
not establish analytical depth. Use additional_blocks/additional_sentences when
needed; stay within the existing 128-unit and output budgets. Preserve experimental
conditions and distinguish data, parameter and computational efficiency when relevant.
ABSTRACT/CONCLUSION OVERRIDE: every summary unit has ONLY text and body_references
(an array of section_key/block_index/sentence_index objects). Write a concise atomic
scientific synthesis in text; select existing SCIENTIFIC body units supplying every
premise. The host derives cards and citations from those units. Do not emit kind,
fact_card_ids, arbitrary references or new facts. Never infer broader causality,
quantifiers, rankings, numerical changes or missing conditions. Summaries are still
independently evaluated against the original papers. Avoid copying entire body
sentences or repeating the abstract verbatim as the conclusion.
"""
    + APPLICABILITY_INSTRUCTION
)


def _digest(value):
    """Hash the complete preparation identity without source truncation."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def build_analysis_plan(query, requirements, cards, section_ids=None):
    """Index complete candidate originals without claiming semantic support."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or not isinstance(cards, Mapping)
    ):
        raise ValueError("analysis_preparation_invalid")
    if not cards or any(
        not isinstance(c, Mapping) or not c.get("source_quote") for c in cards.values()
    ):
        raise ValueError("analysis_sources_missing")
    rows = []
    for cid, card in cards.items():
        rows.append(
            {
                "fact_card_id": cid,
                "paper_id": card.get("paper_id"),
                "evidence_id": card.get("evidence_id"),
                "source_sha256": card.get("content_sha256"),
                "finding_candidate": card.get("finding_candidate"),
                "source_scope": card.get("content_scope", "unknown"),
                "candidate_fields": {
                    k: card.get(k)
                    for k in (
                        "method",
                        "conditions",
                        "dataset",
                        "metric",
                        "value",
                        "unit",
                        "qualifiers",
                    )
                },
                "semantic_status": "not_verified",
            }
        )
    ids = set(cards)
    sections = section_ids or {key: list(cards) for key in ROLES}
    if any(not set(values).issubset(ids) for values in sections.values()):
        raise ValueError("analysis_source_identity_invalid")
    return {
        "contract": ANALYSIS_CONTRACT,
        "research_question": query,
        "requirements": [
            {
                **r,
                "candidate_card_ids": [
                    cid
                    for cid, card in cards.items()
                    if r.get("kind") == "question"
                    or r.get("text", "").casefold() in card["source_quote"].casefold()
                ],
                "match_rule": (
                    "selected_pool"
                    if r.get("kind") == "question"
                    else "literal_source_match"
                ),
                "answer_status": "not_verified",
            }
            for r in (requirements or [])
        ],
        "sources": rows,
        "section_duties": [
            {
                "section_key": k,
                "duty": v,
                "candidate_card_ids": list(sections.get(k, [])),
            }
            for k, v in ROLES.items()
        ],
        "requirements_status": "not_verified",
        "missing_fields": {
            k: [r["fact_card_id"] for r in rows if not r["candidate_fields"].get(k)]
            for k in ("method", "conditions", "dataset", "metric", "value")
        },
        "evidence_identity_sha256": _digest(cards),
    }


def body_diagnostics(survey):
    """Count academic prose separately from host appendices and bibliography."""
    from litagent.report_render import build_report_source_units

    split = re.search(r"(?m)^#{1,2}\s+(?:附录|Appendix|参考文献|References)\b", survey)
    body = survey[: split.start()] if split else survey
    units = build_report_source_units(body)
    clean = lambda text: re.sub(r"\s+", " ", re.sub(r"\[E:[^\]]+\]", "", text)).strip()
    texts = [clean(u.text) for u in units]
    texts = [t for t in texts if t and not t.startswith("#")]
    counts = Counter(texts)
    return {
        "contract": ANALYSIS_CONTRACT,
        "report_sha256": hashlib.sha256(survey.encode()).hexdigest(),
        "academic_characters": len(body),
        "non_body_characters": len(survey) - len(body),
        "academic_units": len(texts),
        "repeated_units": sum(n - 1 for n in counts.values()),
        "interpretation": "diagnostics_only_not_a_quality_score",
    }


def analysis_claim_ids_by_section(assertions, structure):
    """Assign scientific claim identities using host sections without text matching."""
    from litagent.review_structure import structure_sections

    sections = {}
    for row in structure:
        uid, section = row.get("source_unit_id"), row.get("section")
        if (
            not isinstance(uid, str)
            or not uid
            or uid in sections
            or not isinstance(section, str)
        ):
            raise ValueError("evaluation_global_input_identity_invalid")
        sections[uid] = section
    titles = {
        key: {
            dict(structure_sections("academic_v2", lang))[key]
            for lang in ("en", "zh-CN")
        }
        for key in ROLES
    }
    grouped, seen = {key: [] for key in ROLES}, set()
    for claim in assertions:
        cid = claim.get("claim_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("evaluation_global_input_identity_invalid")
        seen.add(cid)
        uid = claim.get("source_unit_id")
        if sections and uid not in sections:
            raise ValueError("evaluation_global_input_identity_invalid")
        section = sections.get(uid, claim.get("section"))
        if sections and "section" in claim and claim["section"] != section:
            raise ValueError("evaluation_global_input_identity_invalid")
        if claim.get("kind") == "scientific":
            for key, allowed in titles.items():
                if section in allowed:
                    grouped[key].append(cid)
    return grouped


def validate_analysis_decisions(decisions, assertions, supported_ids, structure):
    """Separate semantic adequacy from host ownership and final scientific support."""
    rows = [d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions]
    by_id = {a["claim_id"]: a for a in assertions}
    grouped = analysis_claim_ids_by_section(assertions, structure)
    if len(rows) != len(ROLES) or {r.get("section_key") for r in rows} != set(ROLES):
        raise ValueError("evaluation_analysis_coverage_invalid")
    for row in rows:
        ids = row.get("claim_ids", [])
        allowed = grouped[row["section_key"]]
        errors = []
        model_adequate = row.get("model_adequate", row.get("adequate")) is True
        if not model_adequate:
            errors.append("analysis_semantics_inadequate")
        if not ids:
            errors.append("analysis_claims_missing")
        if len(ids) != len(set(ids)):
            errors.append("analysis_duplicate_claim")
        for cid in ids:
            if cid not in by_id:
                errors.append("analysis_unknown_claim")
            elif cid not in allowed:
                errors.append("analysis_wrong_section_claim")
            elif cid not in supported_ids:
                errors.append("analysis_claim_not_supported")
        row["model_adequate"] = model_adequate
        row["adequate"] = not errors
        row["host_validation"] = {
            "status": "failed" if errors else "passed",
            "failure_codes": list(dict.fromkeys(errors)),
            "allowed_claim_ids": allowed,
        }
    return {
        "contract": ANALYSIS_CONTRACT,
        "status": "passed" if all(r["adequate"] for r in rows) else "failed",
        "sections": rows,
    }


def analysis_repair_diagnostics(details):
    """Authorize expansion only from complete analysis decisions and exact claims."""
    if details.get("body_analysis_contract") != ANALYSIS_CONTRACT:
        return []
    by_id = {c["claim_id"]: c for c in details.get("assertions", [])}
    output = {}
    from litagent.review_structure import structure_sections

    titles = {
        key: {
            dict(structure_sections("academic_v2", lang))[key]
            for lang in ("en", "zh-CN")
        }
        for key in ROLES
    }
    for row in details.get("analysis_check", {}).get("sections", []):
        if row.get("adequate") is not False or row.get("section_key") not in titles:
            continue
        if row.get("model_adequate") is True:
            # A host rejection of cited science does not authorize rewriting the
            # section's supported claims as if its analytical content were missing.
            continue
        if any(
            code in row.get("host_validation", {}).get("failure_codes", [])
            for code in (
                "analysis_unknown_claim",
                "analysis_wrong_section_claim",
                "analysis_duplicate_claim",
            )
        ):
            # Invalid assessment ownership cannot authorize changing valid prose.
            continue
        for cid in row.get("claim_ids", []):
            claim = by_id.get(cid, {})
            if (
                claim.get("status") != "supported"
                or claim.get("kind") != "scientific"
                or claim.get("section") not in titles[row["section_key"]]
            ):
                continue
            output[cid] = {
                **claim,
                "reason_code": "body_analysis_incomplete",
                "diagnostic_type": "analysis_depth",
                "diagnostic_source": "survey-evaluation-v3",
                "alignment_status": "exact",
                "citation_scope": "exact",
                "explanation": row.get("explanation", ""),
                "preserve_supported_claim": True,
            }
    return list(output.values())


def bound_repair_diagnostics(details):
    """Keep scientific failures separate while sharing the same bounded repair loop."""
    science = list(details.get("unsupported_claims") or [])
    known = {d.get("claim_id") for d in science}
    return science + [
        d for d in analysis_repair_diagnostics(details) if d["claim_id"] not in known
    ]
