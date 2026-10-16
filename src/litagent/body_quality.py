"""Share evidence preparation, analytical duties and body-only diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping

ANALYSIS_CONTRACT = "body-analysis-v1"
ANALYSIS_VERSION = "grounded-writing-v22"
ROLES = {
    "introduction": "Define the user's problem and scope with sourced scientific background.",
    "taxonomy": "Explain classification criteria and relationships; distinguish related but different efficiency or task objectives.",
    "methods": "Explain each requested method's mechanism, contribution, evidence and applicability boundary. Naming a method is not an analysis.",
    "experiments": "Compare methods along requested dimensions. Retain datasets, baselines, training settings, metrics, units and qualifiers. Explicitly mark unmatched conditions as incomparable; do not invent a ranking.",
    "open_problems": "Explain concrete source-supported limitations or trade-offs and their implications. A missing selected source does not establish a field-wide research gap.",
    "abstract": "Concisely synthesize the research question, central findings and a key limitation from existing scientific body units.",
    "conclusion": "Answer the question through evidence-backed choices and trade-offs. Explain remaining boundaries; use a different emphasis from the abstract.",
}
ANALYSIS_INSTRUCTION = """
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


def validate_analysis_decisions(decisions, assertions, supported_ids, structure):
    """Require complete semantic duties anchored to supported claims in their section."""
    rows = [d.model_dump() if hasattr(d, "model_dump") else d for d in decisions]
    by_id = {a["claim_id"]: a for a in assertions}
    sections = {r["source_unit_id"]: r["section"] for r in structure}
    from litagent.review_structure import structure_sections

    titles = {
        key: {
            dict(structure_sections("academic_v2", lang))[key]
            for lang in ("en", "zh-CN")
        }
        for key in ROLES
    }
    if len(rows) != len(ROLES) or {r.get("section_key") for r in rows} != set(ROLES):
        raise ValueError("evaluation_analysis_coverage_invalid")
    for row in rows:
        ids = row.get("claim_ids", [])
        valid = (
            bool(ids)
            and len(ids) == len(set(ids))
            and all(
                cid in supported_ids
                and cid in by_id
                and by_id[cid].get("kind") == "scientific"
                and by_id[cid].get(
                    "section", sections.get(by_id[cid].get("source_unit_id"))
                )
                in titles[row["section_key"]]
                for cid in ids
            )
        )
        row["adequate"] = row.get("adequate") is True and valid
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
