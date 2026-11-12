"""Share evidence preparation, analytical duties and body-only diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping

ANALYSIS_CONTRACT = "body-analysis-v1"
WRITER_CONTENT_CONTRACT = "grounded-writer-content-v1"
NATIVE_REPORT_INPUT_CONTRACT = "host-report-input-v1"
WRITER_CONTENT_INSTRUCTION = """
CONTENT CONTRACT grounded-writer-content-v1:
Each authored text unit should express one independently testable proposition.
Keep the method, dataset, baseline, value, unit and qualifiers of that proposition
together. Split independent mechanisms, experimental results and recommendations
into separate units; splitting does not reduce their assertion count. Punctuation
alone does not establish atomicity or scientific error. Never hide multiple facts
in an enumeration, replace a semicolon, or weaken conditions to evade checking.

Give each chapter a distinct job: introduce the question and scope briefly; explain
classification criteria in taxonomy; develop source-backed mechanisms in methods;
compare actual results and conditions in experiments; discuss concrete supported
limitations in open_problems. Do not repeat the full mechanism list in each chapter.
Do not infer speed, parameter cost, suitability or universal superiority from a
mechanism alone. A comparison needs the actual premises for both methods, with
their original conditions preserved; unmatched settings cannot support a ranking.

The host owns retrieval disclosure and material-gap records. Do not fill academic
paragraphs with freely authored claims that selected sources lack a table, setting
or result. A missing extracted metadata field does not mean the original omits it.
Read the full supplied originals. Do not invent a field-wide gap or present missing
selected material as a method's scientific limitation. Retain all core questions,
named methods and required conditions; gaps must remain unresolved rather than be
replaced by unrelated facts, generic caveats or an empty report.

Abstract and conclusion should select a few central findings, not enumerate every
method and repeat its mechanism. Normally focus on two or three findings in each,
within the existing host summary limits; this is guidance, not a scientific verdict
based on sentence count. Use different emphases, preserve necessary conditions and
direct support, and do not add recommendations or causality beyond the premises.
Do not delete substantive body findings just to shorten summaries or fit capacity.
The independent evaluator counts every atomic assertion across body and summaries,
not JSON units. Its unchanged capacity and scientific gates still apply. Do not
omit hard assertions, relabel science as organization/method, or claim verification
from a source-located card or a model-generated classification.

For Reviewer or global review, apply these criteria to exact host-bound units and
report concrete problems, not a rewrite. Concision targets are not automatic errors
or exemptions. A scientifically supported finding and complete core coverage remain
necessary; these writing instructions never provide scientific evidence.
"""
LEGACY_SUMMARY_SELECTION = """Normally focus on two or three findings in each,
within the existing host summary limits; this is guidance, not a scientific verdict
based on sentence count."""
SCOPED_SUMMARY_SELECTION = """Use host summary_constraints; there is no requirement to fill the capacity.
Units, findings and atomic assertions are distinct counts."""


def writer_content_instruction(summary_constraints=None):
    """Preserve historical prompt bytes unless explicit summary scope is supplied."""
    if summary_constraints is None:
        return WRITER_CONTENT_INSTRUCTION
    return WRITER_CONTENT_INSTRUCTION.replace(
        LEGACY_SUMMARY_SELECTION, SCOPED_SUMMARY_SELECTION
    )


NATIVE_OWNERSHIP_INSTRUCTION = """
HOST NATIVE OWNERSHIP grounded-writer-v1: each supplied :native: fragment is one
exact host-bound authored unit with its actual citation set. Those citations apply
to that complete native unit, including its internal punctuation. Extract every
atomic assertion within it; shared citations remain attached to each assertion.
Do not borrow citations across native fragments or infer ownership from semicolons.
This host binding proves location and ownership only, never scientific support.
Each scientific assertion still requires complete support, correct conditions,
attribution and source fidelity; independently detect omitted or compound claims.
"""
ANALYSIS_VERSION = "grounded-writing-v22"
ANALYSIS_OWNERSHIP_CONTRACT = "host-analysis-claim-ownership-v1"
ANALYSIS_CHAPTER_CONTRACT = "host-analysis-parent-chapter-v1"
UNIT_CITATION_SCOPE_CONTRACT = "unit-citation-scope-v1"
PAPER_BATCH_CONTRACT = "source-field-paper-batch-v1"
EVALUATION_ENCODING_CONTRACT = "native-schema-only-v1"
ADAPTIVE_ASSESSMENT_INPUT_CONTRACT = "host-citation-support-aliases-v2"
EXTRACTION_FIELD_SPAN_CONTRACT = "host-extraction-field-spans-v2"
EXTRACTION_SOURCE_LAYOUT_CONTRACT = "host-source-token-blocks-v1:size=8"
EXTRACTION_CLAIM_CONTRACT = "host-grounded-claims-v1"
EVIDENCE_PREPARATION_POLICY = "question_focused_v1"
PREPARATION_MAX_CHUNKS = 12
PREPARATION_MAX_CHARS = 12_000


def check_preparation_policy(value):
    """Accept only the server-owned opt-in preparation policy or historical None."""
    if value is not None and (
        type(value) is not str or value != EVIDENCE_PREPARATION_POLICY
    ):
        raise ValueError("evidence_preparation_policy_invalid")
    return value


def checked_preparation_chunks(paper):
    """Validate paper-owned canonical chunks without repairing their source hashes."""
    from litagent.rag.models import ContentChunk, ContentScope, normalize_paper_identity

    pid = paper.get("paper_id")
    raw = paper.get("chunks", [])
    if not isinstance(pid, str) or not pid or not isinstance(raw, list):
        raise ValueError("evidence_preparation_source_invalid")
    checked = []
    keys = set()
    for value in raw:
        chunk = ContentChunk.model_validate(value)
        if (
            normalize_paper_identity(chunk.paper_id) != normalize_paper_identity(pid)
            or chunk.chunk_key in keys
            or chunk.content_hash != hashlib.sha256(chunk.text.encode()).hexdigest()
        ):
            raise ValueError("evidence_preparation_source_identity_invalid")
        keys.add(chunk.chunk_key)
        checked.append(chunk.model_dump(mode="json"))
    abstract = paper.get("abstract") or ""
    if not isinstance(abstract, str):
        raise ValueError("evidence_preparation_source_invalid")
    if abstract.strip() and not any(c["content_scope"] == "abstract" for c in checked):
        if "abstract" in keys:
            raise ValueError("evidence_preparation_source_identity_invalid")
        checked.insert(
            0,
            ContentChunk.from_text(
                paper_id=pid,
                chunk_key="abstract",
                text=abstract,
                section="abstract",
                content_scope=ContentScope.ABSTRACT,
            ).model_dump(mode="json"),
        )
    return checked


def evidence_preparation_identity():
    """Bind the complete new preparation path without changing absent identities."""
    import inspect

    from litagent.agents.extraction_strategy import (
        _CONTEXT_STOP_WORDS,
        LLMStrategy,
        _context_words,
        _field_source_chunks,
        _preparation_request_identity,
        _preparation_request_scope,
        _strategy_context,
        focused_extraction_context,
    )
    from litagent.agents.extractor import ExtractorWorker
    from litagent.rag.pdf_parser import (
        _NUMBERED_HEADING,
        NUMBERED_HEADING_POLICY,
        PyMuPDFParser,
    )
    from litagent.rag.targeted_fulltext import TargetedFulltext
    from litagent.safety.budget import CostBudget
    from litagent.safety.planned_budget import (
        PREPARATION_BUDGET_CONTRACT,
        build_budget_plan,
        preparation_budget_plan,
        settled_preparation_budget,
    )

    return hashlib.sha256(
        json.dumps(
            [
                EVIDENCE_PREPARATION_POLICY,
                PREPARATION_MAX_CHUNKS,
                PREPARATION_MAX_CHARS,
                NUMBERED_HEADING_POLICY,
                _NUMBERED_HEADING.pattern,
                _NUMBERED_HEADING.flags,
                sorted(_CONTEXT_STOP_WORDS),
                PREPARATION_BUDGET_CONTRACT,
                inspect.getsource(preparation_budget_plan),
                inspect.getsource(build_budget_plan),
                inspect.getsource(settled_preparation_budget),
                inspect.getsource(CostBudget.activate_plan),
                inspect.getsource(_preparation_request_identity),
                inspect.getsource(_preparation_request_scope),
                inspect.getsource(checked_preparation_chunks),
                inspect.getsource(_context_words),
                inspect.getsource(focused_extraction_context),
                inspect.getsource(_strategy_context),
                inspect.getsource(_field_source_chunks),
                inspect.getsource(LLMStrategy._strict_resources),
                inspect.getsource(LLMStrategy._prepare_grounded_batch),
                inspect.getsource(LLMStrategy.prepare_batch),
                inspect.getsource(LLMStrategy.preflight_batches),
                inspect.getsource(LLMStrategy.extract_batch),
                inspect.getsource(PyMuPDFParser.__init__),
                inspect.getsource(PyMuPDFParser._parse_quality_document),
                inspect.getsource(TargetedFulltext.__init__),
                inspect.getsource(TargetedFulltext.enrich),
                inspect.getsource(ExtractorWorker.execute),
            ],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


METRIC_INFLECTION_CONTRACT = "accuracy-inflection-v1"
SUMMARY_ARRAY_WIRE_CONTRACT = "summary-array-items-v1"
NATIVE_ARRAY_WIRE_CONTRACT = "native-array-wire-v1"
NATIVE_SCHEMA_ENCODING_CONTRACT = "inline-local-schema-v1"
NATIVE_PROMPT_COMPOSITION_CONTRACT = "native-array-prompt-composition-v1"
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


def report_repair_diagnostics(details):
    """Locate genuine report defects without treating support dependencies as defects."""
    from litagent.eval.results import CONTRACT, contract_identity

    stages = details.get("stages", {})
    if (
        details.get("evaluation_result_contract") != CONTRACT
        or details.get("evaluation_result_identity") != contract_identity()
        or stages.get("global_check") != "completed"
        or details.get("inventory_check", {}).get("status") != "passed"
    ):
        return []
    claims = {c["claim_id"]: c for c in details.get("assertions", [])}
    regions = stages.get("global_sections", {})
    output = []
    if regions.get("issues", {}).get("status") == "passed":
        for diagnostic in details.get("unsupported_claims", []):
            if diagnostic.get("diagnostic_owner") != "report":
                continue
            claim = claims.get(diagnostic.get("claim_id"), {})
            if claim.get("status") != "supported":
                continue
            for issue in details.get("global_issues", []):
                if (
                    issue.get("owner") == "report"
                    and issue.get("code") in {"category_count", "repetition"}
                    and diagnostic.get("reason_code")
                    == "evaluation_report_" + issue["code"]
                    and claim.get("claim_id") in issue.get("claim_ids", [])
                    and issue.get("source_unit_id") == claim.get("source_unit_id")
                    and diagnostic.get("explanation") == issue.get("explanation")
                    and diagnostic.get("diagnostic_source") == "survey-evaluation-v3"
                    and all(
                        diagnostic.get(key) == value
                        for key, value in claim.items()
                        if key not in {"explanation", "reason_code"}
                    )
                ):
                    output.append(diagnostic)
                    break

    def locate(rows, field, code):
        for row in rows:
            if (
                row.get(field) is not False
                or not isinstance(row.get("explanation"), str)
                or not row["explanation"].strip()
            ):
                continue
            for cid in row.get("claim_ids", []):
                claim = claims.get(cid, {})
                if (
                    claim.get("status") != "supported"
                    or claim.get("kind") != "scientific"
                ):
                    continue
                output.append(
                    {
                        **claim,
                        "statement_id": claim["claim_id"],
                        "reason_code": code,
                        "diagnostic_owner": "report",
                        "diagnostic_source": "survey-evaluation-v3",
                        "alignment_status": "exact",
                        "citation_scope": "exact",
                        "explanation": row["explanation"],
                        "preserve_supported_claim": True,
                    }
                )

    if regions.get("coverage", {}).get("status") == "passed":
        locate(
            details.get("core_coverage", []),
            "answered",
            "evaluation_report_core_coverage",
        )
    if regions.get("limitations", {}).get("status") == "passed":
        locate(
            [details.get("limitations_check", {})],
            "model_adequate",
            "evaluation_report_limitations",
        )
    if regions.get("conflicts", {}).get("status") == "passed":
        locate(
            [
                {
                    "consistent": False,
                    "claim_ids": [pair.get("claim_a"), pair.get("claim_b")],
                    "explanation": pair.get("reason"),
                }
                for pair in details.get("conflicts", [])
            ],
            "consistent",
            "evaluation_report_contradiction",
        )
    return output


def bound_repair_diagnostics(details):
    """Keep scientific failures separate while sharing the same bounded repair loop."""
    science = list(details.get("unsupported_claims") or [])
    report = report_repair_diagnostics(details)
    known = {d.get("claim_id") for d in science if d not in report}
    return (
        science
        + [d for d in report if d not in science]
        + [
            d
            for d in analysis_repair_diagnostics(details)
            if d["claim_id"] not in known
        ]
    )
