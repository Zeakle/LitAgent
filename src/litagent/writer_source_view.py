"""Project authenticated writer sources without transmitting semantic candidates."""

import copy
import hashlib
import inspect
import re

CONTRACT = "source-only-writer-v1"
GROUNDING_INSTRUCTION = """
Ground every comparison, negative statement and limitation in the actual cited
originals. Not reported in the selected text does not mean absent in a method or
the field. Describe observed settings and findings instead of inventing missing
ablations, costs, universal rankings or negative applicability conclusions.
Keep useful synthesis and all core methods with their necessary conditions;
do not replace them with caveats. Summaries inherit no new premises. Reviewer
must locate unsupported inferences; repairs retain only source-backed claims.
"""
SUMMARY_BODY_VIEW_INSTRUCTION = """
body_units maps host IDs to complete frozen prose; body_sections preserves
chapter and paragraph order as lists of those IDs. Read both as context.
Never delete core requirements, weaken conditions or pack independent facts to
fit: unit ceilings are not assertion allowances.
"""
_PAIR = ("writer_source_view", "writer_source_view_identity")
_LOOKUP_FIELDS = ("source_ref", "evidence_id", "semantic_status")
_CANDIDATE_FIELDS = (
    "method",
    "conditions",
    "dataset",
    "metric",
    "value",
    "unit",
    "qualifiers",
)
_PACKET_FIELDS = (
    "contract",
    "query",
    "language",
    "requirements",
    "section_duties",
    "chapter_recommendations",
    "scope_policy",
    "sources",
    "cards",
)
_PARENTS = {
    "contract": "flat-writer-envelope-v1",
    "global_findings_contract": "anchored-global-findings-v2",
    "summary_constraints_contract": "effective-summary-limits-v1",
    "repair_feedback_contract": "host-repair-feedback-v1",
    "reference_contract": "ordered-reference-set-v1",
    "transport_revision": "host-lossless-transport-v1",
    "support_validation_revision": "host-support-receipts-v2",
    "inventory_recovery_contract": "bounded-inventory-recovery-v2",
    "classification_review_contract": "independent-assertion-kinds-v1",
    "repair_mapping_contract": "capacity-source-map-v4",
    "repair_guidance_contract": "host-structured-repair-v1",
    "repair_guidance_wire_revision": "host-repair-task-view-v1",
    "repair_guidance_capacity_rule": "complete-locked-request-reservation-v1",
    "evaluation_response_contract": "host-bound-evaluation-v2",
    "repair_prompt_revision": "concise-grounded-repair-v1",
}
_AUGMENTATION_FIELDS = {
    "writing_plan",
    "host_priority_card_refs",
    "writing_envelope",
    "summary_constraints",
    "grounding_instruction",
}
_STAGE_FIELDS = {
    "chapter": set(_PACKET_FIELDS)
    | {"requested_body_section", "prior_chapters_unverified", "writing_plan_progress"},
    "review": {
        "query",
        "language",
        "draft",
        "units",
        "sources",
        "cards",
        "requirements",
        "duties",
        "input_contract",
        "report_layout",
        "report_sha256",
        "citation_projection",
        "citation_projection_contract",
    },
    "summary": {
        "query",
        "language",
        "body_units",
        "body_sections",
        "requirements",
        "summary_limits_per_section",
    },
    "repair": set(_PACKET_FIELDS)
    | {
        "current_body",
        "body_sha256",
        "targets",
        "cause",
        "previous_local_failure",
        "capacity_diagnostic",
        "transport_revision",
        "repair_input_contract",
        "capacity_target_binding",
        "target_explanations",
        "repair_guidance",
        "repair_feedback_limits",
        "current_body_format",
        "body_units",
        "summary_source_body_sha256",
    },
}


def _digest(value):
    """Use the existing canonical host digest, including its JSON spacing."""
    from litagent.quality import digest

    return digest(value)


def _invalid():
    """Fail closed without including source material in the exception text."""
    raise ValueError("writer_source_view_invalid")


def _hex_digest(value):
    """Recognize exact SHA-256 records without coercion or normalization."""
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_view_identity() -> str:
    """Bind only this optional projection and its explicit field boundaries."""
    from litagent.writing_v2 import _summary_request_bound, summary_request_upper_bound

    return _digest(
        [
            CONTRACT,
            GROUNDING_INSTRUCTION,
            SUMMARY_BODY_VIEW_INSTRUCTION,
            inspect.getsource(summary_request_upper_bound),
            inspect.getsource(_summary_request_bound),
            _PAIR,
            _LOOKUP_FIELDS,
            _CANDIDATE_FIELDS,
            _PACKET_FIELDS,
            _PARENTS,
            sorted(_AUGMENTATION_FIELDS),
            {key: sorted(value) for key, value in _STAGE_FIELDS.items()},
            *[
                inspect.getsource(function)
                for function in (
                    _digest,
                    _invalid,
                    _hex_digest,
                    selected_source_view,
                    selected_summary_body_view,
                    summary_body_view,
                    validate_summary_body_view,
                    source_view_fields,
                    _source_lookup,
                    _payload_stage,
                    project_source_view,
                    make_source_view_record,
                    validate_source_view_record,
                )
            ],
        ]
    )


def selected_source_view(envelope) -> bool:
    """Authenticate the opt-in pair without recursing into envelope validation."""
    if envelope is None:
        return False
    if type(envelope) is not dict:
        raise ValueError("writer_source_view_policy_invalid")
    if not any(key in envelope for key in _PAIR):
        return False
    from litagent.eval.batched import selected_response_contract
    from litagent.writing_flat import selected_repair_prompt

    if (
        type(envelope.get(_PAIR[0])) is not str
        or envelope[_PAIR[0]] != CONTRACT
        or type(envelope.get(_PAIR[1])) is not str
        or envelope[_PAIR[1]] != source_view_identity()
        or any(envelope.get(key) != value for key, value in _PARENTS.items())
        or type(envelope.get("capacity")) is not dict
        or not selected_response_contract(envelope)
        or not selected_repair_prompt(envelope)
    ):
        raise ValueError("writer_source_view_policy_invalid")
    return True


def source_view_fields(envelope) -> dict:
    """Return local ownership fields only for an authenticated selection."""
    return (
        {key: envelope[key] for key in _PAIR} if selected_source_view(envelope) else {}
    )


def selected_summary_body_view(envelope) -> bool:
    """Select the compact summary premises only on the independent wire."""
    if not selected_source_view(envelope):
        return False
    from litagent.eval.decision_output import selected

    return selected(envelope)


def summary_body_view(units, envelope) -> dict | None:
    """Keep complete ordered prose and paragraphs; retain citation ownership locally."""
    if not selected_summary_body_view(envelope):
        return None
    from litagent.writing_v2 import BODY

    if type(units) is not dict or not units:
        _invalid()
    if list(units) != [f"u{i}" for i in range(1, len(units) + 1)]:
        _invalid()
    texts, sections = {}, {}
    for uid, unit in units.items():
        if type(unit) is not dict or set(unit) != {"path", "text", "fact_card_refs"}:
            _invalid()
        path, refs = unit["path"], unit["fact_card_refs"]
        if (
            type(path) is not list
            or len(path) != 3
            or path[0] not in BODY
            or any(type(i) is not int or i < 0 for i in path[1:])
            or type(refs) is not list
            or any(type(ref) is not str for ref in refs)
        ):
            _invalid()
        section, paragraph, sentence = path
        groups = sections.setdefault(section, [])
        if paragraph == len(groups):
            groups.append([])
        if paragraph >= len(groups) or sentence != len(groups[paragraph]):
            _invalid()
        # Authored positions must be in sequence, not merely present somewhere.
        if paragraph != len(groups) - 1:
            _invalid()
        groups[paragraph].append(uid)
        texts[uid] = unit["text"]
    view = {"body_units": texts, "body_sections": sections}
    validate_summary_body_view(view, envelope)
    return view


def validate_summary_body_view(payload, envelope) -> None:
    """Require a complete ordered partition without searching for equal prose."""
    if not selected_summary_body_view(envelope):
        if "body_sections" in payload:
            _invalid()
        return
    from litagent.writing_v2 import BODY

    texts, sections = payload.get("body_units"), payload.get("body_sections")
    if (
        type(texts) is not dict
        or not texts
        or len(texts) > envelope["body_max_units"]
        or list(texts) != [f"u{i}" for i in range(1, len(texts) + 1)]
        or any(
            type(text) is not str
            or not text.strip()
            or len(text) > envelope["capacity"]["max_unit_characters"]
            for text in texts.values()
        )
        or type(sections) is not dict
        or not sections
        or list(sections) != [key for key in BODY if key in sections]
    ):
        _invalid()
    flattened = []
    for groups in sections.values():
        if type(groups) is not list or not groups:
            _invalid()
        for group in groups:
            if (
                type(group) is not list
                or not group
                or any(type(uid) is not str for uid in group)
            ):
                _invalid()
            flattened.extend(group)
    if flattened != list(texts):
        _invalid()


def _source_lookup(packet):
    """Authenticate complete originals and the ordered canonical cN directory."""
    if (
        type(packet) is not dict
        or set(packet) != set(_PACKET_FIELDS)
        or packet.get("contract") != "grounded-writer-v1"
        or packet.get("scope_policy") != "frozen_union_v1"
        or type(packet.get("sources")) is not dict
        or not packet["sources"]
        or type(packet.get("cards")) is not dict
        or not packet["cards"]
    ):
        _invalid()
    sources, cards = packet["sources"], packet["cards"]
    for sid, source in sources.items():
        if (
            type(sid) is not str
            or type(source) is not dict
            or set(source) != {"paper_id", "source_quote", "content_sha256"}
            or type(source["paper_id"]) is not str
            or not source["paper_id"]
            or type(source["source_quote"]) is not str
            or not source["source_quote"]
            or source["content_sha256"]
            != hashlib.sha256(source["source_quote"].encode()).hexdigest()
            or sid != "s" + _digest([source["paper_id"], source["source_quote"]])[:20]
        ):
            _invalid()
    if list(cards) != [f"c{i}" for i in range(1, len(cards) + 1)]:
        _invalid()
    lookup, evidence, used_sources = {}, set(), set()
    for ref, card in cards.items():
        if (
            type(card) is not dict
            or set(card)
            != set(_LOOKUP_FIELDS)
            | {
                "finding_candidate",
                "document_subject",
                "content_scope",
                "candidate_fields",
            }
            or type(card["source_ref"]) is not str
            or card["source_ref"] not in sources
            or type(card["evidence_id"]) is not str
            or not card["evidence_id"]
            or card["evidence_id"] in evidence
            or card["semantic_status"] != "not_verified"
            or type(card["candidate_fields"]) is not dict
            or set(card["candidate_fields"]) != set(_CANDIDATE_FIELDS)
        ):
            _invalid()
        evidence.add(card["evidence_id"])
        used_sources.add(card["source_ref"])
        lookup[ref] = {key: copy.deepcopy(card[key]) for key in _LOOKUP_FIELDS}
    if used_sources != set(sources):
        _invalid()
    return lookup


def _payload_stage(payload):
    """Select one actual producer shape before applying its closed root set."""
    if type(payload) is not dict:
        _invalid()
    stages = [
        stage
        for stage, field in (
            ("chapter", "requested_body_section"),
            ("review", "units"),
            ("repair", "current_body"),
        )
        if field in payload
    ]
    if not stages and "body_units" in payload:
        stages = ["summary"]
    if len(stages) != 1:
        _invalid()
    stage = stages[0]
    if not set(payload) <= _STAGE_FIELDS[stage] | _AUGMENTATION_FIELDS:
        _invalid()
    if "writing_envelope" in payload and type(payload["writing_envelope"]) is not dict:
        _invalid()
    if any(key in payload.get("writing_envelope", {}) for key in _PAIR):
        _invalid()
    return stage


def project_source_view(system, payload, packet, envelope) -> tuple:
    """Copy a selected request, replacing only authenticated card candidates."""
    if not selected_source_view(envelope):
        return system, payload
    from litagent.safety.planned_budget import validate_writer_envelope

    validate_writer_envelope(envelope)
    if type(system) is not str:
        _invalid()
    lookup = _source_lookup(packet)
    stage = _payload_stage(payload)
    if "grounding_instruction" in payload and (
        stage != "repair" or payload["grounding_instruction"] != GROUNDING_INSTRUCTION
    ):
        _invalid()
    for key in set(packet) - {"cards", "sources"}:
        if key in payload and payload[key] != packet[key]:
            _invalid()
    if stage == "summary":
        validate_summary_body_view(payload, envelope)
        # Summary premises stay body-only; a projection never grants new sources.
        if "sources" in payload or "cards" in payload:
            _invalid()
    elif "sources" in payload or "cards" in payload:
        sources, cards = payload.get("sources"), payload.get("cards")
        if (
            type(sources) is not dict
            or sources != packet["sources"]
            or list(sources) != list(packet["sources"])
            or type(cards) is not dict
            or list(cards) != list(lookup)
            or (cards != packet["cards"] and cards != lookup)
        ):
            _invalid()
    elif stage != "repair" or "body_units" not in payload:
        _invalid()
    result = copy.deepcopy(payload)
    if "cards" in result:
        result["cards"] = lookup
    if stage == "repair":
        # The concise repair system has an existing exact identity. Keep that
        # boundary intact; this authenticated input rule grants no paper support.
        result["grounding_instruction"] = GROUNDING_INSTRUCTION
        return system, result
    if system.count(GROUNDING_INSTRUCTION) > 1:
        _invalid()
    if GROUNDING_INSTRUCTION not in system:
        system += "\n" + GROUNDING_INSTRUCTION
    return system, result


def make_source_view_record(envelope, system, payload, packet) -> dict:
    """Record local final-input hashes without adding model-facing metadata."""
    if not selected_source_view(envelope):
        return {}
    projected_system, projected = project_source_view(system, payload, packet, envelope)
    if projected_system != system or projected != payload:
        _invalid()
    return {
        "contract": CONTRACT,
        "identity": source_view_identity(),
        "source_snapshot": _digest(packet),
        "view_sha": _digest(payload),
        "system_sha": _digest(system),
    }


def validate_source_view_record(envelope, record, packet) -> None:
    """Authenticate local lineage, without claiming unavailable wire replay."""
    if not selected_source_view(envelope):
        if record is not None and record != {}:
            _invalid()
        return
    _source_lookup(packet)
    if (
        type(record) is not dict
        or set(record)
        != {"contract", "identity", "source_snapshot", "view_sha", "system_sha"}
        or record["contract"] != CONTRACT
        or record["identity"] != source_view_identity()
        or record["source_snapshot"] != _digest(packet)
        or not _hex_digest(record["view_sha"])
        or not _hex_digest(record["system_sha"])
    ):
        _invalid()
