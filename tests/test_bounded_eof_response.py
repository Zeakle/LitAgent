"""Replay bounded native EOF responses without altering scientific content."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_responses import envelope as response_envelope
from test_native_responses import owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.writing_v2 import (
    BOUNDED_EOF_FOOTPRINT_CONTRACT,
    EOF_CONTAINER_POLICY,
    WriterError,
    WriterSession,
    body_section_schema,
    intact_unit_footprint,
    normalize_eof_response,
    parse_response,
    provider_response_format,
    summary_request_input,
    writer_budget_plan,
)


@pytest.fixture
def live_eof():
    """Load the unchanged public-paper-derived taxonomy and its source digest."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/public-native-eof-021.json").read_text()
    )
    assert hashlib.sha256(fixture["raw"].encode()).hexdigest() == fixture["raw_sha256"]
    assert len(fixture["raw"]) == 642
    assert fixture["scientific_assessment"] == "not_verified"
    return fixture


def test_actual_021_eof_keeps_text_references_positions_and_full_schema(live_eof):
    """Only the unique EOF container suffix may differ from the received body."""
    schema = body_section_schema(
        live_eof["section"], live_eof["references"], maximum=live_eof["maximum"]
    )
    raw = live_eof["raw"]
    with pytest.raises(WriterError, match="writer_json_invalid"):
        parse_response(raw, schema)
    expected = json.loads(raw + live_eof["suffix"])
    Draft202012Validator(schema).validate(expected)
    normalized, proof = normalize_eof_response(raw, schema)
    assert normalized == expected
    assert proof["suffix"] == live_eof["suffix"]
    assert proof["raw_sha256"] == live_eof["raw_sha256"]
    assert proof["scientific_assessment"] == "not_verified"
    assert proof["units"] == intact_unit_footprint(raw, schema)["units"]
    assert (
        proof["units"] == intact_unit_footprint(raw + proof["suffix"], schema)["units"]
    )
    assert len(normalized["taxonomy"]) == live_eof["paragraphs"]
    units = [u for p in normalized["taxonomy"] for u in p["sentences"]]
    assert len(units) == len(proof["units"]) == live_eof["units"]
    for unit, position in zip(units, proof["units"]):
        assert json.loads(raw[position["start"] : position["end"]]) == unit


@pytest.mark.asyncio
async def test_actual_sdk_bounded_eof_preserves_raw_receipt_and_known_usage(
    case, live_eof, tmp_path
):
    """Replace HTTP only; the real SDK and session own parsing and cost settlement."""
    schema = body_section_schema(
        live_eof["section"], live_eof["references"], maximum=live_eof["maximum"]
    )
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        assert body["thinking"] == {"type": "disabled"}
        assert (
            body["text"]["format"]["schema"]
            == provider_response_format("synthesis", schema)["json_schema"]["schema"]
        )
        return httpx.Response(
            200,
            json=response_envelope(
                live_eof["raw"],
                usage={
                    "input_tokens": 10702,
                    "output_tokens": 346,
                    "total_tokens": 11048,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            ),
        )

    owner, budget, _ = await owner_for(handler)
    session = WriterSession(
        owner,
        tmp_path / "bounded-eof.json",
        "bounded-eof-offline",
        case.packet,
        case.aliases,
        provider_api="responses",
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    try:
        value = await session.call("body:taxonomy", "Write sources only.", {}, schema)
        assert value == json.loads(live_eof["raw"] + live_eof["suffix"])
        assert len(sent) == 1 and budget.used == 11048
        entry = session.journal["calls"]["body:taxonomy"]
        assert entry["raw"] == live_eof["raw"]
        assert entry["normalized_raw"] == live_eof["raw"] + live_eof["suffix"]
        assert entry["provider_receipt"]
        assert entry["usage"] == live_eof["usage"]
        assert entry["syntax_normalization"]["scientific_assessment"] == "not_verified"
        assert session.state["attempts"] == []
        with pytest.raises(WriterError, match="no_replay"):
            await session.call("body:taxonomy", "Write sources only.", {}, schema)
        assert len(sent) == 1 and budget.used == 11048
    finally:
        await owner.close()


@pytest.mark.parametrize("stage", ["body", "summary"])
def test_bounded_eof_accepts_the_same_unit_schema_in_every_slot(case, stage):
    """Body and derived summaries retain their own reference fields and positions."""
    if stage == "body":
        schema = body_section_schema("methods", case.aliases, maximum=7)
        value = {"methods": copy.deepcopy(case.body["methods"])}
    else:
        from litagent.safety.planned_budget import make_writer_envelope

        scope = make_writer_envelope(
            body_units=16,
            assertions=32,
            body_bytes=16000,
            source_identity="a" * 64,
            builder_identity="b" * 64,
        )
        _, _, schema = summary_request_input(case.packet, {"u1": {}}, scope)
        value = {
            key: [
                {"sentences": [{"text": key + " finding.", "body_unit_refs": ["u1"]}]}
            ]
            for key in ("abstract", "conclusion")
        }
    raw = json.dumps(value, ensure_ascii=False)
    parsed, proof = normalize_eof_response(raw[:-2], schema)
    assert parsed == value and proof["suffix"] == "]}"
    assert proof["units"] == intact_unit_footprint(raw, schema)["units"]
    assert proof["scientific_assessment"] == "not_verified"


@pytest.mark.parametrize(
    "fault",
    [
        "different_ref_enum",
        "different_text_limit",
        "different_ref_field",
        "open_paragraph",
        "open_unit",
        "missing_unit_required",
        "empty_outer_prefix",
        "empty_inner_prefix",
        "outer_items_true",
        "inner_items_true",
        "outer_mixed_items",
        "inner_mixed_items",
        "outer_items_without_prefix",
        "inner_items_without_prefix",
        "unknown_schema_ref",
    ],
)
def test_every_bounded_schema_slot_must_be_closed_and_compatible(live_eof, fault):
    """Unused slots cannot silently grant a different schema or fallback authority."""
    schema = body_section_schema("taxonomy", live_eof["references"], maximum=9)
    root = schema["properties"]["taxonomy"]
    paragraph = root["prefixItems"][-1]
    sentences = paragraph["properties"]["sentences"]
    unit = sentences["prefixItems"][-1]
    if fault == "different_ref_enum":
        unit["properties"]["fact_card_refs"]["items"]["enum"] = ["outside"]
    elif fault == "different_text_limit":
        unit["properties"]["text"]["maxLength"] = 1999
    elif fault == "different_ref_field":
        unit["properties"]["body_unit_refs"] = unit["properties"].pop("fact_card_refs")
        unit["required"] = ["text", "body_unit_refs"]
    elif fault == "open_paragraph":
        paragraph["additionalProperties"] = True
    elif fault == "open_unit":
        unit["additionalProperties"] = True
    elif fault == "missing_unit_required":
        unit["required"] = ["text"]
    elif fault == "empty_outer_prefix":
        root["prefixItems"] = []
    elif fault == "empty_inner_prefix":
        sentences["prefixItems"] = []
    elif fault == "outer_items_true":
        root["items"] = True
    elif fault == "inner_items_true":
        sentences["items"] = True
    elif fault == "outer_mixed_items":
        root["items"] = copy.deepcopy(paragraph)
    elif fault == "inner_mixed_items":
        sentences["items"] = copy.deepcopy(unit)
    elif fault == "outer_items_without_prefix":
        del root["prefixItems"]
    elif fault == "inner_items_without_prefix":
        del sentences["prefixItems"]
    elif fault == "unknown_schema_ref":
        sentences["prefixItems"][-1] = {"$ref": "#/$defs/unknown"}
    with pytest.raises(WriterError, match="writer_reemission_schema_ineligible"):
        intact_unit_footprint(live_eof["raw"], schema)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_reference",
        "missing_comma",
        "unfinished_string",
        "unfinished_unit",
        "duplicate_key",
        "unknown_field",
        "unassigned_container",
        "wrong_closer",
        "empty_section",
        "section_capacity",
        "sentence_capacity",
    ],
)
def test_eof_suffix_cannot_fix_content_or_capacity(live_eof, fault):
    """The complete host schema and unchanged scanner still reject unsafe prefixes."""
    schema = body_section_schema("taxonomy", live_eof["references"], maximum=9)
    value = json.loads(live_eof["raw"] + live_eof["suffix"])
    raw = live_eof["raw"]
    if fault == "unknown_reference":
        raw = raw.replace('"c10"', '"outside"', 1)
    elif fault == "missing_comma":
        raw = raw.replace('},{"text":', '}{"text":', 1)
    elif fault == "unfinished_string":
        raw = raw[: raw.index("子空间。") + 2]
    elif fault == "unfinished_unit":
        raw = raw[:-3]
    elif fault == "duplicate_key":
        raw = raw.replace('"text":', '"text":"Duplicate.","text":', 1)
    elif fault == "unknown_field":
        raw = raw.replace('"text":', '"unassigned":"Extra.","text":', 1)
    elif fault == "unassigned_container":
        raw = "[" + raw
    elif fault == "wrong_closer":
        raw += "}"
    elif fault == "empty_section":
        raw = '{"taxonomy":[]'
    elif fault == "section_capacity":
        value["taxonomy"] = value["taxonomy"] * 2
        raw = json.dumps(value, ensure_ascii=False)[:-1]
    elif fault == "sentence_capacity":
        value["taxonomy"][0]["sentences"] *= 2
        raw = json.dumps(value, ensure_ascii=False)[:-1]
    with pytest.raises(WriterError):
        normalize_eof_response(raw, schema)


def test_legacy_items_footprint_and_eof_proof_remain_identical(live_eof):
    """Adding tuple support does not reinterpret the old uniform array contract."""
    legacy = body_section_schema("taxonomy", live_eof["references"])
    bounded = body_section_schema("taxonomy", live_eof["references"], maximum=9)
    old = intact_unit_footprint(live_eof["raw"], legacy)
    new = intact_unit_footprint(live_eof["raw"], bounded)
    assert old["semantic"] == new["semantic"] and old["units"] == new["units"]
    assert old["raw_sha256"] == new["raw_sha256"] == live_eof["raw_sha256"]
    normalized, proof = normalize_eof_response(live_eof["raw"], legacy)
    assert normalized == json.loads(live_eof["raw"] + "]}")
    assert proof["units"] == old["units"]


@pytest.mark.parametrize("maximum,sizes", [(5, [3, 2]), (7, [3, 2, 2])])
def test_unequal_paragraph_capacities_are_checked_at_their_actual_slot(
    live_eof, maximum, sizes
):
    """A shorter later slot cannot borrow the unused capacity of its predecessor."""
    schema = body_section_schema("taxonomy", live_eof["references"], maximum=maximum)
    value = json.loads(live_eof["raw"] + live_eof["suffix"])
    template = value["taxonomy"][0]["sentences"][0]
    value["taxonomy"] = [
        {
            "sentences": [
                {
                    **copy.deepcopy(template),
                    "text": f"Slot {index}, condition {offset}.",
                }
                for offset in range(size)
            ]
        }
        for index, size in enumerate(sizes)
    ]
    raw = json.dumps(value, ensure_ascii=False)
    parsed, proof = normalize_eof_response(raw[:-2], schema)
    assert parsed == value and len(proof["units"]) == maximum
    value["taxonomy"][-1]["sentences"].append(
        copy.deepcopy(value["taxonomy"][-1]["sentences"][0])
    )
    too_many = json.dumps(value, ensure_ascii=False)
    with pytest.raises(WriterError, match="paragraph_coverage_invalid"):
        intact_unit_footprint(too_many[:-2], schema)
    with pytest.raises(WriterError, match="writer_native_capacity_exceeded"):
        normalize_eof_response(too_many[:-2], schema)


@pytest.mark.asyncio
async def test_scientific_failure_remains_blocked_after_bounded_eof_compilation(case):
    """Syntax recovery grants neither factual support nor automatic acceptance."""
    from test_writer_v2 import EvaluatorTransport

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest, strict_acceptance
    from litagent.writing_v2 import compile_document

    body = copy.deepcopy(case.body)
    schema = body_section_schema("methods", case.aliases, maximum=3)
    raw = json.dumps({"methods": body["methods"]}, ensure_ascii=False)
    part, proof = normalize_eof_response(raw[:-2], schema)
    body.update(part)
    draft, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, "en"
    )
    context = {
        "query": case.packet["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "report_language": "en",
        "review_structure": "academic_v2",
        "writing_bindings": bindings,
        "scope_notice_contract": "scope-notice-v1",
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence": case.ledger,
        "coverage_requirements": case.packet["requirements"],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
    }
    details = await evaluate_scoped(
        llm=EvaluatorTransport(unsupported=True),
        survey=draft,
        context=context,
        config=EvalConfig(),
    )
    assert proof["scientific_assessment"] == "not_verified"
    assert details["coverage"]["complete"] and details["unsupported_count"] > 0
    assert strict_acceptance(details)["status"] == "failed"


def test_only_v4_budget_builder_identity_binds_bounded_footprint(case, monkeypatch):
    """A footprint change invalidates v4 plans while v2 and v3 remain stable."""
    import litagent.writing_v2 as writer

    original = {
        version: writer_budget_plan(
            case.packet,
            case.aliases,
            budget_policy=f"planned-budget-v{version}",
            methodology={},
        )
        for version in (2, 3, 4)
    }
    monkeypatch.setattr(writer, "BOUNDED_EOF_FOOTPRINT_CONTRACT", "future-offline-test")
    changed = {
        version: writer_budget_plan(
            case.packet,
            case.aliases,
            budget_policy=f"planned-budget-v{version}",
            methodology={},
        )
        for version in (2, 3, 4)
    }
    assert original[2] == changed[2] and original[3] == changed[3]
    assert original[4]["identity"] != changed[4]["identity"]
    assert (
        original[4]["envelope"]["builder_identity"]
        != changed[4]["envelope"]["builder_identity"]
    )


def test_run_options_binds_bounded_footprint_only_for_v4(monkeypatch):
    """The public run identity invalidates v4 without changing historical defaults."""
    import litagent.writing_v2 as writer
    from litagent.workbench import RunOptions, VerificationOptions

    common = {
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "generation_strategy": "body_summary_v2",
        "verification": VerificationOptions(judge=False, jev=False),
    }
    options = {"legacy": RunOptions()}
    options.update(
        {
            str(version): RunOptions(
                **common, budget_policy=f"planned-budget-v{version}"
            )
            for version in (2, 3, 4)
        }
    )
    original = {name: value.fingerprint() for name, value in options.items()}
    monkeypatch.setattr(writer, "BOUNDED_EOF_FOOTPRINT_CONTRACT", "future-offline-test")
    changed = {name: value.fingerprint() for name, value in options.items()}
    for name in ("legacy", "2", "3"):
        assert original[name] == changed[name]
    assert original["4"] != changed["4"]


@pytest.mark.parametrize("version", [2, 3, 4])
def test_session_footprint_identity_changes_only_for_v4(
    case, tmp_path, monkeypatch, version
):
    """Only the explicit v4 owner appends the new bounded recovery contract."""
    from types import SimpleNamespace

    import litagent.writing_v2 as writer
    from litagent.safety.planned_budget import (
        PlannedBudget,
        adaptive_writer_plan,
        build_budget_plan,
    )

    if version == 2:
        from test_planned_budget import stages

        plan = build_budget_plan(stages(), identity="controlled-legacy-identity")
    elif version == 3:
        from test_adaptive_budget import certified_bounds

        plan = adaptive_writer_plan(
            identity="controlled-adaptive-identity", stage_bounds=certified_bounds()
        )
    else:
        plan = writer_budget_plan(
            case.packet, case.aliases, budget_policy="planned-budget-v4", methodology={}
        )
    owner = SimpleNamespace(_cost_budget=PlannedBudget(plan))
    first = WriterSession(
        owner, tmp_path / "first", "identity", case.packet, case.aliases
    )
    first.save()
    saved = first.journal_path.read_bytes()
    if version == 4:
        assert (
            first.journal["bounded_eof_footprint_contract"]
            == BOUNDED_EOF_FOOTPRINT_CONTRACT
        )
    else:
        assert "bounded_eof_footprint_contract" not in first.journal
    monkeypatch.setattr(writer, "BOUNDED_EOF_FOOTPRINT_CONTRACT", "future-offline-test")
    second = WriterSession(
        owner, tmp_path / "second", "identity", case.packet, case.aliases
    )
    assert (first.identity != second.identity) is (version == 4)
    if version == 4:
        with pytest.raises(WriterError, match="writer_checkpoint_identity_conflict"):
            WriterSession(
                owner, tmp_path / "first", "identity", case.packet, case.aliases
            )
    else:
        restored = WriterSession(
            owner, tmp_path / "first", "identity", case.packet, case.aliases
        )
        assert restored.identity == first.identity
    assert first.journal_path.read_bytes() == saved
