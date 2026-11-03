"""Check the flat transport without granting it a scientific verdict."""

import copy
import hashlib
import json

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.writing_flat import CONTRACT, canonical_schema, encode, parse, prepare
from litagent.writing_v2 import (
    BODY,
    WriterError,
    body_section_schema,
    body_units,
    compile_document,
    parse_response,
    schema_for,
    writer_budget_plan,
)


@pytest.mark.parametrize("section", BODY)
def test_flat_preserves_whole_chapter_values_and_raw_positions(case, section):
    schema = body_section_schema(section, case.aliases, maximum=8)
    value = {section: copy.deepcopy(case.body[section])}
    if section == BODY[0]:
        value["title"] = case.body["title"]
    wire = encode(value, schema)
    # JSON object order never defines logical unit order.
    raw = json.dumps(
        {k: v for k, v in reversed(list(wire.items()))}, ensure_ascii=False
    )
    result = parse(raw, schema)
    assert result["canonical"] == value
    for record in result["bindings"]["units"]:
        text = json.loads(
            raw[record["text_span"]["start"] : record["text_span"]["end"]]
        )
        key, pi, _, si = record["canonical_path"]
        assert value[key][pi]["sentences"][si]["text"] == text
    assert result["bindings"]["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert result["bindings"]["scientific_assessment"] == "not_verified"


def test_flat_allows_transition_but_does_not_change_old_schema(case):
    schema = body_section_schema("methods", case.aliases, maximum=8)
    value = {
        "methods": [
            {"sentences": [{"text": "下面分析方法的适用条件。", "fact_card_refs": []}]}
        ]
    }
    result = parse(json.dumps(encode(value, schema)), schema)
    assert result["canonical"] == value
    with pytest.raises(WriterError):
        parse_response(json.dumps(value), schema)


def test_flat_paragraphs_are_not_fixed_slots(case):
    schema = body_section_schema("methods", case.aliases, maximum=8)
    unit = copy.deepcopy(case.body["methods"][0]["sentences"][0])
    value = {"methods": [{"sentences": [unit] * 5}]}
    assert parse(json.dumps(encode(value, schema)), schema)["canonical"] == value
    assert prepare(schema)["wire_schema"]["properties"]["units"]["maxItems"] == 8


@pytest.mark.parametrize("count", [31, 62])
def test_flat_full_compiler_uses_whole_capacity_without_old_paragraph_limits(
    case, count
):
    from litagent.writing_v2 import (
        RANGE_WRITING_PLAN,
        capacity_repair_targets,
        validate_body,
    )

    scope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )["envelope"]
    body = copy.deepcopy(case.body)
    unit = copy.deepcopy(body["methods"][0]["sentences"][0])
    body["methods"] = [{"sentences": [copy.deepcopy(unit) for _ in range(count)]}]
    schema = body_section_schema("methods", case.aliases, maximum=62)
    decoded = parse(json.dumps(encode({"methods": body["methods"]}, schema)), schema)
    assert decoded["canonical"]["methods"] == body["methods"]
    assert (
        len(
            validate_body(
                body, case.aliases, writing_plan=RANGE_WRITING_PLAN, envelope=scope
            )
        )
        == count + 4
    )
    assert capacity_repair_targets(body, RANGE_WRITING_PLAN, envelope=scope) == {}
    report, bindings = compile_document(
        body,
        case.summaries,
        case.cards,
        case.aliases,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=scope,
    )
    assert len(bindings["units"]) == count + 6
    assert all(
        report[u["source_start"] : u["source_end"]] == u["source_quote"]
        for u in bindings["units"]
    )


def test_flat_whole_capacity_overflow_is_not_silently_repaired(case):
    from litagent.safety.planned_budget import EnvelopeExceeded
    from litagent.writing_v2 import RANGE_WRITING_PLAN, validate_body

    scope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )["envelope"]
    body = copy.deepcopy(case.body)
    unit = body["methods"][0]["sentences"][0]
    body["methods"] = [{"sentences": [copy.deepcopy(unit) for _ in range(63)]}]
    frozen = copy.deepcopy(body)
    with pytest.raises(EnvelopeExceeded):
        validate_body(
            body, case.aliases, writing_plan=RANGE_WRITING_PLAN, envelope=scope
        )
    assert body == frozen


@pytest.mark.parametrize(
    "fault", ["empty", "bad_bool", "unknown", "extra", "duplicate"]
)
def test_flat_rejects_contract_errors(case, fault):
    schema = body_section_schema("methods", case.aliases, maximum=8)
    wire = encode({"methods": case.body["methods"]}, schema)
    if fault == "empty":
        wire["units"] = []
    elif fault == "bad_bool":
        wire["units"][0]["paragraph_start"] = "true"
    elif fault == "unknown":
        wire["units"][0]["fact_card_refs"] = ["not-a-card"]
    elif fault == "extra":
        wire["units"][0]["kind"] = "organization"
    raw = json.dumps(wire)
    if fault == "duplicate":
        raw = raw.replace('"text":', '"text":"duplicate", "text":', 1)
    with pytest.raises(WriterError):
        parse(raw, schema)


def test_flat_eof_requires_intact_original_units(case):
    schema = body_section_schema("methods", case.aliases, maximum=8)
    wire = encode({"methods": case.body["methods"]}, schema)
    raw = json.dumps(wire)
    result = parse(raw[:-2], schema, allow_eof=True)
    assert result["normalization"]["suffix"] == "]}"
    assert result["canonical"] == {"methods": case.body["methods"]}
    with pytest.raises(WriterError):
        parse(raw[:-3], schema, allow_eof=True)
    with pytest.raises(WriterError):
        parse(
            raw.replace("}, {", ", {", 1) if "}, {" in raw else raw[:-3] + "]}",
            schema,
            allow_eof=True,
        )


def test_flat_summaries_and_patches_share_one_transport(case):
    units = body_units(case.body)
    schema = schema_for("summary", units)
    assert (
        parse(json.dumps(encode(case.summaries, schema)), schema)["canonical"]
        == case.summaries
    )
    schema = schema_for("repair", case.aliases, ["target"])
    value = {
        "outcome": "patched",
        "explanation": "Replace the authorized finding.",
        "patches": [
            {
                "target_id": "target",
                "sentences": [case.body["methods"][0]["sentences"][0]],
            }
        ],
    }
    wire = encode(value, schema)
    assert "sentences" not in wire["patches"][0]
    assert parse(json.dumps(wire), schema)["canonical"] == value


def test_flat_v5_keeps_full_scope_and_two_repairs(case):
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    assert plan["status"] == "admitted"
    assert plan["maximum_tokens"] == 4500000
    assert plan["envelope"]["body_max_units"] == 66
    assert plan["envelope"]["model_assertions"] == 128
    assert plan["repair_limit"] == 2
    assert plan["planned_requests"] == 41
    assert plan["scope_bounds"]["envelope_contract"] == plan["envelope"]["contract"]
    assert plan["repair_selection"] == "fixed-two-complete-repairs-v1"
    from litagent.writing_v2 import RANGE_WRITING_PLAN, envelope_writing_spec

    spec = envelope_writing_spec(RANGE_WRITING_PLAN, plan["envelope"])
    assert sum(spec["body_target_units"].values()) == 48
    assert spec["body_max_units"] == 66
    assert spec["summary_section_max_units"] == 2
    assert set(spec["body_section_max_units"].values()) == {62}
    body = copy.deepcopy(case.body)
    body["introduction"][0]["sentences"].append(
        {"text": "下面讨论方法。", "fact_card_refs": []}
    )
    report, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, envelope=plan["envelope"]
    )
    assert any(not u["evidence_ids"] for u in bindings["units"])
    assert "下面讨论方法。" in report


def test_flat_strategy_pair_and_legacy_identity(case):
    from litagent.workbench import RunOptions, VerificationOptions

    legacy = RunOptions().fingerprint()
    args = dict(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        verification=VerificationOptions(judge=False, jev=False),
    )
    new = RunOptions(
        **args, generation_strategy="body_summary_v3", budget_policy="planned-budget-v5"
    )
    assert new.fingerprint() != legacy
    assert RunOptions().fingerprint() == legacy
    with pytest.raises(ValueError):
        RunOptions(
            **args,
            generation_strategy="body_summary_v3",
            budget_policy="planned-budget-v4",
        )
    with pytest.raises(ValueError):
        RunOptions(
            **args,
            generation_strategy="body_summary_v2",
            budget_policy="planned-budget-v5",
        )


def test_flat_summary_input_scope_is_bound_to_run_and_budget_identity(
    case, monkeypatch
):
    import litagent.writing_flat as flat
    from litagent.workbench import RunOptions, VerificationOptions

    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v5",
        verification=VerificationOptions(judge=False, jev=False),
    )
    original = options.fingerprint()
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    legacy = RunOptions().fingerprint()
    monkeypatch.setattr(flat, "REQUEST_BUILDERS", "future-summary-input-scope")
    assert options.fingerprint() != original
    changed = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    assert changed["identity"] != plan["identity"]
    assert RunOptions().fingerprint() == legacy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy,policy",
    [("body_summary_v3", "legacy"), ("body_summary_v2", "planned-budget-v5")],
)
async def test_flat_missing_budget_tag_cannot_silently_fall_back(strategy, policy):
    from types import SimpleNamespace

    from litagent.writing_v2 import synthesize

    task = SimpleNamespace(
        input_data={"generation_strategy": strategy, "budget_policy": policy}
    )
    with pytest.raises(WriterError, match="strategy_budget_pair_invalid"):
        await synthesize(None, task, None, None, None, None)


@pytest.mark.parametrize("schema", [None, [], {}, {"type": "object"}])
def test_flat_invalid_host_contract_has_a_stable_error(schema):
    with pytest.raises(WriterError, match="flat_contract_invalid"):
        prepare(schema)


def test_flat_review_projection_is_reversible_and_keeps_original_bindings(case):
    from litagent.writing_flat import review_projection

    payload = {
        "cards": case.packet["cards"],
        "units": {
            "u": {
                "source_quote": "A claim [E:b]",
                "fact_card_refs": [case.by_eid["b"]],
                "source_start": 7,
                "source_end": 24,
            }
        },
        "report_sha256": "frozen-report",
    }
    before = copy.deepcopy(payload)
    view = review_projection(payload)
    assert payload == before
    assert view["report_sha256"] == payload["report_sha256"]
    assert "source_start" not in view["units"]["u"]
    alias = case.by_eid["b"]
    assert view["units"]["u"]["source_quote"] == f"A claim [E:{alias}]"
    assert view["citation_projection"][alias] == "b"
