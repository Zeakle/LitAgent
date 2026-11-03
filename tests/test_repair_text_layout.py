"""Keep invalid repair prose rejected with precise, content-free diagnostics."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import test_guidance_wire_identity as wire_identity
import test_source_map_pipeline as source_pipeline
import test_structured_repair_pipeline as structured
from jsonschema import Draft202012Validator
from test_guidance_wire_contract import guidance_case  # noqa: F401
from test_repair_task_view_pipeline import select_policy
from test_structured_repair_identity import budget_plan, guidance_envelope
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
from litagent.capacity_repair import (
    capacity_targets,
    normalize_source_map,
    original_unit_directory,
    source_edit_schema,
)
from litagent.safety.planned_budget import PlannedBudget
from litagent.writing_v2 import (
    WriterError,
    body_section_schema,
    request_messages,
    schema_for,
)

FIXTURE = Path(__file__).parent / "fixtures/repair-048-layout.json"
OLD_TASK_IDENTITY = "4ec2c78604993b5daac5eca64ad34ebca9a90af44833c3b9cee5b3b052007a68"
OLD_UNIT_IDENTITY = "52e322d92f903aa02e2d730c9c1cf3db66c52d0217cd89a2293a5efa570b2b85"


@pytest.fixture
def layout_case():
    """Use neutral prose with the original ordered 24-patch/14-LF geometry."""
    fixture = json.loads(FIXTURE.read_text())
    targets = capacity_targets(fixture["body"], {}, maximum=66)
    references = {ref: {} for ref in fixture["reference_ids"]}
    schema = source_edit_schema(schema_for("repair", references, targets), targets)
    return fixture, targets, schema


def parse_layout(value, schema, **kwargs):
    """Exercise real tokenization, schemas, references and raw bindings."""
    return flat.parse(
        json.dumps(value),
        schema,
        reference_contract=flat.REFERENCE_SET_CONTRACT,
        transport_revision=flat.TRANSPORT_REVISION,
        **kwargs,
    )


def independent_legal_reply(fixture):
    """Create a separately labeled positive; never sanitize an accepted response."""
    value = copy.deepcopy(fixture["raw_response"])
    value["explanation"] = "Independent legal structural control, not scientific gold."
    for index, patch in enumerate(value["patches"]):
        patch["text"] = f"Independent replacement number {index}."
    return value


def expected_location(value, index, stage="repair"):
    """Derive physical and target-local positions without searching repeated text."""
    patch = value["patches"][index]
    target_order = list(dict.fromkeys(p["target_id"] for p in value["patches"]))
    local_index = sum(
        p["target_id"] == patch["target_id"] for p in value["patches"][:index]
    )
    return {
        "code": "writer_text_invalid",
        "stage": stage,
        "wire_path": ["patches", index, "text"],
        "canonical_path": [
            "patches",
            target_order.index(patch["target_id"]),
            "sentences",
            local_index,
            "text",
        ],
        "target_id": patch["target_id"],
        "characters": len(patch["text"]),
        "text_sha256": hashlib.sha256(
            patch["text"].encode("utf-8", errors="surrogatepass")
        ).hexdigest(),
        "line_feed_count": patch["text"].count("\n"),
    }


def test_neutral048_geometry_remains_rejected_without_mutating_the_raw(layout_case):
    fixture, _, schema = layout_case
    before = FIXTURE.read_bytes()
    value = fixture["raw_response"]
    assert fixture["science_gold_standard"] is False
    assert fixture["contains_original_scientific_text"] is False
    assert fixture["provenance"]["projection_is_original_raw_response"] is False
    assert len(value["patches"]) == fixture["patch_count"] == 24
    assert [i for i, p in enumerate(value["patches"]) if "\n" in p["text"]] == fixture[
        "line_feed_patch_indexes"
    ]
    assert len(fixture["line_feed_patch_indexes"]) == 14
    Draft202012Validator(flat.prepare(schema)["wire_schema"]).validate(value)
    for kwargs in ({}, {"locate_text_errors": False}):
        with pytest.raises(WriterError, match="writer_text_invalid") as failure:
            parse_layout(value, schema, **kwargs)
        assert failure.value.family == "protocol" and failure.value.diagnostics == []
    with pytest.raises(WriterError, match="writer_text_invalid") as failure:
        parse_layout(value, schema, locate_text_errors=True)
    assert failure.value.diagnostics == [expected_location(value, 0)]
    assert FIXTURE.read_bytes() == before


@pytest.mark.parametrize("index", range(14))
def test_each_lf_position_is_located_independently(layout_case, index):
    fixture, _, schema = layout_case
    selected = fixture["line_feed_patch_indexes"][index]
    value = independent_legal_reply(fixture)
    value["patches"][selected]["text"] = fixture["raw_response"]["patches"][selected][
        "text"
    ]
    original = copy.deepcopy(value)
    with pytest.raises(WriterError, match="writer_text_invalid") as failure:
        parse_layout(
            value, schema, locate_text_errors=True, diagnostic_stage="repair_response"
        )
    assert failure.value.diagnostics == [
        expected_location(value, selected, "repair_response")
    ]
    assert value == original


@pytest.mark.parametrize("bad", [None, 0, 1, "true", [], {}])
def test_location_flag_is_a_strict_local_bool(layout_case, bad):
    _, _, schema = layout_case
    with pytest.raises(WriterError, match="writer_flat_text_diagnostic_flag_invalid"):
        flat.parse("not even JSON", schema, locate_text_errors=bad)


@pytest.mark.parametrize(
    "text",
    [
        "First.\nSecond.",
        "First.\r\nSecond.",
        "# Heading",
        "<b>HTML</b>",
        "A claim [E:source]",
        "A claim [F:card]",
        "   ",
    ],
)
def test_existing_text_guard_stays_authoritative(layout_case, text):
    fixture, _, schema = layout_case
    value = independent_legal_reply(fixture)
    value["patches"][0]["text"] = text
    with pytest.raises(WriterError, match="writer_text_invalid") as failure:
        parse_layout(value, schema, locate_text_errors=True)
    assert failure.value.diagnostics == [expected_location(value, 0)]
    assert "text" not in failure.value.diagnostics[0]


@pytest.mark.parametrize("locate", [False, True])
def test_invalid_unicode_keeps_the_earlier_raw_guard(layout_case, locate):
    """LF diagnostics must not override a stricter raw-token Unicode refusal."""
    fixture, _, schema = layout_case
    value = independent_legal_reply(fixture)
    value["patches"][0]["text"] = "\ud800\n"
    with pytest.raises(WriterError, match="writer_flat_raw_unicode_invalid"):
        parse_layout(value, schema, locate_text_errors=locate)


@pytest.mark.parametrize("text", ["First.\rSecond.", "A valid single-line statement."])
def test_no_new_cr_normalization_or_rejection_is_invented(layout_case, text):
    fixture, _, schema = layout_case
    value = independent_legal_reply(fixture)
    value["patches"][0]["text"] = text
    parsed = parse_layout(value, schema, locate_text_errors=True)
    assert parsed["canonical"]["patches"][0]["sentences"][0]["text"] == text


@pytest.mark.parametrize("fault", ["empty", "wrong_type", "unknown_field"])
def test_schema_errors_are_not_reclassified_as_text_errors(layout_case, fault):
    fixture, _, schema = layout_case
    value = independent_legal_reply(fixture)
    if fault == "unknown_field":
        value["patches"][0]["invented_field"] = True
    else:
        value["patches"][0]["text"] = "" if fault == "empty" else ["not prose"]
    with pytest.raises(WriterError, match="writer_flat_schema_invalid") as failure:
        parse_layout(value, schema, locate_text_errors=True)
    assert all(
        d.get("code") != "writer_text_invalid" for d in failure.value.diagnostics
    )


def test_multiple_flat_patches_share_one_target_and_actual_split_sources(layout_case):
    fixture, targets, schema = layout_case
    tid = next(iter(targets))
    ids = original_unit_directory(targets)[tid]
    value = {
        "outcome": "patched",
        "explanation": "Separate legal one-to-many source declaration.",
        "patches": [
            {
                "target_id": tid,
                "text": text,
                "fact_card_refs": fixture["raw_response"]["patches"][0][
                    "fact_card_refs"
                ],
                "source_unit_ids": list(ids),
            }
            for text in ("First independent unit.", "Second independent unit.")
        ],
        "deleted_sources": [],
    }
    parsed = parse_layout(value, schema, locate_text_errors=True)
    assert len(parsed["canonical"]["patches"]) == 1
    assert len(parsed["canonical"]["patches"][0]["sentences"]) == 2
    assert [
        r["replacement_index"] for r in parsed["canonical"]["source_relations"]
    ] == [0, 1]
    assert all(
        r["source_unit_ids"] == ids for r in parsed["canonical"]["source_relations"]
    )
    normalized = normalize_source_map(
        parsed["canonical"], targets, max_json_bytes=12288
    )
    assert all(
        r["replacement_indexes"] == [0, 1]
        for r in normalized["edit_records"]
        if r["unit_id"] in ids
    )


def test_interleaved_target_and_repeated_text_use_physical_not_search_position(
    layout_case,
):
    fixture, _, schema = layout_case
    value = independent_legal_reply(fixture)
    first, second = value["patches"][:2]
    assert first["target_id"] != second["target_id"]
    third = copy.deepcopy(first)
    first["text"] = "Repeated neutral text."
    third["text"] = "Repeated neutral text.\nIllegal next line."
    value["patches"] = [first, second, third]
    with pytest.raises(WriterError, match="writer_text_invalid") as failure:
        parse_layout(value, schema, locate_text_errors=True)
    assert failure.value.diagnostics == [expected_location(value, 2)]


@pytest.mark.parametrize("kind", ["chapter", "summary"])
def test_nonrepair_unit_location_has_no_invented_target(kind):
    if kind == "chapter":
        schema = body_section_schema("methods", {"c1": {}})
        value = {
            "units": [
                {
                    "text": "First.\nSecond.",
                    "fact_card_refs": ["c1"],
                    "paragraph_start": True,
                }
            ]
        }
        path = ["units", 0, "text"]
        canonical = ["methods", 0, "sentences", 0, "text"]
    else:
        schema = schema_for("summary", {"u1": {}})
        value = {
            key: [
                {
                    "text": "First.\nSecond." if key == "abstract" else "Conclusion.",
                    "body_unit_refs": ["u1"],
                    "paragraph_start": True,
                }
            ]
            for key in ("abstract", "conclusion")
        }
        path = ["abstract", 0, "text"]
        canonical = ["abstract", 0, "sentences", 0, "text"]
    with pytest.raises(WriterError, match="writer_text_invalid") as failure:
        flat.parse(json.dumps(value), schema, locate_text_errors=True)
    diagnostic = failure.value.diagnostics[0]
    assert diagnostic["stage"] == kind
    assert diagnostic["wire_path"] == path and diagnostic["canonical_path"] == canonical
    assert "target_id" not in diagnostic


def final_messages(case, revision, monkeypatch):
    """Reuse an existing trusted fixture with the real final request constructor."""
    with monkeypatch.context() as local:
        if revision is not None:
            local.setattr(
                wire_identity,
                "scope",
                lambda: guidance_envelope(repair_guidance_wire_revision=revision),
            )
        (system, payload, schema), _, _, _ = wire_identity.build_input(
            case, selected=revision is not None
        )
    return request_messages(system, payload, schema, wire_contract=flat.CONTRACT)


def test_layout_instruction_reaches_only_the_selected_actual_request(
    guidance_case, monkeypatch
):
    assert flat.repair_guidance_wire_identity() == OLD_UNIT_IDENTITY
    assert (
        flat.repair_guidance_wire_identity(flat.REPAIR_TASK_VIEW_REVISION)
        != OLD_TASK_IDENTITY
    )
    old_messages = {
        revision: final_messages(guidance_case, revision, monkeypatch)
        for revision in (None, flat.REPAIR_GUIDANCE_WIRE_REVISION)
    }
    messages = final_messages(
        guidance_case, flat.REPAIR_TASK_VIEW_REVISION, monkeypatch
    )
    assert messages[0]["content"].count(flat.REPAIR_TASK_TEXT_INSTRUCTION) == 1
    assert all(
        flat.REPAIR_TASK_TEXT_INSTRUCTION not in value[0]["content"]
        for value in old_messages.values()
    )
    with monkeypatch.context() as local:
        local.setattr(
            flat,
            "REPAIR_TASK_VIEW_INSTRUCTION",
            flat.REPAIR_TASK_VIEW_INSTRUCTION
            + " Test-only selected identity mutation.",
        )
        assert flat.repair_guidance_wire_identity() == OLD_UNIT_IDENTITY
        for revision, expected in old_messages.items():
            assert final_messages(guidance_case, revision, monkeypatch) == expected


def test_checkpoint_from_pre_layout_task_identity_cannot_resume(monkeypatch):
    old_instruction = flat.REPAIR_TASK_VIEW_INSTRUCTION.removesuffix(
        " " + flat.REPAIR_TASK_TEXT_INSTRUCTION
    )
    with monkeypatch.context() as local:
        local.setattr(flat, "REPAIR_TASK_VIEW_INSTRUCTION", old_instruction)
        old_scope = guidance_envelope(
            repair_guidance_wire_revision=flat.REPAIR_TASK_VIEW_REVISION
        )
        assert old_scope["repair_guidance_wire_identity"] == OLD_TASK_IDENTITY
        old_plan = budget_plan(old_scope)
        checkpoint = PlannedBudget(old_plan).snapshot()
    new_scope = guidance_envelope(
        repair_guidance_wire_revision=flat.REPAIR_TASK_VIEW_REVISION
    )
    owner = PlannedBudget(budget_plan(new_scope))
    before = owner.snapshot()
    with pytest.raises(ValueError):
        owner.restore(checkpoint)
    assert owner.snapshot() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "revision", [None, "host-unit-ref-guidance-v1", "host-repair-task-view-v1"]
)
async def test_actual_sdk_text_refusal_keeps_usage_and_never_retries(
    case, tmp_path, monkeypatch, revision
):
    """Only external HTTP changes; all writer, budget and persistence guards run."""
    select_policy(monkeypatch)
    flags = {**structured.FLAGS}
    if revision is None:
        flags.pop("repair_guidance_wire_revision")
    else:
        flags["repair_guidance_wire_revision"] = revision
    monkeypatch.setattr(structured, "FLAGS", flags)
    parent = source_pipeline.SourceMapGateway

    class InvalidLayoutGateway(parent):
        def answer(self, payload, native_schema):
            result = super().answer(payload, native_schema)
            if payload.get("cause") == "capacity":
                result["patches"][0]["text"] += "\nNeutral illegal layout continuation."
            return result

    monkeypatch.setattr(source_pipeline, "SourceMapGateway", InvalidLayoutGateway)
    report, gateway = await source_pipeline.run_source_case(case, tmp_path, monkeypatch)
    state, journal, ledger = source_pipeline.products(tmp_path)
    repairs = source_pipeline.sent_with(gateway, "targets")
    assert len(repairs) == len(state["state"]["attempts"]) == 1
    assert not report["delivery"]["publishable"]
    assert journal["candidates"] == []
    call = journal["calls"]["repair_0"]
    assert call["status"] == "failed_no_replay"
    assert call["failure"]["code"] == "writer_text_invalid"
    raw = json.loads(call["raw"])
    assert "\n" in raw["patches"][0]["text"]
    assert call["raw_sha256"] == hashlib.sha256(call["raw"].encode()).hexdigest()
    assert "canonical_sha256" not in call and "wire_bindings" not in call
    if revision == flat.REPAIR_TASK_VIEW_REVISION:
        assert call["failure"]["diagnostics"] == [expected_location(raw, 0)]
        assert (
            flat.REPAIR_TASK_TEXT_INSTRUCTION
            in repairs[0]["request"]["input"][0]["content"]
        )
    else:
        assert call["failure"]["diagnostics"] == []
    sent = [entry for entry in ledger["entries"].values() if entry["sent"]]
    assert ledger["requests_sent"] == len(gateway.sent) == len(sent)
    assert ledger["usage_uncertain"] is False and all(
        row["status"] == "settled" for row in sent
    )
    assert ledger["used_tokens"] == sum(row["actual_tokens"] for row in sent)
    assert len([entry for entry in sent if entry["stage"] == "repair"]) == 1
