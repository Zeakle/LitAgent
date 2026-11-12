"""Keep opt-in reply reminders schema-derived and protocol rejection strict."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_writer_v2 import case  # noqa: F401

from litagent.eval.batched import GLOBAL_PROMPT, global_prompt
from litagent.eval.inventory_recovery import CONTRACT, MIXED_CONTRACT
from litagent.eval.scoped import (
    MIXED_GLOBAL_BASE_PROMPT,
    MIXED_GLOBAL_FINDINGS_INSTRUCTION,
    mixed_global_findings_identity,
)
from litagent.quality import digest
from litagent.safety.budget import strict_request_reservation
from litagent.writing_flat import (
    REPLY_LAYOUT_CONTRACT,
    encode,
    parse,
    prepare,
    reply_layout_instruction,
)
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    body_section_schema,
    body_units,
    chapter_request_input,
    prepare_planned_input,
    provider_response_format,
    repair_request_input,
    request_messages,
    schema_for,
    summary_request_input,
    writer_budget_plan,
)


@pytest.fixture
def vit035():
    """Keep original public provider prose and schema, including its invalid root."""
    value = json.loads(
        (Path(__file__).parent / "fixtures/mixed-vit035-reply.json").read_text()
    )
    assert hashlib.sha256(value["raw"].encode()).hexdigest() == value["raw_sha256"]
    assert digest(value["host_schema"]) == value["host_schema_sha256"]
    assert digest(value["wire_schema"]) == value["wire_schema_sha256"]
    return value


def response_shape(case, kind):
    """Use the actual host schema producers for each supported reply stage."""
    if kind in {"first_chapter", "other_chapter"}:
        section = BODY[0] if kind == "first_chapter" else "methods"
        return body_section_schema(section, case.aliases, maximum=8)
    if kind == "summary":
        return schema_for("summary", body_units(case.body))
    return schema_for(
        "repair",
        body_units(case.body) if kind == "summary_patch" else case.aliases,
        {"target": {}},
        summary_repair=kind == "summary_patch",
    )


@pytest.mark.parametrize(
    "kind", ["first_chapter", "other_chapter", "summary", "body_patch", "summary_patch"]
)
def test_reply_layout_reads_actual_root_and_item_properties_without_new_schema(
    case, kind
):
    schema = response_shape(case, kind)
    before = copy.deepcopy(schema)
    native = provider_response_format(
        "synthesis", schema, wire_contract="flat-writer-v1"
    )
    wire = prepare(schema)["wire_schema"]
    instruction = reply_layout_instruction(schema)
    root, items = instruction.split("root keys=", 1)[1].split(
        "; item keys by root array="
    )
    assert json.loads(root) == list(wire["properties"])
    actual_items, _ = json.JSONDecoder().raw_decode(items)
    expected_items = {
        key: list(value["items"]["properties"])
        for key, value in wire["properties"].items()
        if isinstance(value.get("items"), dict) and "properties" in value["items"]
    }
    assert actual_items == expected_items
    assert "fact_card_refs" not in json.loads(root)
    assert "body_unit_refs" not in json.loads(root)
    if kind == "first_chapter":
        assert json.loads(root) == ["title", "units"]
    elif kind == "other_chapter":
        assert json.loads(root) == ["units"]
    elif kind == "summary":
        assert set(json.loads(root)) == {"abstract", "conclusion"}
    else:
        assert json.loads(root) == ["outcome", "explanation", "patches"]
        assert "target_id" in actual_items["patches"]
    for marker in (None, CONTRACT, MIXED_CONTRACT):
        payload = {"writing_envelope": {"inventory_recovery_contract": marker}}
        messages = request_messages(
            "Existing instruction.", payload, schema, wire_contract="flat-writer-v1"
        )
        assert (REPLY_LAYOUT_CONTRACT in messages[0]["content"]) is (
            marker == MIXED_CONTRACT
        )
        assert messages[0]["content"].count(REPLY_LAYOUT_CONTRACT) <= 1
    assert (
        provider_response_format("synthesis", schema, wire_contract="flat-writer-v1")
        == native
    )
    assert schema == before


def test_035_vit_complete_original_reply_remains_rejected_without_field_deletion(
    vit035,
):
    before = copy.deepcopy(vit035)
    value = json.loads(vit035["raw"])
    assert list(value) == ["units", "fact_card_refs"]
    assert value["fact_card_refs"] == []
    assert len(value["units"]) == vit035["received_items"] == 22
    item_schema = prepare(vit035["host_schema"])["wire_schema"]["properties"]["units"][
        "items"
    ]
    for unit in value["units"]:
        Draft202012Validator(item_schema).validate(unit)
    with pytest.raises(WriterError, match="writer_flat_schema_invalid") as legacy:
        parse(vit035["raw"], vit035["host_schema"], allow_eof=True)
    assert legacy.value.diagnostics == [{"path": [], "reason": "additionalProperties"}]
    with pytest.raises(WriterError, match="writer_flat_schema_invalid") as mixed:
        parse(
            vit035["raw"],
            vit035["host_schema"],
            allow_eof=True,
            diagnostic_stage="synthesis",
        )
    diagnostic = mixed.value.diagnostics[0]
    assert diagnostic["path"] == [] and diagnostic["reason"] == "additionalProperties"
    assert diagnostic["extra_root_fields"] == ["fact_card_refs"]
    assert diagnostic["allowed_root_fields"] == ["units"]
    assert diagnostic["wire_schema_sha256"] == vit035["wire_schema_sha256"]
    assert diagnostic["stage"] == "synthesis" and diagnostic["reply_kind"] == "chapter"
    assert vit035 == before


@pytest.mark.parametrize(
    "raw", ['{"units":', '{"units":[],"units":[]}', '{"units":"wrong"}']
)
def test_new_diagnostic_context_does_not_accept_bad_json_duplicate_keys_or_types(
    case, raw
):
    with pytest.raises(WriterError):
        parse(
            raw,
            body_section_schema("methods", case.aliases),
            diagnostic_stage="synthesis",
        )


@pytest.mark.parametrize("flag", [None, 1, "true", [], {}])
def test_mixed_plan_flag_is_strict_boolean(case, flag):
    with pytest.raises(ValueError, match="mixed_inventory_recovery_policy_invalid"):
        writer_budget_plan(
            case.packet,
            case.aliases,
            methodology={},
            budget_policy="planned-budget-v7",
            inventory_recovery=True,
            mixed_inventory_recovery=flag,
        )


@pytest.mark.parametrize(
    "policy,recovery", [("planned-budget-v6", True), ("planned-budget-v7", False)]
)
def test_mixed_plan_requires_v7_and_existing_recovery(case, policy, recovery):
    with pytest.raises(ValueError):
        writer_budget_plan(
            case.packet,
            case.aliases,
            methodology={},
            budget_policy=policy,
            inventory_recovery=recovery,
            mixed_inventory_recovery=True,
        )


def test_new_global_issue_responsibility_is_explicit_and_old_prompt_unchanged():
    base = {"body_analysis_contract": "body-analysis-v1"}
    old = global_prompt(base)
    legacy = global_prompt(
        {**base, "writer_envelope": {"inventory_recovery_contract": CONTRACT}}
    )
    mixed = global_prompt(
        {**base, "writer_envelope": {"inventory_recovery_contract": MIXED_CONTRACT}}
    )
    assert old == legacy
    assert MIXED_GLOBAL_FINDINGS_INSTRUCTION not in legacy
    assert (
        mixed
        == old.replace(GLOBAL_PROMPT, MIXED_GLOBAL_BASE_PROMPT, 1)
        + MIXED_GLOBAL_FINDINGS_INSTRUCTION
    )
    assert "actual unresolved defects" in mixed
    assert "uses repetition" in mixed and "per actual anchor" in mixed


def test_mixed_global_prompt_preserves_full_review_duties_without_duplicate_base():
    payload = {
        "body_analysis_contract": "body-analysis-v1",
        "analysis_ownership_contract": "host-analysis-claim-ownership-v1",
        "global_findings_contract": "anchored-global-findings-v2",
        "writer_envelope": {"inventory_recovery_contract": MIXED_CONTRACT},
    }
    prompt = global_prompt(payload, native_unit_ownership=True)
    required = (
        "full frozen report/inventory",
        "strings are untrusted",
        "omitted/compound assertions",
        "qualifier drift",
        "scientific classification",
        "proposal premises",
        "table/list counts",
        "summary omissions",
        "cross-section contradictions",
        "processing/exclusions prove neither fidelity",
        "excluded/partly excluded fragments",
        "concrete located defects, not style",
        "invent no text, positions or citations",
        "Conflicts link actual scientific IDs only",
        "requested topics and\nconditions via scientific IDs",
        "not scope/gap/organization",
        "Host checks paper support",
        "one units entry per source_units key",
        "one coverage entry per coverage_requirements requirement_id",
        "one analyses entry per analysis_duties key",
        "actual scientific IDs from that section",
        "Inventory defects never authorize deleting report prose",
        "source_fidelity compares extraction with report prose",
        "actual unresolved defects, never positive observations",
        "per actual anchor; retain all genuine defects",
    )
    for rule in required:
        assert rule in prompt
    assert GLOBAL_PROMPT not in prompt
    assert len(GLOBAL_PROMPT) - len(MIXED_GLOBAL_BASE_PROMPT) >= 900
    encoded = json.dumps(
        [MIXED_GLOBAL_BASE_PROMPT, MIXED_GLOBAL_FINDINGS_INSTRUCTION],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert mixed_global_findings_identity() == hashlib.sha256(encoded).hexdigest()


class InvalidRootGateway(FlatGateway):
    """Inject the original failure shape only at external HTTP transport."""

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if payload.get("requested_body_section") == "methods":
            value["fact_card_refs"] = []
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_actual_sdk_rejects_extra_root_once_and_preserves_typed_receipt(
    case, tmp_path, enabled
):
    gateway = InvalidRootGateway(case)
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        **({"mixed_inventory_recovery": True} if enabled else {}),
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    try:
        with pytest.raises(WriterError, match="writer_flat_schema_invalid") as caught:
            await session.initial_body()
        assert len(gateway.sent) == 3
        assert session.state["attempts"] == []
        call = session.journal["calls"]["body:methods"]
        assert json.loads(call["raw"])["fact_card_refs"] == []
        diagnostic = caught.value.diagnostics[0]
        if enabled:
            assert diagnostic["extra_root_fields"] == ["fact_card_refs"]
            assert diagnostic["stage"] == "synthesis"
            assert diagnostic["allowed_root_fields"] == ["units"]
        else:
            assert diagnostic == {"path": [], "reason": "additionalProperties"}
        for sent in gateway.sent:
            messages = sent["request"]["input"]
            assert (REPLY_LAYOUT_CONTRACT in messages[0]["content"]) is enabled
        assert not budget.snapshot()["usage_uncertain"]
    finally:
        await owner.close()


def test_new_layout_appears_in_known_actual_budget_request_not_only_runtime(case):
    old = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
    )
    new = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
    )
    assert old["status"] == new["status"] == "admitted"
    assert old["identity"] != new["identity"]
    for plan in (old, new):
        system, payload, schema = chapter_request_input(
            case.packet,
            case.aliases,
            BODY[0],
            {},
            {},
            RANGE_WRITING_PLAN,
            plan["envelope"],
        )
        system, payload = prepare_planned_input(
            system,
            payload,
            case.packet,
            case.aliases,
            RANGE_WRITING_PLAN,
            plan["envelope"],
        )
        messages = request_messages(
            system, payload, schema, wire_contract="flat-writer-v1"
        )
        native = provider_response_format(
            "synthesis", schema, wire_contract="flat-writer-v1"
        )
        reservation = strict_request_reservation(messages, 20000, native)
        assert reservation <= plan["envelope"]["request_caps"]["body"]
        assert (REPLY_LAYOUT_CONTRACT in messages[0]["content"]) is (plan is new)


@pytest.mark.parametrize(
    "stage",
    ["first_chapter", "other_chapter", "summary", "body_patch", "summary_patch"],
)
def test_actual_stage_builders_share_only_the_new_envelope_layout_reminder(case, stage):
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
    )
    envelope = plan["envelope"]
    if stage in {"first_chapter", "other_chapter"}:
        request = chapter_request_input(
            case.packet,
            case.aliases,
            BODY[0] if stage == "first_chapter" else "methods",
            {},
            {},
            RANGE_WRITING_PLAN,
            envelope,
        )
    elif stage == "summary":
        request = summary_request_input(case.packet, body_units(case.body), envelope)
    else:
        summary = stage == "summary_patch"
        document = case.summaries if summary else case.body
        section = "abstract" if summary else "methods"
        targets = {
            "target": {
                "path": [section, 0, 0],
                "original": document[section][0]["sentences"][0],
                "explanation": "Controlled exact source-bound target.",
            }
        }
        request = repair_request_input(
            case.packet,
            document,
            targets,
            body_units(case.body) if summary else case.aliases,
            cause="evaluation",
            previous_failure=None,
            writing_plan=RANGE_WRITING_PLAN,
            envelope=envelope,
            wire_contract="flat-writer-v1",
            **({"summary_body": case.body} if summary else {}),
        )
    system, payload, schema = request
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, envelope
    )
    assert payload["writing_envelope"]["inventory_recovery_contract"] == MIXED_CONTRACT
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    assert messages[0]["content"].count(REPLY_LAYOUT_CONTRACT) == 1
    assert reply_layout_instruction(schema) in messages[0]["content"]


def mixed_alias_request():
    """Build a transport fixture with explicit host identities and prior concerns."""
    from litagent.eval.scoped import flat_inventory_schema

    units = [
        {
            "source_unit_id": f"unit:{index}",
            "source_fragments": [
                {
                    "source_fragment_id": f"fragment:{index}",
                    "text": f"Complete proposition {index}. [E:e{index}]",
                    "citation_aliases": [f"e{index}"],
                }
            ],
        }
        for index in [1, 2]
    ]
    feedback = {
        "contract": MIXED_CONTRACT,
        "source_report_sha256": "a" * 64,
        "previous_inventory_sha256": "b" * 64,
        "bindings_sha256": "c" * 64,
        "findings": [
            {
                "origin": "global",
                "issue_index": 0,
                "source_unit_id": "unit:1",
                "source_fragment_ids": ["fragment:1"],
                "claim_ids": ["prior:one"],
                "code": "classification",
                "explanation": "Keep this unresolved concern and its condition.",
            }
        ],
        "concern_claims": [
            {
                "claim_id": "prior:one",
                "source_unit_id": "unit:1",
                "source_fragment_id": "fragment:1",
                "claim_text": "Complete proposition 1.",
                "kind": "report_structure",
                "correct_kind": "scientific",
                "atomic": True,
                "classification_valid": False,
                "source_faithful": True,
                "evidence_ids": ["paper:A"],
            }
        ],
        "global_finding_bindings": [
            {
                "finding_index": 0,
                "code": "classification",
                "anchor_ids": ["claim:prior:one", "unit:unit:1"],
                "source_unit_ids": ["unit:1"],
                "explanation": "Keep the original anchor explanation.",
            }
        ],
    }
    payload = {
        "source_units": units,
        "allowed_citation_aliases": ["e1", "e2"],
        "inventory_recovery": feedback,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "host_flat_fragment_inventory_v1",
            "strict": True,
            "schema": flat_inventory_schema(units),
        },
    }
    return payload, response_format, {"paper:A": "e1", "paper:B": "e2"}


def test_mixed_prior_aliases_are_losslessly_bound_to_actual_input_and_host_evidence():
    from litagent.writing_flat import inventory_identity_projection

    payload, schema, evidence = mixed_alias_request()
    before = copy.deepcopy((payload, schema, evidence))
    wire, native = inventory_identity_projection(
        payload, schema, evidence_aliases=evidence
    )
    prior = wire["inventory_recovery"]
    assert prior["findings"][0]["source_fragment_ids"] == ["f1"]
    assert prior["findings"][0]["source_unit_id"] == "v1"
    assert prior["findings"][0]["claim_ids"] == ["p1"]
    assert prior["concern_claims"][0]["source_fragment_id"] == "f1"
    assert prior["concern_claims"][0]["evidence_ids"] == ["e1"]
    assert prior["global_finding_bindings"][0]["anchor_ids"] == ["claim:p1", "unit:v1"]
    assert prior["global_finding_bindings"][0]["source_unit_ids"] == ["v1"]
    assert [unit["source_unit_id"] for unit in wire["source_units"]] == ["v1", "v2"]
    expected = copy.deepcopy(payload)
    expected["source_units"] = [
        {
            **unit,
            "source_unit_id": f"v{index}",
            "source_fragments": [
                {**fragment, "source_fragment_id": f"f{index}"}
                for fragment in unit["source_fragments"]
            ],
        }
        for index, unit in enumerate(payload["source_units"], 1)
    ]
    expected["inventory_recovery"]["findings"][0].update(
        source_fragment_ids=["f1"], source_unit_id="v1", claim_ids=["p1"]
    )
    expected["inventory_recovery"]["concern_claims"][0].update(
        source_fragment_id="f1", source_unit_id="v1", claim_id="p1", evidence_ids=["e1"]
    )
    expected["inventory_recovery"]["global_finding_bindings"][0].update(
        anchor_ids=["claim:p1", "unit:v1"], source_unit_ids=["v1"]
    )
    assert wire == expected
    assert list(
        native["json_schema"]["schema"]["properties"]["fragments"]["properties"]
    ) == ["f1", "f2"]
    assert (payload, schema, evidence) == before


@pytest.mark.parametrize(
    "fault",
    [
        "fragment",
        "unit",
        "claim_anchor",
        "unit_anchor",
        "evidence",
        "evidence_alias",
        "no_host_map",
    ],
)
def test_mixed_aliases_reject_foreign_ids_and_never_guess_evidence_mapping(fault):
    from litagent.writing_flat import inventory_identity_projection

    payload, schema, evidence = mixed_alias_request()
    recovery = payload["inventory_recovery"]
    if fault == "fragment":
        recovery["findings"][0]["source_fragment_ids"] = ["not-present"]
    elif fault == "unit":
        recovery["findings"][0]["source_unit_id"] = "unit:2"
    elif fault == "claim_anchor":
        recovery["global_finding_bindings"][0]["anchor_ids"] = ["claim:not-present"]
    elif fault == "unit_anchor":
        recovery["global_finding_bindings"][0]["anchor_ids"] = ["unit:not-present"]
    elif fault == "evidence":
        recovery["concern_claims"][0]["evidence_ids"] = ["foreign:source"]
    elif fault == "evidence_alias":
        evidence["paper:A"] = "e2"
        evidence["paper:B"] = "e1"
    else:
        evidence = None
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="inventory_recovery_alias_identity_invalid"):
        inventory_identity_projection(payload, schema, evidence_aliases=evidence)
    assert payload == before


@pytest.mark.asyncio
async def test_mixed_recovery_actual_sdk_transmits_complete_prior_concerns_with_bound_aliases(
    case, tmp_path, monkeypatch
):
    from test_mixed_inventory_pipeline import execute

    from litagent.eval.inventory_recovery import inventory_recovery_feedback
    from litagent.evidence import extract_evidence_refs_ordered

    report, gateway = await execute(case, tmp_path, monkeypatch)
    first = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    canonical = inventory_recovery_feedback(
        report["survey"], first, contract=MIXED_CONTRACT
    )
    payload = next(
        row["payload"] for row in gateway.sent if "inventory_recovery" in row["payload"]
    )
    wire = payload["inventory_recovery"]
    assert len(wire["findings"]) == len(canonical["findings"])
    assert len(wire["concern_claims"]) == len(canonical["concern_claims"])
    for field in [
        "contract",
        "source_report_sha256",
        "previous_inventory_sha256",
        "bindings_sha256",
        "assessment_sha256",
        "writer_bindings_sha256",
    ]:
        assert wire[field] == canonical[field]
    for old, new in zip(canonical["findings"], wire["findings"], strict=True):
        assert old["explanation"] == new["explanation"] and old["code"] == new["code"]
    evidence = {
        eid: f"e{index}"
        for index, eid in enumerate(extract_evidence_refs_ordered(report["survey"]), 1)
    }
    for index, (old, new) in enumerate(
        zip(canonical["concern_claims"], wire["concern_claims"], strict=True), 1
    ):
        assert new["claim_id"] == f"p{index}"
        assert new["evidence_ids"] == [evidence[eid] for eid in old["evidence_ids"]]
        for field in [
            "claim_text",
            "kind",
            "correct_kind",
            "atomic",
            "classification_valid",
            "source_faithful",
        ]:
            assert new[field] == old[field]
    assert report["delivery"]["publishable"]


def test_recovery_rejection_records_exact_prepared_request_without_rebuilding_aliases(
    monkeypatch,
):
    from litagent.config import EvalConfig
    from litagent.eval.scoped import (
        _record_request_rejection,
        _request_identity,
        build_evaluation_request,
    )

    payload, schema, evidence = mixed_alias_request()
    payload.update(
        inventory_output_contract="host-flat-fragment-inventory-v1",
        evaluation_input_contract="flat-evaluation-input-v1",
        schema=schema["json_schema"]["schema"],
    )
    messages, native = build_evaluation_request(
        "Retain every prior concern.", payload, recovery_evidence_aliases=evidence
    )

    def no_rebuild(*_args, **_kwargs):
        raise AssertionError(
            "A rejected prepared request must not lose its host alias map."
        )

    monkeypatch.setattr("litagent.eval.scoped.build_evaluation_request", no_rebuild)
    progress, events = {}, []
    config = EvalConfig(thinking_mode="disabled")
    llm = SimpleNamespace(
        _cost_budget=None, _trace_hook=lambda name, data: events.append((name, data))
    )
    _record_request_rejection(
        llm,
        "Retain every prior concern.",
        payload,
        reason="evaluation_context_budget_exceeded",
        phase="inventory",
        config=config,
        progress=progress,
        messages=messages,
        response_format=native,
    )
    assert progress["rejected_requests"][0]["input_sha256"] == _request_identity(
        messages, native
    )
    assert (
        progress["rejected_requests"][0]["reason_code"]
        == "evaluation_context_budget_exceeded"
    )
    assert events[0][1]["messages"] == messages
    assert events[0][1]["response_format"] == native
