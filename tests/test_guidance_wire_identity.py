"""Pin one opt-in guidance representation across real builders and frozen plans."""

import copy
import json
from types import SimpleNamespace

import pytest
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_structured_repair_guidance import context_for, guidance_case, rebind
from test_structured_repair_identity import (
    SELECTION,
    budget_plan,
    guidance_envelope,
    rehash,
)
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
from litagent.config import load_config
from litagent.frozen_writing import (
    plan_frozen_budget,
    run_frozen,
    validate_frozen_writer_plan,
)
from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_budget import PlannedBudget, validate_writer_envelope
from litagent.workbench import RunOptions
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    body_units,
    check_guidance_wire_record,
    prepare_planned_input,
    repair_request_input,
    request_messages,
    support_validation_fields,
)
from scripts import run_frozen_writing as campaign

REVISION = "host-unit-ref-guidance-v1"
FIELDS = {"repair_guidance_wire_revision", "repair_guidance_wire_identity"}


def scope(**changes):
    """Use the real complete selected envelope, with one optional wire revision."""
    return guidance_envelope(repair_guidance_wire_revision=REVISION, **changes)


def test_none_preserves_complete_parent_and_other_semantic_identities():
    parent = guidance_envelope()
    assert guidance_envelope(repair_guidance_wire_revision=None) == parent
    assert not FIELDS.intersection(parent)
    selected = scope()
    assert {k: v for k, v in selected.items() if k not in FIELDS | {"identity"}} == {
        k: v for k, v in parent.items() if k != "identity"
    }
    assert (
        selected["repair_guidance_wire_identity"]
        == flat.repair_guidance_wire_identity()
    )
    assert selected["identity"] != parent["identity"]
    for name in (
        "repair_guidance_identity",
        "transport_identity",
        "source_map_identity",
    ):
        assert selected[name] == parent[name]


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_unknown_wire_selector_never_coerces_into_a_valid_choice(bad):
    with pytest.raises(ValueError, match="repair_guidance_wire_policy_invalid"):
        guidance_envelope(repair_guidance_wire_revision=bad)


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_missing_half_of_wire_marker_is_rejected_after_rehash(field):
    value = scope()
    value.pop(field)
    rehash(value)
    with pytest.raises(
        ValueError, match="writer_envelope_repair_guidance_wire_invalid"
    ):
        validate_writer_envelope(value)
    with pytest.raises(ValueError):
        accounting_envelope(value)


@pytest.mark.parametrize(
    "changes",
    [
        {"repair_guidance_contract": None},
        {"repair_mapping_contract": "capacity-edit-map-v2"},
        {"support_validation_revision": None},
        {"repair_feedback": False},
        {"mixed_inventory_recovery": False},
    ],
)
def test_wire_revision_cannot_relax_a_structured_parent(changes):
    with pytest.raises(ValueError):
        scope(**changes)


def test_wire_identity_drift_changes_only_selected_owner_not_pinned_fee_data(
    monkeypatch,
):
    parent = guidance_envelope()
    selected = scope()
    plan = budget_plan(selected)
    before = copy.deepcopy(PlannedBudget(plan).snapshot())
    monkeypatch.setattr(flat, "repair_guidance_wire_identity", lambda: "e" * 64)
    assert guidance_envelope() == parent
    assert scope()["identity"] != selected["identity"]
    with pytest.raises(
        ValueError, match="writer_envelope_repair_guidance_wire_invalid"
    ):
        validate_writer_envelope(selected)
    assert accounting_envelope(selected).max_body_units == 66
    assert before["plan"] == plan


def test_actual_frozen_plan_none_is_full_value_compatible_and_new_choice_is_pinned(
    frozen_bundle,
):
    flags = {**SELECTION, "repair_guidance_contract": "host-structured-repair-v1"}
    old = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **flags)
    assert (
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            repair_guidance_wire_revision=None,
            **flags,
        )
        == old
    )
    selected = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        repair_guidance_wire_revision=REVISION,
        **flags,
    )
    assert selected["envelope"]["repair_guidance_wire_revision"] == REVISION
    assert (
        selected["envelope"]["builder_identity"] != old["envelope"]["builder_identity"]
    )
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(selected, old)
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(old, selected)
    assert selected["scope_bounds"]["shared_repairs"] == 2
    assert selected["maximum_tokens"] == 4500000 and selected["maximum_requests"] == 64


@pytest.mark.asyncio
async def test_new_wire_cannot_start_without_a_trusted_frozen_plan():
    with pytest.raises(ValueError, match="repair_guidance_wire_frozen_plan_required"):
        await run_frozen(
            SimpleNamespace(_wired=False),
            {},
            run_id="offline-unstarted",
            repair_guidance_wire_revision=REVISION,
        )


def build_input(case, *, selected=True, summary=False):
    """Use the final production builder, including real summary read-only views."""
    case = copy.deepcopy(case)
    body, targets, packet = case[:3]
    packet.update(
        query="A bounded source contract",
        language="en",
        requirements=[],
        chapter_recommendations={key: ["c1"] for key in BODY},
    )
    wrapper = context_for(case)
    summaries = None
    if summary:
        summaries = {
            "abstract": [
                {
                    "sentences": [
                        {"text": case[-1]["claim_text"], "body_unit_refs": ["u1"]}
                    ]
                }
            ]
        }
        fragment = next(iter(wrapper["host_directory"]["fragments"].values()))
        fragment["native_paths"] = [["abstract", 0, 0]]
        fragment["body_premise_paths"] = [["introduction", 0, 0]]
        wrapper["host_directory"]["current_body_sha256"] = digest(summaries)
        wrapper["host_directory"]["source_body_sha256"] = digest(summaries)
        rebind(wrapper)
        unit = summaries["abstract"][0]["sentences"][0]
        targets = {
            "r1": {
                "path": ["abstract", 0, 0],
                "original": unit,
                "original_sha256": digest(unit),
                "maximum": 1,
            }
        }
    else:
        targets = {"r1": targets["r1"]}
    envelope = scope() if selected else guidance_envelope()
    captured = []
    result = repair_request_input(
        packet,
        summaries if summary else body,
        targets,
        body_units(body) if summary else {"c1": "E1"},
        cause="evaluation",
        previous_failure=None,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=envelope,
        wire_contract="flat-writer-v1",
        summary_body=body if summary else None,
        guidance_context={"current": wrapper, "prior_candidate": None},
        guidance_wire_capture=captured.append,
    )
    return result, captured, packet, envelope


@pytest.mark.parametrize("summary", [False, True])
def test_final_builder_exposes_legend_complete_sources_and_summary_views(
    guidance_case, summary
):
    (system, payload, schema), captures, packet, envelope = build_input(
        guidance_case, summary=summary
    )
    assert len(captures) == 1
    record = captures[0]
    assert flat.REPAIR_GUIDANCE_WIRE_INSTRUCTION in system
    assert payload["repair_guidance"]["wire_revision"] == REVISION
    assert payload["sources"] == packet["sources"]
    assert not any(key in payload for key in ("authority", "receipt", "canonical"))
    if summary:
        assert payload["body_units"] == body_units(record["authority"]["summary_body"])
        assert payload["summary_source_body_sha256"] == digest(
            record["authority"]["summary_body"]
        )
    else:
        assert record["authority"]["summary_body"] is None
    check_guidance_wire_record(envelope, record, packet=packet)
    final_system, final_payload = prepare_planned_input(
        system,
        payload,
        packet,
        {"c1": "E1"},
        RANGE_WRITING_PLAN,
        envelope,
    )
    messages = request_messages(
        final_system, final_payload, schema, wire_contract="flat-writer-v1"
    )
    assert json.loads(messages[1]["content"]) == final_payload
    assert {key: final_payload["writing_envelope"][key] for key in FIELDS} == {
        key: envelope[key] for key in FIELDS
    }


@pytest.mark.parametrize("part", ["canonical", "receipt", "payload", "authority"])
def test_local_receipt_tampering_is_rejected_without_touching_the_wire(
    guidance_case, part
):
    _, captures, packet, envelope = build_input(guidance_case)
    record = copy.deepcopy(captures[0])
    record[part]["forged"] = True
    with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
        check_guidance_wire_record(envelope, record, packet=packet)


def test_none_final_builder_has_no_capture_or_wire_metadata(guidance_case):
    (system, payload, _), captured, _, _ = build_input(guidance_case, selected=False)
    assert captured == []
    assert flat.REPAIR_GUIDANCE_WIRE_INSTRUCTION not in system
    assert "wire_revision" not in payload["repair_guidance"]


def test_pure_history_binding_checks_actual_wire_payload_without_running_codec(
    guidance_case, monkeypatch
):
    (system, payload, schema), captures, packet, envelope = build_input(guidance_case)
    final_system, final_payload = prepare_planned_input(
        system,
        payload,
        packet,
        {"c1": "E1"},
        RANGE_WRITING_PLAN,
        envelope,
    )
    record = copy.deepcopy(captures[0])
    record["final_payload_sha256"] = digest(final_payload)
    record["identity"] = digest({k: v for k, v in record.items() if k != "identity"})
    call = {
        "stage": "repair",
        **support_validation_fields(envelope),
        **{k: envelope[k] for k in FIELDS},
        "guidance_wire_record": record,
    }
    start = {
        "messages": request_messages(
            final_system, final_payload, schema, wire_contract="flat-writer-v1"
        )
    }
    monkeypatch.setattr(
        flat,
        "restore_repair_guidance_wire",
        lambda *a, **k: pytest.fail("cost reader ran a codec"),
    )
    campaign._historical_call_binding(call, envelope)
    campaign._historical_writer_envelope(start, envelope, call=call)
    changed = copy.deepcopy(start)
    value = json.loads(changed["messages"][1]["content"])
    value["repair_guidance"]["current"]["units"].reverse()
    value["repair_guidance"]["current"]["header"]["coverage"]["complete"] = False
    changed["messages"][1]["content"] = json.dumps(value)
    with pytest.raises(ValueError, match="writer_repair_guidance_wire_receipt_invalid"):
        campaign._historical_writer_envelope(changed, envelope, call=call)


def test_manifest_marker_pair_is_internal_and_live_flag_fails_before_config(
    monkeypatch, tmp_path
):
    manifest = {
        "writing_profile": campaign.ROUND_PROFILE,
        **SELECTION,
        "repair_guidance_contract": "host-structured-repair-v1",
        "repair_guidance_wire_revision": REVISION,
        "repair_guidance_wire_identity": flat.repair_guidance_wire_identity(),
    }
    assert campaign.repair_guidance_wire_options(manifest) == {
        "repair_guidance_wire_revision": REVISION
    }
    assert not FIELDS.intersection(campaign.manifest_options(manifest))
    assert not FIELDS.intersection(RunOptions.model_fields)
    for key in FIELDS:
        partial = copy.deepcopy(manifest)
        partial.pop(key)
        with pytest.raises(
            ValueError, match="frozen_repair_guidance_wire_configuration_invalid"
        ):
            campaign.repair_guidance_wire_options(partial)
    monkeypatch.setattr(
        campaign,
        "load_config",
        lambda: pytest.fail("read config before rejecting live flag"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_frozen_writing",
            "--root",
            str(tmp_path / "unstarted"),
            "--live",
            "--repair-guidance-wire-revision",
            REVISION,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2 and not (tmp_path / "unstarted").exists()


@pytest.mark.parametrize("revision", [None, REVISION])
def test_real_prepare_and_validate_share_the_same_explicit_wire_choice(
    case, tmp_path, monkeypatch, revision
):
    """Mock file/Git inputs only, preserving the actual campaign consumers."""
    monkeypatch.chdir(tmp_path)
    archives = tmp_path / "artifacts/runs"
    archives.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        archive = {
            "run_id": run_id,
            "query": case.packet["query"],
            "events": [],
            "nodes": {
                "selection": {
                    "kind": "evidence.retrieve",
                    "output": case.selection.to_dict(),
                },
                "extract": {
                    "kind": "worker",
                    "task_id": "extract",
                    "agent_type": "extractor",
                    "output": [{"paper_id": "evit"}],
                },
            },
        }
        (archives / (run_id + ".json")).write_text(json.dumps(archive))
    history = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable test-only frozen input."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: "a" * 40)
    cfg = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    root = tmp_path / "prepared"
    prepared = campaign.prepare(
        root,
        cfg,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        repair_guidance_contract="host-structured-repair-v1",
        repair_guidance_wire_revision=revision,
        **SELECTION,
    )
    manifest = campaign.validate(root, cfg)
    assert prepared["identity"] == manifest["identity"]
    assert not FIELDS.intersection(manifest["options"])
    assert ("repair_guidance_wire_revision" in manifest) == (revision is not None)
    for row in manifest["cases"]:
        expected = plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=cfg.capacity,
            repair_guidance_contract="host-structured-repair-v1",
            repair_guidance_wire_revision=revision,
            **SELECTION,
        )
        assert manifest["budget_plans"][row["case_id"]] == expected
    if revision is not None:
        assert (
            manifest["repair_guidance_wire_identity"]
            == flat.repair_guidance_wire_identity()
        )
        assert (
            manifest["writing_parameters"]["repair_guidance_wire_revision"] == revision
        )
        manifest.pop("repair_guidance_wire_identity")
        with pytest.raises(
            ValueError, match="frozen_repair_guidance_wire_configuration_invalid"
        ):
            campaign.repair_guidance_wire_options(manifest)


@pytest.mark.asyncio
async def test_invalid_local_witness_is_rejected_before_shared_attempt_or_http(
    case, tmp_path
):
    """A host metadata failure never spends a repair merely to reject its input."""
    import test_structured_repair_pipeline as structured
    from test_native_responses import owner_for
    from test_native_wire_session import make_session

    from litagent.writing_v2 import writer_budget_plan

    gateway = structured.LocalGateway(case)
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **{**structured.FLAGS, "repair_guidance_wire_revision": REVISION},
    )
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    try:
        original = await session.initial_body()
        before_body = copy.deepcopy(original)
        before_calls = copy.deepcopy(session.journal["calls"])
        before_attempts = copy.deepcopy(session.state["attempts"])
        before_sent = len(gateway.sent)
        before_budget = budget.snapshot()
        before_records = copy.deepcopy(session.journal.get("guidance_wire_records", {}))
        before_known = sum(
            entry["actual_tokens"]
            for entry in before_budget["entries"].values()
            if entry["status"] == "settled"
        )
        before_unknown = sum(
            entry["reserved_tokens"]
            for entry in before_budget["entries"].values()
            if entry["sent"] and entry["status"] != "settled"
        )
        assert before_known > 0
        target = {
            "writer:methods:0:0": {
                "path": ["methods", 0, 0],
                "original": copy.deepcopy(original["methods"][0]["sentences"][0]),
                "explanation": "Check the exact local unit independently.",
            }
        }
        with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
            await session.patch(
                original,
                target,
                cause="local",
                guidance_replay_witness={"unexpected": True},
            )
        assert len(gateway.sent) == before_sent
        assert session.state["attempts"] == before_attempts
        assert session.journal["calls"] == before_calls
        assert session.journal.get("guidance_wire_records", {}) == before_records
        assert original == before_body
        after_budget = budget.snapshot()
        assert after_budget["requests_sent"] == before_budget["requests_sent"]
        assert after_budget["used_tokens"] == before_budget["used_tokens"]
        assert after_budget["usage_uncertain"] == before_budget["usage_uncertain"]
        assert after_budget["entries"] == before_budget["entries"]
        assert (
            sum(
                entry["actual_tokens"]
                for entry in after_budget["entries"].values()
                if entry["status"] == "settled"
            )
            == before_known
        )
        assert (
            sum(
                entry["reserved_tokens"]
                for entry in after_budget["entries"].values()
                if entry["sent"] and entry["status"] != "settled"
            )
            == before_unknown
        )
    finally:
        await owner.close()
