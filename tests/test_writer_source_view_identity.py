"""Bind the optional writer source view to frozen authority and historical data."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
from test_concise_repair_prompt import FLAGS as PARENT_FLAGS
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_reference_set_identity import envelope as parent_envelope
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
from litagent.quality import build_fact_cards, digest
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_accounting import _plan as accounting_plan
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writer_source_view import selected_source_view, source_view_identity
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    chapter_request_input,
    prepare_planned_input,
    provider_response_format,
    request_messages,
    source_packet,
)
from scripts import run_frozen_writing as campaign

VIEW = "source-only-writer-v1"
PAIR = {"writer_source_view", "writer_source_view_identity"}
FLAGS = {**PARENT_FLAGS, "evaluation_response_contract": "host-bound-evaluation-v2"}


def rehash(value):
    """Rehash caller data without creating host selection authority."""
    value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
    return value


def view_envelope(**changes):
    """Use the actual complete envelope producer for the selected view."""
    return parent_envelope(
        **{
            **FLAGS,
            "global_findings_contract": "anchored-global-findings-v2",
            "summary_constraints": True,
            "writer_source_view": VIEW,
            **changes,
        }
    )


def view_plan(bundle, **changes):
    """Obtain real frozen initial planning without constructing a model client."""
    return plan_frozen_budget(
        bundle,
        **{
            "budget_policy": "planned-budget-v7",
            **FLAGS,
            "writer_source_view": VIEW,
            **changes,
        },
    )


def view_manifest():
    """Keep the manifest pair complete, including all existing parent identities."""
    scope = view_envelope()
    return {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        **{key: scope[key] for key in PAIR},
        **{
            key: scope[key]
            for key in (
                "evaluation_response_identity",
                "repair_prompt_identity",
                "repair_guidance_wire_identity",
                "repair_guidance_capacity_rule",
            )
        },
    }


def test_view_is_independent_and_none_adds_no_fields_or_plan_changes(frozen_bundle):
    old = view_plan(frozen_bundle, writer_source_view=None)
    implicit = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **FLAGS
    )
    assert implicit == old and not PAIR.intersection(old["envelope"])
    assert not selected_source_view(old["envelope"])
    selected = view_plan(frozen_bundle)
    assert selected["status"] == "admitted"
    assert selected["identity"] != old["identity"]
    assert selected["envelope"]["source_identity"] == old["envelope"]["source_identity"]
    assert selected["envelope"]["writer_source_view"] == VIEW
    assert selected["envelope"]["writer_source_view_identity"] == source_view_identity()
    for key in (
        "capacity",
        "body_max_units",
        "summary_max_units_per_section",
    ):
        assert selected["envelope"][key] == old["envelope"][key]
    old_caps = old["envelope"]["request_caps"]
    caps = selected["envelope"]["request_caps"]
    assert {k: v for k, v in caps.items() if k != "body"} == {
        k: v for k, v in old_caps.items() if k != "body"
    }
    assert caps["body"] <= old_caps["body"]
    selection = EvidenceSelection.from_dict(frozen_bundle["selection"])
    packet, aliases = source_packet(
        build_fact_cards(selection.selected_items),
        selection,
        frozen_bundle["query"],
        frozen_bundle["requirements"],
        frozen_options().report_language,
    )
    reservations = []
    for section in BODY:
        system, payload, schema = chapter_request_input(
            packet, aliases, section, {}, {}, RANGE_WRITING_PLAN, selected["envelope"]
        )
        system, payload = prepare_planned_input(
            system, payload, packet, aliases, RANGE_WRITING_PLAN, selected["envelope"]
        )
        messages = request_messages(
            system, payload, schema, wire_contract="flat-writer-v1"
        )
        native = provider_response_format(
            "synthesis", schema, wire_contract="flat-writer-v1"
        )
        reservations.append(strict_request_reservation(messages, 20000, native))
    assert len(reservations) == 5
    assert (
        caps["body"] == max(reservations) + selected["envelope"]["body_json_bytes"] + 4
    )
    for stage in ("body_initial", "body", "summary", "repair"):
        assert selected["stages"][stage]["output_tokens"] == 20000
    assert selected["maximum_tokens"] == 4500000
    assert selected["maximum_requests"] == 64
    assert selected["repair_limit"] == selected["scope_bounds"]["shared_repairs"] == 2
    assert selected["context_reservation_limit"] == 117964
    assert accounting_envelope(selected["envelope"]).max_body_units == 66
    accounting_plan(selected)


@pytest.mark.parametrize("value", [False, True, 0, 1, "", "unknown", [], {}])
def test_frozen_view_selector_rejects_unknown_values_before_inputs(value):
    with pytest.raises(ValueError, match="writer_source_view"):
        view_plan({}, writer_source_view=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_policy": "planned-budget-v6"},
        {"evaluation_response_contract": None},
        {"evaluation_response_contract": "host-bound-evaluation-v1"},
        {"repair_prompt_revision": None},
        {"support_validation_revision": "host-support-receipts-v1"},
        {"repair_guidance_wire_revision": "host-unit-ref-guidance-v1"},
    ],
)
def test_view_never_infers_its_required_parent_stack(frozen_bundle, changes):
    with pytest.raises(ValueError):
        view_plan(frozen_bundle, **changes)


@pytest.mark.parametrize("field", sorted(PAIR))
def test_half_or_wrong_pair_rejects_even_after_rehash(field):
    for missing in (True, False):
        scope = view_envelope()
        if missing:
            scope.pop(field)
        else:
            scope[field] = "f" * 64 if field.endswith("identity") else "future-view"
        rehash(scope)
        before = copy.deepcopy(scope)
        with pytest.raises(ValueError):
            validate_writer_envelope(scope)
        assert scope == before


def test_pure_reader_accepts_pinned_historical_pair_without_executable_view(
    monkeypatch,
):
    import litagent.writer_source_view as view

    scope = view_envelope()
    scope["writer_source_view_identity"] = "f" * 64
    rehash(scope)
    with pytest.raises(ValueError):
        validate_writer_envelope(scope)
    monkeypatch.setattr(
        view, "source_view_identity", lambda: pytest.fail("current identity executed")
    )
    monkeypatch.setattr(
        view,
        "project_source_view",
        lambda *a, **kw: pytest.fail("current projector executed"),
    )
    assert accounting_envelope(scope).max_body_units == 66


def test_manifest_selection_has_no_public_option_and_requires_exact_parents():
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    assert campaign.writer_source_view_options({}) == {}
    assert not PAIR.intersection(SurveyRequest.model_fields)
    assert not PAIR.intersection(RunOptions.model_fields)
    manifest = view_manifest()
    before = copy.deepcopy(manifest)
    assert campaign.writer_source_view_options(manifest) == {"writer_source_view": VIEW}
    assert manifest == before
    for field in (
        *PAIR,
        "inventory_recovery",
        "evaluation_response_contract",
        "repair_prompt_revision",
    ):
        damaged = copy.deepcopy(manifest)
        damaged.pop(field)
        with pytest.raises(ValueError):
            campaign.writer_source_view_options(damaged)
    for changes in (
        {"writer_source_view_identity": "0" * 64},
        {"writing_profile": campaign.FLAT_PROFILE},
        {"writer_source_view": "future"},
    ):
        with pytest.raises(ValueError):
            campaign.writer_source_view_options({**manifest, **changes})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing_plan", "strip_selector", "wrong_identity", "old_plan"]
)
async def test_runtime_selection_requires_the_same_frozen_plan_before_agent_use(
    frozen_bundle, fault
):
    plan = view_plan(frozen_bundle)
    selector = VIEW
    if fault == "missing_plan":
        plan = None
    elif fault == "strip_selector":
        selector = None
    elif fault == "old_plan":
        plan = view_plan(frozen_bundle, writer_source_view=None)
    else:
        plan["envelope"]["writer_source_view_identity"] = "0" * 64
        rehash(plan["envelope"])
        rehash(plan)
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="writer_source_view"):
        await run_frozen(
            agent,
            frozen_bundle,
            "unstarted-source-view",
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            writer_source_view=selector,
            **FLAGS,
        )
    assert vars(agent) == {"_wired": False}


def historical_view_data(case):
    """Supply pinned record data, not a synthetic model or executable owner."""
    envelope = view_envelope()
    pair = {key: envelope[key] for key in PAIR}
    payload = {
        "query": "Controlled historical request",
        "sources": case.packet["sources"],
    }
    system = "Complete original source inspection."
    record = {
        "contract": VIEW,
        "identity": pair["writer_source_view_identity"],
        "source_snapshot": digest(case.packet),
        "view_sha": digest(payload),
        "system_sha": digest(system),
    }
    call = {
        **pair,
        "writer_source_view_record": record,
        "payload_sha256": digest(payload),
        "system_sha256": digest(system),
    }
    journal = {
        **pair,
        "writer_source_view_source_sha256": digest(case.packet),
        "calls": {"body:first": call},
    }
    start = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ]
    }
    archive = {"report": {"metadata": {"writer_document": {"packet": case.packet}}}}
    return envelope, journal, call, start, archive


def test_historical_view_receipts_bind_actual_payload_system_and_local_snapshot(
    case, monkeypatch
):
    import litagent.writer_source_view as view

    envelope, journal, call, start, archive = historical_view_data(case)
    monkeypatch.setattr(
        view, "source_view_identity", lambda: pytest.fail("historical runtime identity")
    )
    monkeypatch.setattr(
        view,
        "project_source_view",
        lambda *a, **kw: pytest.fail("historical projection"),
    )
    before = copy.deepcopy((envelope, journal, call, start, archive))
    campaign._historical_source_view_binding(journal, envelope)
    campaign._historical_source_view_archive(journal, envelope, archive)
    campaign._historical_source_view_request(start, envelope, call)
    # A technical failure can lack a complete report. The pinned journal remains
    # readable without fabricating a canonical packet or releasing unknown usage.
    campaign._historical_source_view_archive(journal, envelope, {"report": None})
    assert (envelope, journal, call, start, archive) == before


@pytest.mark.parametrize(
    "fault", ["pair", "record", "payload", "system", "snapshot", "archive"]
)
def test_historical_view_tampering_cannot_be_accepted_by_self_hashes(case, fault):
    envelope, journal, call, start, archive = historical_view_data(case)
    if fault == "pair":
        call["writer_source_view_identity"] = "0" * 64
    elif fault == "record":
        call.pop("writer_source_view_record")
    elif fault == "payload":
        start["messages"][1]["content"] = json.dumps({"query": "tampered"})
    elif fault == "system":
        start["messages"][0]["content"] += " Extra instruction."
    elif fault == "snapshot":
        call["writer_source_view_record"]["source_snapshot"] = "0" * 64
    else:
        archive["report"]["metadata"]["writer_document"]["packet"] = {"sources": {}}
    with pytest.raises(ValueError):
        campaign._historical_source_view_archive(journal, envelope, archive)
        campaign._historical_source_view_request(start, envelope, call)


def test_old_history_rejects_injected_local_view_authority():
    campaign._historical_source_view_binding({}, {})
    for value in (
        {"writer_source_view_source_sha256": "a" * 64},
        {"writer_source_view": VIEW},
    ):
        with pytest.raises(ValueError):
            campaign._historical_source_view_binding(value, {})


def test_live_cli_cannot_override_the_frozen_view_before_configuration(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_writing",
            "--live",
            "--root",
            str(tmp_path),
            "--writer-source-view",
            VIEW,
        ],
    )
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("live override reached config")
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2


def test_prepare_validate_and_execution_parameters_keep_the_same_pair(
    case, tmp_path, monkeypatch
):
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
    baseline = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable source-view preparation fixture."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    config = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    root = tmp_path / "prepared"
    prepared = campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        writer_source_view=VIEW,
        **FLAGS,
    )
    manifest = campaign.validate(root, config)
    assert manifest["identity"] == prepared["identity"]
    for container in (manifest, manifest["writing_parameters"]):
        assert container["writer_source_view"] == VIEW
        assert container["writer_source_view_identity"] == source_view_identity()
    for row in manifest["cases"]:
        assert manifest["budget_plans"][row["case_id"]] == view_plan(
            row["bundle"], capacity=config.capacity
        )
    assert not PAIR.intersection(manifest["options"])
    assert not (root / "state.json").exists() and not (root / ".live.lock").exists()
    for remove in [PAIR, {"writer_source_view_identity"}]:
        damaged = copy.deepcopy(manifest)
        for container in (damaged, damaged["writing_parameters"]):
            for key in remove:
                container.pop(key)
        rehash(damaged)
        (root / "manifest.json").write_text(json.dumps(damaged))
        with pytest.raises((ValueError, WriterError)):
            campaign.validate(root, config)
