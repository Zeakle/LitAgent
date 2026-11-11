"""Bind explicit Evaluation v2 definitions without changing historical choices."""

import copy
import json
from types import SimpleNamespace

import pytest
from model_config_helpers import load_previous_default_config as load_config
from pydantic import ValidationError
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_guidance_wire_identity import build_input
from test_host_lossless_identity import selected_options as parent_options
from test_structured_repair_guidance import guidance_case  # noqa: F401
from test_structured_repair_identity import SELECTION, guidance_envelope, rehash
from test_writer_v2 import case  # noqa: F401

import litagent.eval.scoped as scoped
from litagent.frozen_writing import (
    plan_frozen_budget,
    validate_frozen_writer_plan,
)
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_budget import PlannedBudget, validate_writer_envelope
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    chapter_request_input,
    check_guidance_wire_record,
    prepare_planned_input,
    request_messages,
    support_validation_fields,
    writer_budget_plan,
)
from scripts import run_frozen_writing as campaign

V1 = "host-support-receipts-v1"
V2 = "host-support-receipts-v2"
FIELDS = {"support_validation_revision", "support_validation_identity"}
FLAGS = {
    **SELECTION,
    "repair_guidance_contract": "host-structured-repair-v1",
    "repair_guidance_wire_revision": "host-unit-ref-guidance-v1",
}
V1_IDENTITY_045 = "1cafb459f74f09ce5276aacebabc85bb8dbac3ee092ace9de01aa3039313f607"


def scope(revision=V2, **changes):
    """Select the unchanged full frozen parent with one existing version field."""
    return guidance_envelope(
        **{
            "repair_guidance_wire_revision": FLAGS["repair_guidance_wire_revision"],
            "support_validation_revision": revision,
            **changes,
        }
    )


def options(revision=V2, **changes):
    """Keep the existing RunOptions selector, without duplicate fixture kwargs."""
    return parent_options(**{"support_validation_revision": revision, **changes})


def planner(case, revision=V2):
    """Build the actual selected budget and request preflight together."""
    return writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **{**FLAGS, "support_validation_revision": revision},
    )


def test_zero_argument_and_explicit_v1_identity_match_immutable_045_golden():
    assert scoped.SUPPORT_VALIDATION_REVISION == V1
    assert scoped.support_validation_identity() == V1_IDENTITY_045
    assert scoped.support_validation_identity(V1) == V1_IDENTITY_045
    assert scoped.support_validation_identity(V2) != V1_IDENTITY_045
    assert scoped.SUPPORT_VALIDATION_REVISIONS == {V1, V2}


@pytest.mark.parametrize("bad", [None, False, True, 0, 1, "", "future", [], {}])
def test_explicit_identity_has_a_finite_version_domain(bad):
    with pytest.raises(ValueError, match="support_validation_revision_invalid"):
        scoped.support_validation_identity(bad)


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_invalid_new_selection_rejects_before_frozen_material_is_read(bad):
    with pytest.raises(ValueError):
        plan_frozen_budget(
            {},
            budget_policy="planned-budget-v7",
            **{**FLAGS, "support_validation_revision": bad},
        )


def test_new_version_uses_existing_fields_and_preserves_other_contracts():
    old, new = scope(V1), scope(V2)
    assert set(old) == set(new)
    assert validate_writer_envelope(new) == new
    assert accounting_envelope(new).max_body_units == 66
    assert support_validation_fields(new) == {
        "support_validation_revision": V2,
        "support_validation_identity": scoped.support_validation_identity(V2),
    }
    assert {k: v for k, v in old.items() if k not in FIELDS | {"identity"}} == {
        k: v for k, v in new.items() if k not in FIELDS | {"identity"}
    }


@pytest.mark.parametrize("missing", sorted(FIELDS))
def test_selected_pair_cannot_be_stripped_or_half_present(missing):
    value = scope()
    value.pop(missing)
    rehash(value)
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(value)


@pytest.mark.parametrize("revision,other", [(V1, V2), (V2, V1)])
def test_cross_version_identity_is_rejected_even_after_envelope_rehash(revision, other):
    value = scope(revision)
    value["support_validation_identity"] = scoped.support_validation_identity(other)
    rehash(value)
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(value)


@pytest.mark.parametrize(
    "change",
    [
        {"classification_review": False},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
    ],
)
def test_v2_retains_all_explicit_parent_requirements(change):
    with pytest.raises(ValueError):
        scope(**change)
    with pytest.raises((ValueError, ValidationError)):
        options(**change)


def test_changing_actual_v2_definition_preserves_none_v1_and_rejects_old_v2(
    case, monkeypatch
):
    base_flags = {
        k: v for k, v in SELECTION.items() if k != "support_validation_revision"
    }
    none_before = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **{**base_flags, "repair_mapping_contract": "capacity-edit-map-v2"},
    )
    v1_before, v2_before = planner(case, V1), planner(case, V2)
    options_v1 = options(V1)
    options_v2 = options(V2)
    fingerprints = (options_v1.fingerprint(), options_v2.fingerprint())
    definitions = copy.deepcopy(scoped.EVALUATION_DEFINITIONS_V2)
    definitions["atomic"] += " Explicit identity-only test definition change."
    monkeypatch.setattr(scoped, "EVALUATION_DEFINITIONS_V2", definitions)
    assert scoped.support_validation_identity() == V1_IDENTITY_045
    assert planner(case, V1) == v1_before
    assert (
        writer_budget_plan(
            case.packet,
            case.aliases,
            methodology={},
            budget_policy="planned-budget-v7",
            **{**base_flags, "repair_mapping_contract": "capacity-edit-map-v2"},
        )
        == none_before
    )
    assert options_v1.fingerprint() == fingerprints[0]
    assert options_v2.fingerprint() != fingerprints[1]
    assert planner(case, V2)["identity"] != v2_before["identity"]
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(v2_before["envelope"])
    assert accounting_envelope(v2_before["envelope"]).max_body_units == 66


@pytest.mark.parametrize("revision", [V1, V2])
def test_actual_generation_request_keeps_source_instruction_for_both_versions(
    case, revision
):
    plan = planner(case, revision)
    envelope = plan["envelope"]
    system, payload, schema = chapter_request_input(
        case.packet,
        case.aliases,
        "introduction",
        {},
        {},
        RANGE_WRITING_PLAN,
        envelope,
    )
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, envelope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    assert messages[0]["content"].count(scoped.WRITER_SUPPORT_RECEIPT_INSTRUCTION) == 1
    actual = json.loads(messages[1]["content"])
    assert actual["writing_envelope"]["support_validation_revision"] == revision
    assert "support_validation_identity" not in actual["writing_envelope"]
    assert plan["repair_limit"] == 2
    assert plan["maximum_tokens"] == 4500000 and plan["maximum_requests"] == 64


@pytest.mark.parametrize("summary", [False, True])
def test_final_repair_and_summary_records_use_the_new_owner_without_codec_change(
    guidance_case, monkeypatch, summary
):
    import test_guidance_wire_identity as wire_fixture

    monkeypatch.setattr(wire_fixture, "scope", lambda: scope(V2))
    (system, payload, schema), captures, packet, envelope = build_input(
        guidance_case, summary=summary
    )
    record = captures[0]
    check_guidance_wire_record(envelope, record, packet=packet)
    system, actual = prepare_planned_input(
        system, payload, packet, {"c1": "E1"}, RANGE_WRITING_PLAN, envelope
    )
    messages = request_messages(system, actual, schema, wire_contract="flat-writer-v1")
    assert actual["writing_envelope"]["support_validation_revision"] == V2
    assert messages[0]["content"].count(scoped.WRITER_SUPPORT_RECEIPT_INSTRUCTION) == 1
    if summary:
        assert record["authority"]["summary_body"] is not None
        assert payload["body_units"]


def test_real_frozen_plan_and_checkpoint_cannot_reuse_other_definition_version(
    frozen_bundle,
):
    old = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **FLAGS)
    new = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        **{**FLAGS, "support_validation_revision": V2},
    )
    assert old["envelope"]["support_validation_revision"] == V1
    assert new["envelope"]["support_validation_revision"] == V2
    assert new["envelope"][
        "support_validation_identity"
    ] == scoped.support_validation_identity(V2)
    assert new["repair_limit"] == 2 and new["status"] == "admitted"
    assert old["envelope"]["builder_identity"] != new["envelope"]["builder_identity"]
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(new, old)
    owner = PlannedBudget(new)
    before = owner.snapshot()
    with pytest.raises(ValueError, match="identity_conflict"):
        owner.restore(PlannedBudget(old).snapshot())
    assert owner.snapshot() == before


def test_journal_and_pure_cost_receipt_bind_the_exact_definition_pair(case, tmp_path):
    old, new = planner(case, V1), planner(case, V2)
    path = tmp_path / "repair.json"
    args = (path, "same-local-run", case.packet, case.aliases)
    kwargs = dict(
        provider_api="responses",
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )
    first = WriterSession(
        SimpleNamespace(_cost_budget=PlannedBudget(old)), *args, **kwargs
    )
    first.save()
    original = first.journal_path.read_bytes()
    with pytest.raises(WriterError, match="writer_checkpoint"):
        WriterSession(SimpleNamespace(_cost_budget=PlannedBudget(new)), *args, **kwargs)
    assert first.journal_path.read_bytes() == original
    pair = support_validation_fields(old["envelope"])
    campaign._historical_support_binding(pair, old["envelope"])
    with pytest.raises(ValueError, match="semantic_binding_invalid"):
        campaign._historical_support_binding(pair, new["envelope"])


def test_v2_is_not_a_new_public_request_field_and_live_cli_cannot_override(
    monkeypatch, tmp_path
):
    from litagent.api import SurveyRequest

    assert "support_validation_revision" not in SurveyRequest.model_fields
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurveyRequest(query="Compare methods.", support_validation_revision=V2)
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("loaded live config")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "frozen",
            "--root",
            str(tmp_path / "not-created"),
            "--live",
            "--support-validation-revision",
            V2,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2 and not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("revision", [V1, V2])
def test_actual_campaign_prepare_validate_and_three_plan_rebuilds_keep_revision(
    case, tmp_path, monkeypatch, revision
):
    """Replace file/Git inputs only; exercise the real campaign and planner."""
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
        **{**FLAGS, "support_validation_revision": revision},
    )
    manifest = campaign.validate(root, cfg)
    assert prepared["identity"] == manifest["identity"]
    assert manifest["support_validation_revision"] == revision
    assert manifest["options"]["support_validation_revision"] == revision
    assert manifest["writing_parameters"]["support_validation_revision"] == revision
    for row in manifest["cases"]:
        actual = plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=cfg.capacity,
            **{**FLAGS, "support_validation_revision": revision},
        )
        assert manifest["budget_plans"][row["case_id"]] == actual
        assert actual["envelope"]["support_validation_revision"] == revision


@pytest.mark.parametrize("revision", [V1, V2])
def test_run_options_forward_the_exact_version_to_actual_synthesis_graph(revision):
    from litagent.orchestrator.task_graph import SubTask, TaskGraph
    from litagent.runner import LitAgent

    graph = TaskGraph()
    graph.add_task(SubTask("write", "Write the frozen report", "synthesis"))
    LitAgent._apply_run_options_to_graph(graph, options(revision))
    assert graph.tasks["write"].input_data["support_validation_revision"] == revision
