"""Bind opt-in support receipt semantics across plans, wire and recovery."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_host_lossless_identity import FLAGS as BASE_FLAGS
from test_host_lossless_identity import selected_envelope as parent_envelope
from test_host_lossless_identity import selected_options as parent_options
from test_reference_set_identity import options_for
from test_writer_v2 import case  # noqa: F401

from litagent.config import AppConfig
from litagent.eval.scoped import (
    SUPPORT_VALIDATION_REVISION,
    WRITER_SUPPORT_RECEIPT_INSTRUCTION,
    support_validation_identity,
)
from litagent.frozen_writing import (
    plan_frozen_budget,
    run_frozen,
    validate_frozen_writer_plan,
)
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import digest
from litagent.runner import LitAgent
from litagent.safety.planned_budget import (
    PlannedBudget,
    _digest,
    validate_writer_envelope,
)
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    chapter_request_input,
    prepare_planned_input,
    request_messages,
    support_validation_fields,
    writer_budget_plan,
)
from scripts import run_frozen_writing as campaign

REVISION = SUPPORT_VALIDATION_REVISION
FLAGS = {**BASE_FLAGS, "transport_revision": "host-lossless-transport-v1"}


def selected_options(**changes):
    """Select the new policy on the complete immutable parent configuration."""
    return parent_options(support_validation_revision=REVISION, **changes)


def selected_envelope(**changes):
    """Construct the actual selected envelope without replacing the producer."""
    return parent_envelope(support_validation_revision=REVISION, **changes)


@pytest.mark.parametrize(
    "policy", ["planned-budget-v5", "planned-budget-v6", "planned-budget-v7"]
)
def test_none_preserves_historical_options_serialization_and_fingerprint(policy):
    old = options_for(policy)
    explicit = options_for(policy, support_validation_revision=None)
    assert old.model_dump(mode="json") == explicit.model_dump(mode="json")
    assert old.fingerprint() == explicit.fingerprint()
    assert "support_validation_revision" not in explicit.model_dump(mode="json")


@pytest.mark.parametrize("wrong", [True, False, 0, 1, "", "unknown", {}, []])
def test_invalid_selector_types_are_not_coerced(wrong):
    with pytest.raises(ValidationError):
        parent_options(support_validation_revision=wrong)


@pytest.mark.parametrize(
    "change",
    [
        {"generation_strategy": "body_summary_v2"},
        {"classification_review": False},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
    ],
)
def test_options_require_every_explicit_parent(change):
    with pytest.raises(
        ValueError, match="support_validation_requires_explicit_v7_contracts"
    ):
        selected_options(**change)


def test_selector_remains_absent_from_public_survey_request():
    from litagent.api import SurveyRequest

    assert "support_validation_revision" not in SurveyRequest.model_fields
    assert (
        "support_validation_revision"
        not in SurveyRequest.model_json_schema()["properties"]
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurveyRequest(
            query="Compare the public methods.", support_validation_revision=REVISION
        )


def test_derived_semantics_changes_only_explicit_identity(monkeypatch):
    import litagent.eval.scoped as scoped

    old, selected = parent_options(), selected_options()
    old_fp, selected_fp = old.fingerprint(), selected.fingerprint()
    scope = selected_envelope()
    assert scope["support_validation_revision"] == REVISION
    assert scope["support_validation_identity"] == support_validation_identity()
    assert validate_writer_envelope(scope) == scope
    old_scope = parent_envelope()
    assert "support_validation_revision" not in old_scope
    assert parent_envelope(support_validation_revision=None) == old_scope
    monkeypatch.setattr(scoped, "support_validation_identity", lambda: "f" * 64)
    assert old.fingerprint() == old_fp
    assert selected.fingerprint() != selected_fp
    assert parent_envelope() == old_scope
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(scope)


@pytest.mark.parametrize(
    "instruction",
    ["SUPPORT_VALIDATION_INSTRUCTION", "WRITER_SUPPORT_RECEIPT_INSTRUCTION"],
)
def test_actual_instruction_text_invalidates_only_selected_recovery(
    case, tmp_path, monkeypatch, instruction
):
    """Bind real prompt bytes to opted-in plans without rewriting old requests."""
    import litagent.eval.scoped as scoped

    def request_bytes(plan):
        """Serialize the actual chapter prompt, payload and canonical schema."""
        system, payload, schema = chapter_request_input(
            case.packet,
            case.aliases,
            "introduction",
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
        return json.dumps(
            [
                request_messages(
                    system, payload, schema, wire_contract="flat-writer-v1"
                ),
                schema,
            ],
            ensure_ascii=False,
        ).encode()

    historical = {}
    for policy in ("planned-budget-v5", "planned-budget-v6", "planned-budget-v7"):
        options = options_for(policy, support_validation_revision=None)
        plan = writer_budget_plan(
            case.packet, case.aliases, budget_policy=policy, methodology={}
        )
        historical[policy] = (
            options.model_dump_json().encode(),
            options.fingerprint(),
            plan,
            request_bytes(plan),
        )
    common = {"budget_policy": "planned-budget-v7", "methodology": {}, **FLAGS}
    selected = selected_options()
    old_fingerprint = selected.fingerprint()
    old_semantics = support_validation_identity()
    old_plan = writer_budget_plan(
        case.packet, case.aliases, **common, support_validation_revision=REVISION
    )
    old_owner = PlannedBudget(old_plan)
    session_args = (
        tmp_path / "repair.json",
        "instruction-revision",
        case.packet,
        case.aliases,
    )
    session_kwargs = dict(
        provider_api="responses",
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )
    session = WriterSession(
        SimpleNamespace(_cost_budget=old_owner), *session_args, **session_kwargs
    )
    session.save()
    old_journal = session.journal_path.read_bytes()
    old_checkpoint = old_owner.snapshot()

    # Change the consumer's actual text, rather than replacing its digest helper.
    monkeypatch.setattr(
        scoped, instruction, getattr(scoped, instruction) + "\nClarify cited support.\n"
    )
    assert support_validation_identity() != old_semantics
    assert selected.fingerprint() != old_fingerprint
    new_plan = writer_budget_plan(
        case.packet, case.aliases, **common, support_validation_revision=REVISION
    )
    assert new_plan["identity"] != old_plan["identity"]
    assert new_plan["envelope"]["identity"] != old_plan["envelope"]["identity"]
    assert (
        new_plan["envelope"]["support_validation_identity"]
        == support_validation_identity()
    )
    assert new_plan["maximum_tokens"] == old_plan["maximum_tokens"] == 4500000
    assert new_plan["maximum_requests"] == old_plan["maximum_requests"] == 64
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(old_plan["envelope"])
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(old_plan, new_plan)
    new_owner = PlannedBudget(new_plan)
    current_checkpoint = new_owner.snapshot()
    with pytest.raises(ValueError, match="planned_budget_identity_conflict"):
        new_owner.restore(old_checkpoint)
    assert new_owner.snapshot() == current_checkpoint
    with pytest.raises(WriterError, match="writer_checkpoint_identity_conflict"):
        WriterSession(
            SimpleNamespace(_cost_budget=new_owner), *session_args, **session_kwargs
        )
    assert session.journal_path.read_bytes() == old_journal

    for policy, (serialized, fingerprint, old_plan, old_wire) in historical.items():
        for options in (
            options_for(policy),
            options_for(policy, support_validation_revision=None),
        ):
            assert options.model_dump_json().encode() == serialized
            assert options.fingerprint() == fingerprint
        plan = writer_budget_plan(
            case.packet, case.aliases, budget_policy=policy, methodology={}
        )
        assert plan == old_plan
        assert request_bytes(plan) == old_wire


@pytest.mark.parametrize(
    "change",
    [
        {"support_validation_revision": None},
        {"support_validation_identity": None},
        {"support_validation_identity": "0" * 64},
        {"classification_review_contract": None},
        {"inventory_recovery_contract": "bounded-inventory-recovery-v1"},
        {"repair_feedback_contract": None},
        {"transport_revision": None},
    ],
)
def test_rehashed_envelope_cannot_hide_missing_semantic_ancestors(change):
    scope = selected_envelope()
    scope.update(change)
    scope["identity"] = _digest({k: v for k, v in scope.items() if k != "identity"})
    with pytest.raises(ValueError):
        validate_writer_envelope(scope)


@pytest.mark.parametrize(
    "change",
    [
        {"support_validation_revision": "unknown"},
        {"round_budget": False},
        {"capacity": None},
        {"inventory_recovery": False},
        {"classification_review": False},
        {"transport_revision": None},
    ],
)
def test_builder_rejects_invalid_selector_before_constructing_materials(change):
    kwargs = {"support_validation_revision": REVISION, **change}
    with pytest.raises(ValueError, match="support_validation_policy_invalid"):
        parent_envelope(**kwargs)


@pytest.mark.parametrize("selected", [False, True])
def test_runner_forwards_only_the_explicit_semantic_selector(selected):
    graph = TaskGraph()
    graph.add_task(SubTask("write", "Write frozen evidence", "synthesis"))
    options = parent_options(support_validation_revision=REVISION if selected else None)
    LitAgent._apply_run_options_to_graph(graph, options)
    assert (
        "support_validation_revision" in graph.tasks["write"].input_data
    ) is selected
    if selected:
        assert (
            graph.tasks["write"].input_data["support_validation_revision"] == REVISION
        )


@pytest.mark.parametrize(
    "change",
    [
        {"budget_policy": "planned-budget-v6"},
        {"inventory_recovery": False},
        {"classification_review": False},
        {"mixed_inventory_recovery": False},
        {"transport_revision": None},
    ],
)
def test_frozen_plan_rejects_bad_scope_before_material_read(change):
    kwargs = {
        "budget_policy": "planned-budget-v7",
        **FLAGS,
        "support_validation_revision": REVISION,
        **change,
    }
    with pytest.raises(ValueError, match="support_validation_policy_invalid"):
        plan_frozen_budget({}, **kwargs)


@pytest.mark.asyncio
async def test_frozen_run_rejects_invalid_scope_before_wiring():
    with pytest.raises(ValueError, match="support_validation_policy_invalid"):
        await run_frozen(
            SimpleNamespace(_wired=False),
            {},
            "not-sent",
            support_validation_revision=REVISION,
        )


def test_real_budget_wire_and_checkpoint_share_the_explicit_revision(case, tmp_path):
    common = {"budget_policy": "planned-budget-v7", "methodology": {}, **FLAGS}
    old = writer_budget_plan(case.packet, case.aliases, **common)
    assert (
        writer_budget_plan(
            case.packet, case.aliases, **common, support_validation_revision=None
        )
        == old
    )
    selected = writer_budget_plan(
        case.packet, case.aliases, **common, support_validation_revision=REVISION
    )
    assert selected["status"] == "admitted" and selected["repair_limit"] == 2
    assert selected["maximum_tokens"] == 4500000 and selected["maximum_requests"] == 64
    assert selected["identity"] != old["identity"]
    assert (
        selected["envelope"]["support_validation_identity"]
        == support_validation_identity()
    )
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(old, selected)
    owner = PlannedBudget(selected)
    saved = owner.snapshot()
    with pytest.raises(ValueError, match="identity_conflict"):
        owner.restore(PlannedBudget(old).snapshot())
    assert owner.snapshot() == saved
    system, payload, schema = chapter_request_input(
        case.packet,
        case.aliases,
        "introduction",
        {},
        {},
        RANGE_WRITING_PLAN,
        selected["envelope"],
    )
    system, payload = prepare_planned_input(
        system,
        payload,
        case.packet,
        case.aliases,
        RANGE_WRITING_PLAN,
        selected["envelope"],
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    wire = json.loads(messages[1]["content"])
    assert wire["writing_envelope"]["support_validation_revision"] == REVISION
    assert "support_validation_identity" not in wire["writing_envelope"]
    assert messages[0]["content"].count(WRITER_SUPPORT_RECEIPT_INSTRUCTION) == 1
    llm = SimpleNamespace(_cost_budget=owner)
    session_args = (
        llm,
        tmp_path / "repair.json",
        "new-scope",
        case.packet,
        case.aliases,
    )
    session_kw = dict(
        provider_api="responses",
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )
    session = WriterSession(*session_args, **session_kw)
    assert support_validation_fields(session.envelope) == {
        "support_validation_revision": REVISION,
        "support_validation_identity": support_validation_identity(),
    }
    session.save()
    journal = json.loads(session.journal_path.read_text())
    journal.pop("support_validation_revision")
    session.journal_path.write_text(json.dumps(journal))
    with pytest.raises(
        WriterError, match="writer_checkpoint_support_validation_conflict"
    ):
        WriterSession(*session_args, **session_kw)


def manifest_fixture(**changes):
    """Use the complete server manifest selector view, never a public request."""
    return {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        "support_validation_revision": REVISION,
        **changes,
    }


def test_manifest_helper_requires_parents_and_omits_absent_selection():
    old = manifest_fixture()
    old.pop("support_validation_revision")
    assert campaign.support_validation_options(old) == {}
    assert "support_validation_revision" not in campaign.manifest_options(old)
    selected = manifest_fixture()
    assert campaign.support_validation_options(selected) == {
        "support_validation_revision": REVISION
    }
    assert (
        campaign.manifest_options(selected)["support_validation_revision"] == REVISION
    )
    for change in (
        {"support_validation_revision": None},
        {"support_validation_revision": False},
        {"classification_review": False},
        {"inventory_recovery": False},
        {"transport_revision": None},
    ):
        with pytest.raises(
            ValueError, match="frozen_support_validation_configuration_invalid"
        ):
            campaign.support_validation_options(manifest_fixture(**change))


def test_live_cli_cannot_override_frozen_validation_rules(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frozen",
            "--root",
            str(tmp_path),
            "--live",
            "--support-validation-revision",
            REVISION,
        ],
    )
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("Reached live configuration")
    )
    with pytest.raises(SystemExit) as caught:
        campaign.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("selected", [False, True])
def test_prepare_cli_forwards_only_selected_revision(monkeypatch, tmp_path, selected):
    calls = []
    monkeypatch.setattr(campaign, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(campaign, "profile_config", lambda config, profile: config)
    monkeypatch.setattr(campaign, "prepare", lambda *a, **kw: calls.append(kw) or {})
    argv = ["frozen", "--root", str(tmp_path), "--prepare"]
    if selected:
        argv += ["--support-validation-revision", REVISION]
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    assert ("support_validation_revision" in calls[0]) is selected


@pytest.mark.parametrize("selected", [False, True])
def test_real_prepare_validate_and_plan_rebuilds_keep_scope(
    case, tmp_path, monkeypatch, selected
):
    monkeypatch.chdir(tmp_path)
    archive_root = tmp_path / "artifacts/runs"
    archive_root.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        (archive_root / (run_id + ".json")).write_text(json.dumps({"source": run_id}))
    history = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    bundle = {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "methodology": {},
        "extractions": [],
    }
    monkeypatch.setattr(
        campaign,
        "read_bundle",
        lambda path: {
            **copy.deepcopy(bundle),
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(campaign, "source_paths", lambda: [])
    monkeypatch.setattr(campaign, "check_profile", lambda *a, **kw: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    cfg = AppConfig(agent={}, logging={})
    cfg.adversarial.thinking_mode = "disabled"
    root = tmp_path / "offline-support-identity"
    selector = {"support_validation_revision": REVISION} if selected else {}
    campaign.prepare(
        root,
        cfg,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        **FLAGS,
        **selector,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert campaign.validate(root, cfg) == manifest
    for container in (manifest, manifest["options"], manifest["writing_parameters"]):
        assert ("support_validation_revision" in container) is selected
    for row in manifest["cases"]:
        plan = manifest["budget_plans"][row["case_id"]]
        assert ("support_validation_revision" in plan["envelope"]) is selected
        assert plan == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=cfg.capacity,
            **FLAGS,
            **selector,
        )
    assert not (root / "state.json").exists()
    if selected:
        complete_manifest = copy.deepcopy(manifest)
        manifest["writing_parameters"].pop("support_validation_revision")
        manifest["identity"] = campaign.digest(
            {k: v for k, v in manifest.items() if k != "identity"}
        )
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="frozen_writing_parameters_conflict"):
            campaign.validate(root, cfg)
        for container in (
            complete_manifest,
            complete_manifest["options"],
            complete_manifest["writing_parameters"],
        ):
            container.pop("support_validation_revision")
        complete_manifest["identity"] = campaign.digest(
            {k: v for k, v in complete_manifest.items() if k != "identity"}
        )
        (root / "manifest.json").write_text(json.dumps(complete_manifest))
        with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
            campaign.validate(root, cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
async def test_actual_writer_reviewer_and_repair_guidance_is_explicit_only(
    case, tmp_path, monkeypatch, selected
):
    """Inspect actual SDK messages through both shared repairs and full evaluation."""
    from test_repair_feedback import run_feedback_case
    from test_support_receipts_pipeline import select_frozen

    import litagent.writing_v2 as writer

    feedback_checked = []
    if selected:
        select_frozen(monkeypatch)
        original_feedback = writer.repair_feedback_directory

        def checked_feedback(*args):
            result = original_feedback(*args)
            prior = args[4]
            for field in ("support_validation_revision", "support_validation_identity"):
                assert field in prior["repair_receipt"]
                changed = copy.deepcopy(list(args))
                changed[4]["repair_receipt"].pop(field)
                with pytest.raises(
                    WriterError, match="writer_feedback_identity_invalid"
                ):
                    original_feedback(*changed)
            feedback_checked.append(True)
            return result

        monkeypatch.setattr(writer, "repair_feedback_directory", checked_feedback)
    report, gateway = await run_feedback_case(case, tmp_path, monkeypatch, 1)
    assert report["delivery"]["publishable"]
    assert gateway.repair_count == 2 and gateway.inventories == 3
    writing = [row for row in gateway.sent if "writing_envelope" in row["payload"]]
    assert any("requested_body_section" in row["payload"] for row in writing)
    assert any("body_units" in row["payload"] for row in writing)
    assert sum("targets" in row["payload"] for row in writing) == 2
    assert any("units" in row["payload"] for row in writing)
    for row in writing:
        messages = row["request"].get("input", row["request"].get("messages"))
        assert messages[0]["content"].count(WRITER_SUPPORT_RECEIPT_INSTRUCTION) == int(
            selected
        )
        assert (
            "support_validation_revision" in row["payload"]["writing_envelope"]
        ) is selected
    assert bool(feedback_checked) is selected
