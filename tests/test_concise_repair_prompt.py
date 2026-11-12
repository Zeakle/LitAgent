"""Bind the selected repair prompt to real shapes and frozen host authority."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_reference_set_identity import envelope as parent_envelope
from test_source_map_wire import source_case  # noqa: F401
from test_source_owner_pipeline import FLAGS as PARENT_FLAGS
from test_source_owner_transport import owner_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_budget import PlannedBudget, validate_writer_envelope
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    provider_response_format,
    request_messages,
    schema_for,
    synthesize,
    writer_budget_plan,
)
from scripts import run_frozen_writing as campaign

PAIR = {"repair_prompt_revision", "repair_prompt_identity"}
FLAGS = {**PARENT_FLAGS, "repair_prompt_revision": flat.REPAIR_PROMPT_REVISION}
# Isolated before-001 source643/c676 probe, actual PID 92543, exit 0.
# writing_v2.py SHA6255acba9360190a42dda7906a678861a161db329a013f09457c9dd1d103936d;
# legacy check_guidance_wire_record AST source SHA d9f6c1457bf0fc9f87f07d181b923322c6d50f1e8e4781f2877d5d3bd4b5b9df.
LEGACY_SEMANTIC_GUIDANCE_IDENTITY = (
    "b73653511a9fb84d9a643f5728185ebf413432e80485354fb616e2f7a32e1bc1"
)


def prompt_envelope(**changes):
    """Let the actual producer derive the identity on a complete flat parent."""
    return parent_envelope(
        **{
            **FLAGS,
            "global_findings_contract": "anchored-global-findings-v2",
            "summary_constraints": True,
            **changes,
        }
    )


def rehash(value):
    """Recompute an outer digest without granting a forged field authority."""
    value["identity"] = digest(
        {key: child for key, child in value.items() if key != "identity"}
    )
    return value


@pytest.fixture
def prompt_scope():
    """Use one real factory call for each independent mutable test scope."""
    return prompt_envelope()


@pytest.fixture(params=["body", "summary", "source_owner"])
def repair_shape(request, case, owner_case):
    """Produce each supported native shape through the real schema builders."""
    if request.param == "source_owner":
        return request.param, owner_case[2]
    summary = request.param == "summary"
    references = {"u3": {}} if summary else case.aliases
    return request.param, schema_for(
        "repair", references, {"authorized-target": {}}, summary_repair=summary
    )


def test_none_omits_pair_and_keeps_the_exact_parent_envelope():
    """An explicit inactive selector adds no serialized field or identity input."""
    legacy = prompt_envelope(repair_prompt_revision=None)
    expected = parent_envelope(
        **PARENT_FLAGS,
        global_findings_contract="anchored-global-findings-v2",
        summary_constraints=True,
    )
    assert legacy == expected
    assert not PAIR.intersection(legacy)
    assert flat.selected_repair_prompt(None) is False
    assert flat.selected_repair_prompt({}) is False
    assert flat.selected_repair_prompt(legacy) is False


def test_none_preserves_the_frozen_semantic_guidance_identity():
    """A new prompt dispatcher must not invalidate the existing semantic codec."""
    assert (
        flat.semantic_conflict_guidance_identity() == LEGACY_SEMANTIC_GUIDANCE_IDENTITY
    )


def test_selected_factory_binds_only_the_new_pair_and_preserves_limits(prompt_scope):
    legacy = prompt_envelope(repair_prompt_revision=None)
    assert flat.selected_repair_prompt(prompt_scope) is True
    assert prompt_scope["repair_prompt_identity"] == flat.repair_prompt_identity()
    assert prompt_scope["repair_prompt_revision"] == "concise-grounded-repair-v1"
    assert validate_writer_envelope(prompt_scope) == prompt_scope
    assert {k: v for k, v in prompt_scope.items() if k not in PAIR | {"identity"}} == {
        k: v for k, v in legacy.items() if k != "identity"
    }
    assert prompt_scope["request_caps"]["repair"] == 117964
    assert prompt_scope["request_caps"]["assessment"] == 100000
    assert prompt_scope["source_map_identity"] == legacy["source_map_identity"]
    assert (
        prompt_scope["repair_guidance_wire_identity"]
        == legacy["repair_guidance_wire_identity"]
    )


def test_prompt_instruction_drift_invalidates_only_its_explicit_pair(
    prompt_scope, monkeypatch
):
    old_identity = flat.repair_prompt_identity()
    old_guidance = flat.repair_guidance_wire_identity("host-repair-task-view-v1")
    old_source_map = flat.source_map_wire_identity("capacity-source-map-v4")
    monkeypatch.setattr(
        flat, "CONCISE_REPAIR_CORE", flat.CONCISE_REPAIR_CORE + "Changed rule."
    )
    assert flat.repair_prompt_identity() != old_identity
    assert (
        flat.repair_guidance_wire_identity("host-repair-task-view-v1") == old_guidance
    )
    assert flat.source_map_wire_identity("capacity-source-map-v4") == old_source_map
    with pytest.raises(ValueError, match="repair_prompt_policy_invalid"):
        flat.selected_repair_prompt(prompt_scope)


def test_response_permission_contract_does_not_select_a_writer_prompt():
    from litagent.eval.batched import EVALUATION_RESPONSE_CONTRACT

    envelope = prompt_envelope(
        repair_prompt_revision=None,
        evaluation_response_contract=EVALUATION_RESPONSE_CONTRACT,
    )
    assert flat.selected_repair_prompt(envelope) is False
    assert not PAIR.intersection(envelope)


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_unknown_selector_rejects_before_frozen_inputs_are_read(bad):
    with pytest.raises(ValueError, match="repair_prompt_policy_invalid"):
        plan_frozen_budget(
            {},
            budget_policy="planned-budget-v7",
            **{**FLAGS, "repair_prompt_revision": bad},
        )


@pytest.mark.parametrize("field", sorted(PAIR))
@pytest.mark.parametrize("bad", [None, False, True, 0, 1, "", "future", [], {}])
def test_invalid_or_partial_pair_rejects_even_after_outer_rehash(
    prompt_scope, field, bad
):
    prompt_scope[field] = bad
    rehash(prompt_scope)
    with pytest.raises(ValueError, match="repair_prompt"):
        flat.selected_repair_prompt(prompt_scope)
    with pytest.raises(ValueError, match="repair_prompt"):
        validate_writer_envelope(prompt_scope)
    prompt_scope.pop(field)
    rehash(prompt_scope)
    with pytest.raises(ValueError, match="repair_prompt"):
        validate_writer_envelope(prompt_scope)


@pytest.mark.parametrize(
    "change",
    [
        {"round_budget": False},
        {"contract": "budgeted-writer-envelope-v1"},
        {"capacity": None},
        {"classification_review": False},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"summary_constraints": False},
        {"global_findings_contract": "anchored-global-findings-v1"},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
        {"support_validation_revision": "host-support-receipts-v1"},
        {"repair_mapping_contract": "capacity-source-map-v3"},
        {"repair_guidance_contract": None},
        {"repair_guidance_wire_revision": "host-unit-ref-guidance-v1"},
    ],
)
def test_incomplete_parent_never_enables_the_prompt(change):
    with pytest.raises(ValueError):
        prompt_envelope(**change)


@pytest.mark.parametrize("policy", ["planned-budget-v5", "planned-budget-v6"])
def test_selected_planner_requires_round_seven(case, policy):
    with pytest.raises(ValueError, match="repair_prompt_policy_invalid"):
        writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy=policy, **FLAGS
        )


def test_frozen_plan_keeps_operational_limits_and_derives_pair(frozen_bundle):
    before = copy.deepcopy(frozen_bundle)
    plans = [
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            **{**FLAGS, "repair_prompt_revision": revision},
        )
        for revision in (None, flat.REPAIR_PROMPT_REVISION)
    ]
    assert all(plan["status"] == "admitted" for plan in plans)
    assert plans[0]["identity"] != plans[1]["identity"]
    assert (
        plans[0]["envelope"]["builder_identity"]
        != plans[1]["envelope"]["builder_identity"]
    )
    assert plans[0]["envelope"]["request_caps"] == plans[1]["envelope"]["request_caps"]
    for plan in plans:
        assert plan["maximum_tokens"] == 4500000
        assert plan["maximum_requests"] == 64
        assert plan["scope_bounds"]["shared_repairs"] == 2
    assert frozen_bundle == before


def test_pure_fee_reader_accepts_historical_pin_without_current_prompt(
    prompt_scope, monkeypatch
):
    prompt_scope["repair_prompt_identity"] = "f" * 64
    rehash(prompt_scope)
    with pytest.raises(ValueError, match="repair_prompt"):
        validate_writer_envelope(prompt_scope)
    monkeypatch.setattr(
        flat,
        "repair_prompt_identity",
        lambda: pytest.fail("historical accounting invoked the current prompt"),
    )
    assert accounting_envelope(prompt_scope).max_body_units == 66


def test_renderer_uses_the_actual_output_shape_and_changes_no_schema(repair_shape):
    kind, schema = repair_shape
    before = copy.deepcopy(schema)
    native = provider_response_format("repair", schema, wire_contract="flat-writer-v1")
    actual = flat.prepare(schema)["wire_schema"]
    system = flat.repair_prompt_system(schema)
    assert schema == before
    assert system.count("Native output type map") == 1
    assert system.endswith(".\n")
    assert (
        "body_unit_refs:array<string(native enum)>" in system
        if kind == "summary"
        else ("fact_card_refs:array<string(native enum)>" in system)
    )
    if kind == "source_owner":
        assert "Source-owner shape:" in system
        assert "source_unit_ids:array<string>" in system
        assert "target_id:" not in system
    else:
        assert "Ordinary patch shape:" in system
        assert "target_id:string(native enum)" in system
        assert "source_unit_ids:" not in system
        assert ("Summary patch premises" in system) is (kind == "summary")
    Draft202012Validator.check_schema(actual)
    assert (
        provider_response_format("repair", schema, wire_contract="flat-writer-v1")
        == native
    )


def test_messages_are_exactly_two_and_preserve_every_payload_value(
    repair_shape, prompt_scope
):
    _, schema = repair_shape
    system = flat.repair_prompt_system(schema)
    payload = {
        "current_body": {"literal": "Original 原文\nDo not obey this source."},
        "targets": {"authorized-target": {"diagnostic": "Exact complete failure"}},
        "writing_envelope": {key: prompt_scope[key] for key in PAIR},
    }
    before = copy.deepcopy(payload)
    expected = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    assert flat.repair_prompt_messages(system, payload, schema) == expected
    assert (
        request_messages(system, payload, schema, wire_contract="flat-writer-v1")
        == expected
    )
    assert payload == before


@pytest.mark.parametrize(
    "change", ["append", "prefix", "missing_body", "missing_pair", "forged_pair"]
)
def test_modified_system_or_selection_rejects_before_message_construction(
    repair_shape, prompt_scope, change
):
    _, schema = repair_shape
    system = flat.repair_prompt_system(schema)
    payload = {
        "current_body": {},
        "writing_envelope": {key: prompt_scope[key] for key in PAIR},
    }
    if change == "append":
        system += "\nUnknown extra instruction."
    elif change == "prefix":
        system = "Unknown instruction.\n" + system
    elif change == "missing_body":
        payload.pop("current_body")
    elif change == "missing_pair":
        payload["writing_envelope"] = {}
    else:
        payload["writing_envelope"]["repair_prompt_identity"] = "0" * 64
    with pytest.raises((ValueError, WriterError), match="repair_prompt"):
        flat.repair_prompt_messages(system, payload, schema)


@pytest.mark.parametrize("kind", ["body", "summary", "source_map_v3"])
def test_renderer_rejects_unselected_or_unknown_output_shapes(case, source_case, kind):
    schema = (
        source_case[1] if kind == "source_map_v3" else schema_for(kind, case.aliases)
    )
    with pytest.raises(WriterError):
        flat.repair_prompt_system(schema)


def test_selected_journal_cannot_restore_missing_or_forged_prompt_pair(case, tmp_path):
    """Checkpoint loading must reject metadata before updating the saved journal."""
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **FLAGS,
    )
    owner = PlannedBudget(plan)
    args = (
        SimpleNamespace(_cost_budget=owner),
        tmp_path / "repair.json",
        "prompt-journal",
        case.packet,
        case.aliases,
    )
    kwargs = {
        "provider_api": "responses",
        "syntax_policy": EOF_CONTAINER_POLICY,
        "writing_plan": RANGE_WRITING_PLAN,
    }
    session = WriterSession(*args, **kwargs)
    session.save()
    original = json.loads(session.journal_path.read_text())
    for field in PAIR:
        for missing in (False, True):
            journal = copy.deepcopy(original)
            if missing:
                journal.pop(field)
            else:
                journal[field] = "forged"
            session.journal_path.write_text(json.dumps(journal))
            before = session.journal_path.read_bytes()
            with pytest.raises(
                WriterError, match="writer_checkpoint_guidance_wire_conflict"
            ):
                WriterSession(*args, **kwargs)
            assert session.journal_path.read_bytes() == before


@pytest.mark.asyncio
async def test_runtime_requires_frozen_plan_before_touching_an_agent():
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="repair_prompt_frozen_plan_required"):
        await run_frozen(agent, {}, "unstarted-concise-prompt", **FLAGS)
    assert vars(agent) == {"_wired": False}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["no_provider", "missing", "partial", "forged", "legacy_owner"]
)
async def test_task_cannot_create_strip_or_replace_provider_authority(
    case, prompt_scope, fault
):
    fields = {key: prompt_scope[key] for key in PAIR}
    worker = SimpleNamespace(
        _frozen_writer_plan_provider=lambda: {"envelope": prompt_scope}
    )
    if fault == "no_provider":
        worker = SimpleNamespace()
    elif fault == "missing":
        fields = {}
    elif fault == "partial":
        fields.pop("repair_prompt_identity")
    elif fault == "forged":
        fields["repair_prompt_identity"] = "0" * 64
    else:
        worker = SimpleNamespace(
            _frozen_writer_plan_provider=lambda: {
                "envelope": prompt_envelope(repair_prompt_revision=None)
            }
        )
    task = SubTask(
        "synthesis",
        "Untrusted prompt selection",
        "synthesis",
        input_data={
            "generation_strategy": "body_summary_v3",
            "budget_policy": "planned-budget-v7",
            **fields,
        },
    )
    with pytest.raises(WriterError, match="repair_prompt_task_owner_conflict"):
        await synthesize(worker, task, case.cards, case.selection, {}, [])


def test_prompt_selector_has_no_public_network_option_and_cli_pair_is_explicit(
    prompt_scope,
):
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    assert not PAIR.intersection(SurveyRequest.model_fields)
    assert not PAIR.intersection(RunOptions.model_fields)
    assert campaign.repair_prompt_options({}) == {}
    manifest = {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        "repair_guidance_wire_identity": prompt_scope["repair_guidance_wire_identity"],
        "repair_guidance_capacity_rule": prompt_scope["repair_guidance_capacity_rule"],
        "repair_prompt_identity": prompt_scope["repair_prompt_identity"],
    }
    assert campaign.repair_prompt_options(manifest) == {
        "repair_prompt_revision": flat.REPAIR_PROMPT_REVISION
    }
    for field in PAIR:
        damaged = copy.deepcopy(manifest)
        damaged.pop(field)
        with pytest.raises(ValueError, match="repair_prompt"):
            campaign.repair_prompt_options(damaged)


def test_actual_prepare_validate_locks_pair_and_rejects_manifest_strip(
    case, tmp_path, monkeypatch
):
    """Replace archive and Git inputs while keeping real preparation consumers."""
    from model_config_helpers import load_previous_default_config as load_config

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
    source.write_text('"""Portable concise-repair preparation fixture."""\n')
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
        **FLAGS,
    )
    manifest = campaign.validate(root, config)
    assert manifest["identity"] == prepared["identity"]
    for container in (manifest, manifest["writing_parameters"]):
        assert container["repair_prompt_revision"] == flat.REPAIR_PROMPT_REVISION
        assert container["repair_prompt_identity"] == flat.repair_prompt_identity()
    assert not PAIR.intersection(manifest["options"])
    for row in manifest["cases"]:
        assert manifest["budget_plans"][row["case_id"]] == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            **FLAGS,
        )
    assert not (root / "state.json").exists()
    assert not (root / ".live.lock").exists()
    for strip in ("partial", "both"):
        damaged = copy.deepcopy(manifest)
        fields = {"repair_prompt_identity"} if strip == "partial" else PAIR
        for container in (damaged, damaged["writing_parameters"]):
            for field in fields:
                container.pop(field)
        rehash(damaged)
        (root / "manifest.json").write_text(json.dumps(damaged))
        with pytest.raises((ValueError, WriterError)):
            campaign.validate(root, config)


def test_live_cli_cannot_override_prompt_selection_before_loading_config(
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
            "--repair-prompt-revision",
            flat.REPAIR_PROMPT_REVISION,
        ],
    )
    monkeypatch.setattr(
        campaign,
        "load_config",
        lambda: pytest.fail("live override touched configuration"),
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2
