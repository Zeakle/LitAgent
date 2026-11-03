"""Bind optional evaluation dimensions without migrating historical authority."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
from test_concise_repair_prompt import FLAGS as PARENT_FLAGS
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_reference_set_identity import envelope as parent_envelope
from test_writer_v2 import case  # noqa: F401

from litagent.eval import dimensions
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_accounting import _plan as accounting_plan
from litagent.safety.planned_budget import PlannedBudget, validate_writer_envelope
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    synthesize,
    writer_budget_plan,
)
from scripts import run_frozen_writing as campaign

CONTRACT = "separated-evaluation-dimensions-v1"
PAIR = ("evaluation_dimensions_contract", "evaluation_dimensions_identity")
FLAGS = {**PARENT_FLAGS, "evaluation_response_contract": "host-bound-evaluation-v2"}


def rehash(value):
    """Rehash supplied data without granting frozen host ownership."""
    value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
    return value


def dimension_envelope(**changes):
    """Produce the complete selected envelope through the actual constructor."""
    return parent_envelope(
        **{
            **FLAGS,
            "global_findings_contract": "anchored-global-findings-v2",
            "summary_constraints": True,
            "evaluation_dimensions_contract": CONTRACT,
            **changes,
        }
    )


def dimension_plan(bundle, **changes):
    """Construct frozen plans without clients, owners or model requests."""
    return plan_frozen_budget(
        bundle,
        **{
            "budget_policy": "planned-budget-v7",
            **FLAGS,
            "evaluation_dimensions_contract": CONTRACT,
            **changes,
        },
    )


def dimension_manifest():
    """Bind the private manifest selector to its complete response parent pair."""
    scope = dimension_envelope()
    return {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        **{key: scope[key] for key in (*PAIR, "evaluation_response_identity")},
    }


def test_dimensions_are_independent_of_writer_source_view(frozen_bundle):
    old = dimension_plan(frozen_bundle, evaluation_dimensions_contract=None)
    implicit = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **FLAGS
    )
    assert implicit == old and not set(PAIR).intersection(old["envelope"])
    assert dimensions.dimensions_fields(old["envelope"]) == {}
    selected = dimension_plan(frozen_bundle)
    assert selected["status"] == "admitted"
    assert selected["identity"] != old["identity"]
    assert selected["envelope"]["source_identity"] == old["envelope"]["source_identity"]
    assert selected["envelope"][PAIR[1]] == dimensions.dimensions_identity()
    assert "writer_source_view" not in selected["envelope"]
    assert selected["maximum_tokens"] == 4500000
    assert selected["maximum_requests"] == 64
    assert selected["repair_limit"] == selected["scope_bounds"]["shared_repairs"] == 2
    assert selected["context_reservation_limit"] == 117964
    accounting_plan(selected)
    combined = dimension_plan(frozen_bundle, writer_source_view="source-only-writer-v1")
    assert dimensions.selected_dimensions(combined["envelope"])
    assert combined["envelope"]["writer_source_view"] == "source-only-writer-v1"
    assert combined["identity"] not in (old["identity"], selected["identity"])


# Independently evaluated from copied source657 before-001 (2026-11-03,
# subprocess exit 0); these pins keep default tests independent of artifacts/.
LEGACY_RESPONSE_IDENTITIES = {
    "host-bound-evaluation-v1": "686e24fa9b9016c0174789d898dc88dad9bb6de6790abd72e77ba9402ed0c2b5",
    "host-bound-evaluation-v2": "13c9b9dd7a8b4dde54e63f2cc61d1c5d8506f516b0cd33da7ccd8b7a9cf9ad22",
}


@pytest.mark.parametrize(
    "response", [None, "host-bound-evaluation-v1", "host-bound-evaluation-v2"]
)
def test_inactive_dimensions_keep_complete_plans_and_legacy_ids(
    frozen_bundle, response
):
    """Keep absence identical to explicit None on all three old response paths."""
    from litagent.eval.batched import response_contract_identity

    flags = {**FLAGS, "evaluation_response_contract": response}
    implicit = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **flags
    )
    explicit = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        evaluation_dimensions_contract=None,
        **flags,
    )
    assert implicit == explicit
    assert not set(PAIR).intersection(implicit["envelope"])
    assert dimensions.dimensions_fields(implicit["envelope"]) == {}
    if response is not None:
        assert (
            implicit["envelope"]["evaluation_response_identity"]
            == LEGACY_RESPONSE_IDENTITIES[response]
        )
        assert (
            response_contract_identity(contract=response)
            == LEGACY_RESPONSE_IDENTITIES[response]
        )


@pytest.mark.parametrize("value", [False, True, 0, 1, "", "future", [], {}])
def test_invalid_dimensions_selector_rejects_before_bundle_access(value):
    with pytest.raises(ValueError, match="evaluation_dimensions"):
        dimension_plan({}, evaluation_dimensions_contract=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_policy": "planned-budget-v6"},
        {"support_validation_revision": "host-support-receipts-v1"},
        {"evaluation_response_contract": None},
        {"evaluation_response_contract": "host-bound-evaluation-v1"},
        {"mixed_inventory_recovery": False},
        {"classification_review": False},
    ],
)
def test_dimensions_do_not_infer_the_required_parent_stack(frozen_bundle, changes):
    with pytest.raises(ValueError):
        dimension_plan(frozen_bundle, **changes)


@pytest.mark.parametrize("field", PAIR)
@pytest.mark.parametrize("value", [None, "missing", "forged"])
def test_half_or_forged_envelope_pair_rejects_after_rehash(field, value):
    scope = dimension_envelope()
    if value == "missing":
        scope.pop(field)
    else:
        scope[field] = value
    rehash(scope)
    before = copy.deepcopy(scope)
    with pytest.raises(ValueError):
        validate_writer_envelope(scope)
    with pytest.raises(ValueError):
        accounting_envelope(scope)
    assert scope == before


def test_historical_reader_uses_pinned_pair_without_current_rules(monkeypatch):
    scope = dimension_envelope()
    scope[PAIR[1]] = "f" * 64
    rehash(scope)
    with pytest.raises(ValueError):
        validate_writer_envelope(scope)
    monkeypatch.setattr(
        dimensions,
        "dimensions_identity",
        lambda: pytest.fail("current identity executed"),
    )
    assert accounting_envelope(scope).max_body_units == 66
    pair = {key: scope[key] for key in PAIR}
    campaign._historical_dimensions_binding(pair, scope)
    campaign._historical_support_binding(
        {
            **pair,
            **{
                key: scope[key]
                for key in (
                    "support_validation_revision",
                    "support_validation_identity",
                    "semantic_conflict_policy",
                    "semantic_conflict_identity",
                )
                if key in scope
            },
        },
        scope,
    )
    for field in PAIR:
        damaged = {**pair, field: "0" * 64}
        with pytest.raises(ValueError, match="evaluation_dimensions_binding_invalid"):
            campaign._historical_dimensions_binding(damaged, scope)
        with pytest.raises(ValueError, match="evaluation_dimensions_binding_invalid"):
            campaign._historical_dimensions_binding({field: None}, {})


def test_manifest_has_no_public_option_and_requires_complete_pair():
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    assert not set(PAIR).intersection(SurveyRequest.model_fields)
    assert not set(PAIR).intersection(RunOptions.model_fields)
    assert campaign.evaluation_dimensions_options({}) == {}
    manifest = dimension_manifest()
    assert campaign.evaluation_dimensions_options(manifest) == {PAIR[0]: CONTRACT}
    for field in (*PAIR, "evaluation_response_identity", "mixed_inventory_recovery"):
        damaged = copy.deepcopy(manifest)
        damaged.pop(field)
        with pytest.raises(ValueError):
            campaign.evaluation_dimensions_options(damaged)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing_plan", "strip_selector", "wrong_identity", "old_plan"]
)
async def test_runtime_requires_frozen_pair_before_touching_agent(frozen_bundle, fault):
    plan = dimension_plan(frozen_bundle)
    selector = CONTRACT
    if fault == "missing_plan":
        plan = None
    elif fault == "strip_selector":
        selector = None
    elif fault == "old_plan":
        plan = dimension_plan(frozen_bundle, evaluation_dimensions_contract=None)
    else:
        plan["envelope"][PAIR[1]] = "0" * 64
        rehash(plan["envelope"])
        rehash(plan)
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="evaluation_dimensions"):
        await run_frozen(
            agent,
            frozen_bundle,
            "unstarted-dimensions",
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            evaluation_dimensions_contract=selector,
            **FLAGS,
        )
    assert vars(agent) == {"_wired": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "partial", "forged", "legacy_owner"])
async def test_task_cannot_strip_or_create_dimensions_authority(case, fault):
    scope = dimension_envelope()
    pair = dimensions.dimensions_fields(scope)
    if fault == "missing":
        pair = {}
    elif fault == "partial":
        pair.pop(PAIR[1])
    elif fault == "forged":
        pair[PAIR[1]] = "0" * 64
    else:
        scope = dimension_envelope(evaluation_dimensions_contract=None)
    worker = SimpleNamespace(_frozen_writer_plan_provider=lambda: {"envelope": scope})
    task = SubTask(
        "synthesis",
        "Untrusted dimensions selection",
        "synthesis",
        input_data={
            "generation_strategy": "body_summary_v3",
            "budget_policy": "planned-budget-v7",
            **pair,
        },
    )
    with pytest.raises(WriterError, match="evaluation_dimensions_task_owner_conflict"):
        await synthesize(worker, task, case.cards, case.selection, {}, [])


@pytest.mark.parametrize("selected", [False, True])
def test_journal_and_each_record_require_exact_presence_on_restore(
    case, tmp_path, selected
):
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **FLAGS,
        **({PAIR[0]: CONTRACT} if selected else {}),
    )
    args = (
        SimpleNamespace(_cost_budget=PlannedBudget(plan)),
        tmp_path / "repair.json",
        "dimensions-journal",
        case.packet,
        case.aliases,
    )
    kwargs = {
        "provider_api": "responses",
        "syntax_policy": EOF_CONTAINER_POLICY,
        "writing_plan": RANGE_WRITING_PLAN,
    }
    session = WriterSession(*args, **kwargs)
    pair = dimensions.dimensions_fields(plan["envelope"])
    session.journal["calls"]["not-sent"] = {"status": "rejected_not_sent", **pair}
    session.save()
    original = json.loads(session.journal_path.read_text())
    assert WriterSession(*args, **kwargs).journal == original
    for target in ("journal", "call"):
        for field in PAIR:
            for missing in (False, True) if selected else (False,):
                journal = copy.deepcopy(original)
                row = journal if target == "journal" else journal["calls"]["not-sent"]
                if missing:
                    row.pop(field)
                else:
                    row[field] = "forged" if selected else None
                session.journal_path.write_text(json.dumps(journal))
                before = session.journal_path.read_bytes()
                with pytest.raises(WriterError, match="evaluation_dimensions_conflict"):
                    WriterSession(*args, **kwargs)
                assert session.journal_path.read_bytes() == before


def test_live_cli_cannot_override_dimensions_before_configuration(
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
            "--evaluation-dimensions-contract",
            CONTRACT,
        ],
    )
    monkeypatch.setattr(campaign, "load_config", lambda: pytest.fail("config reached"))
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2


def test_dimensions_do_not_require_writer_or_repair_view_selectors():
    scope = dimension_envelope(
        repair_prompt_revision=None,
        repair_mapping_contract=None,
        repair_guidance_contract=None,
        repair_guidance_wire_revision=None,
    )
    assert dimensions.selected_dimensions(scope)
    assert not {
        "writer_source_view",
        "repair_prompt_revision",
        "repair_guidance_contract",
        "repair_guidance_wire_revision",
    }.intersection(scope)
    assert scope["repair_mapping_contract"] == "capacity-edit-map-v2"


def test_prepare_and_validate_keep_the_same_private_pair(case, tmp_path, monkeypatch):
    """Exercise real preparation and recomputation without starting a run."""
    from litagent.config import load_config

    monkeypatch.chdir(tmp_path)
    archives = tmp_path / "artifacts/runs"
    archives.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        (archives / (run_id + ".json")).write_text(
            json.dumps(
                {
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
            )
        )
    baseline = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable dimensions preparation fixture."""\n')
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
        evaluation_dimensions_contract=CONTRACT,
        **FLAGS,
    )
    manifest = campaign.validate(root, config)
    assert manifest["identity"] == prepared["identity"]
    for container in (manifest, manifest["writing_parameters"]):
        assert container[PAIR[0]] == CONTRACT
        assert container[PAIR[1]] == dimensions.dimensions_identity()
    for row in manifest["cases"]:
        assert manifest["budget_plans"][row["case_id"]] == dimension_plan(
            row["bundle"], capacity=config.capacity
        )
    assert not set(PAIR).intersection(manifest["options"])
    assert not (root / "state.json").exists()
    for missing in (PAIR, (PAIR[1],)):
        damaged = copy.deepcopy(manifest)
        for container in (damaged, damaged["writing_parameters"]):
            for field in missing:
                container.pop(field)
        rehash(damaged)
        (root / "manifest.json").write_text(json.dumps(damaged))
        with pytest.raises((ValueError, WriterError)):
            campaign.validate(root, config)


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
async def test_repair_replay_uses_funded_dimensions_and_clears_fake_context(
    case, tmp_path, monkeypatch, selected
):
    """Replay actual settled raw through the same owner-selected host validator."""
    import test_report_repair_eligibility as semantic
    import test_structured_repair_pipeline as structured

    import litagent.writing_v2 as writing
    from litagent.eval import scoped
    from litagent.eval.dimensions import dimensions_fields

    semantic.prepare_case(case)
    monkeypatch.setattr(
        structured,
        "FLAGS",
        {**FLAGS, **({PAIR[0]: CONTRACT} if selected else {})},
    )
    original_gateway = semantic.SemanticGateway

    def gateway(*arguments, **options):
        peer = original_gateway(*arguments, **options)
        original_chat = peer.evaluator.chat

        async def chat(messages, **settings):
            response = await original_chat(messages, **settings)
            payload = peer.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            value = json.loads(response.content)
            for index, decision in enumerate(value["decisions"]):
                # The actual V2 native response derives this host-owned field.
                decision.pop("classification_valid", None)
                if selected:
                    value["decisions"][index] = {
                        "claim_id": decision["claim_id"],
                        "report_check": {
                            key: decision[key]
                            for key in (
                                "correct_kind",
                                "atomic",
                                "source_faithful",
                                "explanation",
                            )
                        },
                        "evidence_check": {
                            key: decision[key]
                            for key in (
                                "status",
                                "reason_code",
                                "support_basis",
                                "aspect_checks",
                            )
                        },
                    }
            response.content = json.dumps(value)
            return response

        peer.evaluator.chat = chat
        return peer

    monkeypatch.setattr(semantic, "SemanticGateway", gateway)
    replay = writing._replay_repair_assessment
    validate = scoped.validate_assessment_decisions
    active, observed = [], []

    def checked_replay(runner, **arguments):
        envelope = arguments["report"]["evaluation"]["faithfulness"]["details"][
            "stages"
        ]["writer_envelope"]
        build_context = runner._build_eval_context

        def untrusted_context(**values):
            context = build_context(**values)
            context["_evaluation_dimensions_envelope"] = {
                PAIR[0]: "untrusted",
                PAIR[1]: "0" * 64,
            }
            return context

        active.append(envelope)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(runner, "_build_eval_context", untrusted_context)
                return replay(runner, **arguments)
        finally:
            active.pop()

    def checked_validate(*arguments, **options):
        if active:
            envelope = active[-1]
            actual = options["context"]["_evaluation_dimensions_envelope"]
            assert actual is (envelope if dimensions_fields(envelope) else None)
            observed.append(bool(dimensions_fields(envelope)))
        return validate(*arguments, **options)

    monkeypatch.setattr(writing, "_replay_repair_assessment", checked_replay)
    monkeypatch.setattr(scoped, "validate_assessment_decisions", checked_validate)
    report, peer = await semantic.run_semantic(case, tmp_path, monkeypatch)
    assert observed and set(observed) == {selected}
    assert peer.repair_count == 1
    assert report["delivery"]["publishable"] is True
