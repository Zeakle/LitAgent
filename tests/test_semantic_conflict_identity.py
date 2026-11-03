"""Bind conservative conflict handling to frozen owners and actual settlements."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_adaptive_capacity import settle_initial
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_source_owner_pipeline import FLAGS
from test_structured_repair_identity import budget_plan, guidance_envelope, rehash
from test_unavailable_cost_exports import display_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

import litagent.eval.semantic_conflict as semantic
from litagent.capacity_rounds import plan_round, request_proof
from litagent.config import load_config
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import (
    PlannedBudget,
    request_scope,
    settled_assessment_receipt,
    validate_settled_assessment_receipt,
    validate_writer_envelope,
)
from litagent.writing_v2 import RANGE_WRITING_PLAN
from scripts import run_frozen_writing as campaign

POLICY = semantic.SEMANTIC_CONFLICT_POLICY
FIELDS = {"semantic_conflict_policy", "semantic_conflict_identity"}


def scope(**changes):
    """Build the actual complete current parent with an explicit private policy."""
    return guidance_envelope(**{**FLAGS, "semantic_conflict_policy": POLICY, **changes})


def manifest_selection(**changes):
    """Describe the same private frozen selection without ordinary run options."""
    from litagent.writing_flat import (
        REPAIR_TASK_VIEW_ADMISSION_RULE,
        repair_guidance_wire_identity,
    )

    return {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        "repair_guidance_wire_identity": repair_guidance_wire_identity(
            FLAGS["repair_guidance_wire_revision"]
        ),
        "repair_guidance_capacity_rule": REPAIR_TASK_VIEW_ADMISSION_RULE,
        "semantic_conflict_policy": POLICY,
        "semantic_conflict_identity": semantic.semantic_conflict_identity(),
        **changes,
    }


def test_none_omits_pair_and_preserves_complete_parent_value():
    parent = guidance_envelope(**FLAGS)
    assert scope(semantic_conflict_policy=None) == parent
    assert not FIELDS.intersection(parent)
    selected = scope()
    assert {k: v for k, v in selected.items() if k not in FIELDS | {"identity"}} == {
        k: v for k, v in parent.items() if k != "identity"
    }
    assert selected["identity"] != parent["identity"]
    plan = budget_plan(selected)
    assert plan["maximum_tokens"] == 4500000 and plan["maximum_requests"] == 64
    assert plan["scope_bounds"]["shared_repairs"] == 2


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_private_policy_rejects_coerced_or_unknown_values(bad):
    with pytest.raises(ValueError):
        scope(semantic_conflict_policy=bad)


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_partial_pair_is_rejected_even_after_digest_reseal(field):
    selected = scope()
    selected.pop(field)
    rehash(selected)
    with pytest.raises(ValueError):
        validate_writer_envelope(selected)
    with pytest.raises(ValueError):
        accounting_envelope(selected)


@pytest.mark.parametrize(
    "change",
    [
        {"support_validation_revision": "host-support-receipts-v1"},
        {"repair_mapping_contract": "capacity-source-map-v3"},
        {"repair_guidance_wire_revision": "host-unit-ref-guidance-v1"},
        {"repair_guidance_contract": None},
        {"round_budget": False},
        {"classification_review": False},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
    ],
)
def test_policy_cannot_relax_any_current_parent(change):
    with pytest.raises(ValueError):
        scope(**change)


def test_current_identity_drift_refuses_owner_but_fees_keep_pinned_data(monkeypatch):
    parent = scope(semantic_conflict_policy=None)
    selected = scope()
    plan = budget_plan(selected)
    snapshot = PlannedBudget(plan).snapshot()
    monkeypatch.setattr(semantic, "semantic_conflict_identity", lambda: "e" * 64)
    assert scope(semantic_conflict_policy=None) == parent
    with pytest.raises(ValueError):
        validate_writer_envelope(selected)
    with pytest.raises(ValueError):
        PlannedBudget(plan)
    assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot


def test_manifest_pair_is_private_and_binds_both_fields():
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    selected = manifest_selection()
    assert campaign.semantic_conflict_options(selected) == {
        "semantic_conflict_policy": POLICY
    }
    assert campaign.semantic_conflict_options({}) == {}
    assert not FIELDS.intersection(campaign.manifest_options(selected))
    assert not FIELDS.intersection(RunOptions.model_fields)
    assert not FIELDS.intersection(SurveyRequest.model_fields)
    for key in FIELDS:
        broken = copy.deepcopy(selected)
        broken.pop(key)
        with pytest.raises(ValueError):
            campaign.semantic_conflict_options(broken)
    broken = {**selected, "semantic_conflict_identity": "e" * 64}
    with pytest.raises(ValueError):
        campaign.semantic_conflict_options(broken)


def test_actual_frozen_plan_none_and_selected_are_separate(frozen_bundle):
    parent = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **FLAGS
    )
    assert (
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            **FLAGS,
            semantic_conflict_policy=None,
        )
        == parent
    )
    selected = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        **FLAGS,
        semantic_conflict_policy=POLICY,
    )
    assert selected["envelope"]["semantic_conflict_policy"] == POLICY
    assert parent["identity"] != selected["identity"]
    assert parent["input_identity"] == selected["input_identity"]


@pytest.mark.asyncio
async def test_selected_runtime_requires_frozen_plan_before_client_access():
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="semantic_conflict_frozen_plan_required"):
        await run_frozen(agent, {}, "never-sent", semantic_conflict_policy=POLICY)
    assert agent.__dict__ == {"_wired": False}


@pytest.fixture
def settled():
    """Use actual plan, round reservation, send marker, binding and settlement."""
    owner = PlannedBudget(budget_plan(scope()))
    settle_initial(owner)
    messages = [{"role": "user", "content": "A complete selected assessment."}]
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "offline", "schema": {"type": "object"}},
    }
    requests = [request_proof("assessment", messages, 16384, rf) for _ in range(8)]
    requests.append(
        request_proof("global", [{"role": "user", "content": "global"}], 16384, rf)
    )
    plan_round(
        owner, assertions=128, batches=8, requests=requests, report_sha256="a" * 64
    )
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "usage_available": True}
    with request_scope("assessment", "assessment-0"):
        amount = owner.check_wire(messages, 16384, rf)
        owner.reserve(amount)
        owner.bind_wire(
            [
                messages,
                {
                    "max_tokens": 16384,
                    "response_format": rf,
                    "model": "test",
                    "temperature": 0,
                },
                "chat",
            ]
        )
        owner.begin_request()
        owner.record(usage)
    return owner, messages, rf, usage


def test_receipt_is_read_only_and_replay_uses_original_slot(settled, monkeypatch):
    owner, messages, rf, usage = settled
    before = owner.snapshot()
    monkeypatch.setattr(owner, "check_wire", lambda *a, **k: pytest.fail("new slot"))
    with request_scope("assessment", "assessment-0"):
        receipt = settled_assessment_receipt(owner, messages, 16384, rf, usage)
    assert receipt["round_number"] == 1 and receipt["actual_tokens"] == 120
    assert receipt["input_sha256"] == receipt["round_wire_sha256"]
    assert (
        validate_settled_assessment_receipt(before, receipt, request_id="assessment-0")
        == receipt
    )
    assert owner.snapshot() == before


@pytest.mark.parametrize(
    "fault", ["usage", "unknown_usage", "input", "output", "format", "request", "scope"]
)
def test_response_usage_cannot_replace_real_settlement(settled, fault):
    owner, messages, rf, usage = settled
    kw = {"request_id": "assessment-0"}
    if fault == "usage":
        usage["prompt_tokens"] += 1
    elif fault == "unknown_usage":
        usage["usage_available"] = False
    elif fault == "input":
        messages[0]["content"] += " changed"
    elif fault == "format":
        rf = None
    elif fault == "request":
        kw["request_id"] = "not-the-settled-request"
    elif fault == "scope":
        kw = {}
    with pytest.raises(ValueError):
        settled_assessment_receipt(
            owner, messages, 1 if fault == "output" else 16384, rf, usage, **kw
        )


@pytest.mark.parametrize(
    "fault", ["receipt", "entry", "round", "inflight", "unknown", "numeric_alias"]
)
def test_replay_rejects_tampered_or_unsettled_ledger(settled, fault):
    owner, messages, rf, usage = settled
    receipt = settled_assessment_receipt(
        owner, messages, 16384, rf, usage, request_id="assessment-0"
    )
    snapshot = owner.snapshot()
    if fault == "receipt":
        receipt["actual_tokens"] += 1
    elif fault == "numeric_alias":
        receipt["round_number"] = True
    elif fault == "entry":
        snapshot["entries"]["assessment-0"]["wire_sha256"] = "e" * 64
    elif fault == "round":
        snapshot["entries"]["assessment-0"]["round_number"] += 1
    elif fault == "inflight":
        snapshot["entries"]["assessment-0"]["status"] = "started_no_replay"
    elif fault == "unknown":
        snapshot["usage_uncertain"] = True
    with pytest.raises(ValueError):
        validate_settled_assessment_receipt(
            snapshot, receipt, request_id="assessment-0"
        )


def test_fee_records_cannot_strip_or_add_semantic_pair():
    selected = scope()
    fields = {
        key: selected[key]
        for key in (
            "support_validation_revision",
            "support_validation_identity",
            *sorted(FIELDS),
        )
    }
    campaign._historical_support_binding(fields, selected)
    fields.pop("semantic_conflict_identity")
    with pytest.raises(ValueError, match="semantic_binding_invalid"):
        campaign._historical_support_binding(fields, selected)
    with pytest.raises(ValueError, match="semantic_binding_invalid"):
        campaign._historical_support_binding({"semantic_conflict_policy": POLICY}, {})


@pytest.mark.parametrize(
    "fault", [None, "missing", "wire", "raw", "input", "batch", "numeric_alias"]
)
def test_actual_fee_consumer_authenticates_frozen_settlement(
    settled, fault, monkeypatch
):
    """Fee reads compare the pinned receipt without reopening today's policy code."""
    from litagent.eval.scoped import _request_identity

    owner, messages, rf, usage = settled
    receipt = settled_assessment_receipt(
        owner, messages, 16384, rf, usage, request_id="assessment-0"
    )
    raw = '{"decisions":[]}'
    progress = {
        "phase": "assessment",
        "input_sha256": _request_identity(messages, rf),
        "reserved_tokens": owner.entries["assessment-0"]["reserved_tokens"],
        "status": "response_received",
        "response": raw,
        "usage": usage,
        "semantic_conflict_settlement": {
            "receipt": receipt,
            "report_sha256": "a" * 64,
            "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "input_sha256": _request_identity(messages, rf),
            "batch_payload_sha256": "b" * 64,
        },
    }
    record = {
        "identity": "c" * 64,
        "checkpoint": {
            "report_sha256": "a" * 64,
            "progress": {
                "requests": [progress],
                "batches": [{"input_sha256": "b" * 64}],
            },
        },
    }
    manifest = {
        "config": {
            "effective": {
                "eval": {"max_tokens": 16384, "thinking_mode": "provider_default"},
                "llm": {"model": "test"},
            }
        }
    }
    start = {
        "messages": messages,
        "response_format": rf,
        "model": "test",
        "max_tokens": 16384,
        "temperature": 0,
        "output_limit_parameter": "max_tokens",
        "operation_id": "op",
    }
    if fault == "missing":
        progress.pop("semantic_conflict_settlement")
    elif fault == "wire":
        receipt["wire_sha256"] = "e" * 64
    elif fault == "raw":
        progress["response"] += " "
    elif fault == "input":
        progress["semantic_conflict_settlement"]["input_sha256"] = "e" * 64
    elif fault == "batch":
        progress["semantic_conflict_settlement"]["batch_payload_sha256"] = "e" * 64
    elif fault == "numeric_alias":
        receipt["round_number"] = True
    monkeypatch.setattr(semantic, "semantic_conflict_identity", lambda: "f" * 64)
    if fault:
        with pytest.raises(ValueError):
            campaign.authenticate_planned_evaluation_start(
                start, record, owner.snapshot(), manifest
            )
    else:
        actual = campaign.authenticate_planned_evaluation_start(
            start, record, owner.snapshot(), manifest
        )
        assert actual["request_id"] == "assessment-0"


@pytest.mark.parametrize("partial", [False, True])
def test_four_exports_keep_original_verdict_and_host_effect_separate(
    display_case, partial
):
    """Publish the selected diagnostic without altering scores or the 3-case scope."""
    from pathlib import Path

    from test_unavailable_cost_exports import exported

    root, manifest, state, _ = display_case
    plan = budget_plan(scope())
    manifest["budget_plans"] = {cid: copy.deepcopy(plan) for cid, _ in campaign.CASES}
    manifest.update({key: plan["envelope"][key] for key in FIELDS})
    row = state["cases"][0]
    row.update(
        budget_snapshot=PlannedBudget(plan).snapshot(),
        supported=0,
        unsupported=0,
        unverified=1,
        publishable=False,
        status="blocked",
    )
    effect = {
        "contract": semantic.EFFECT_CONTRACT,
        "policy": POLICY,
        "policy_identity": plan["envelope"]["semantic_conflict_identity"],
        "code": "negative_status_with_complete_positive_diagnostics",
        "original_status": "unsupported",
        "original_reason_code": "insufficient_support",
        "decision_sha256": "d" * 64,
        "binding_sha256": "b" * 64,
    }
    claim = {
        "claim_id": "claim_48",
        "status": "unverifiable",
        "assessment_conflict_effect": effect,
    }
    details = {"assertions": [], "unsupported_claims": []}
    if partial:
        details["partial_assessment"] = {
            "stage": "host_validated",
            "assertions": [claim],
        }
    else:
        details.update(
            assessment_stage="host_validated",
            assertions=[claim],
            unsupported_claims=[copy.deepcopy(claim)],
        )
    report = {
        "survey": "# Unverified draft",
        "evaluation": {
            "faithfulness": {"score": None, "skipped": True, "details": details}
        },
    }
    archive_path = Path("artifacts/runs", row["run_id"] + ".json")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps({"run_id": row["run_id"], "report": report}))
    row["report_sha256"] = hashlib.sha256(report["survey"].encode()).hexdigest()
    before = copy.deepcopy(report)
    public, csv_rows, md, html = exported(root, state, manifest)
    actual = public["cases"][0]
    key = (
        "partial_semantic_conflict_effects" if partial else "semantic_conflict_effects"
    )
    assert len(actual[key]) == 1
    assert actual[key][0]["original_status"] == "unsupported"
    assert actual[key][0]["host_status"] == "unverified"
    assert json.loads(csv_rows[0][key]) == actual[key]
    assert actual["semantic_conflict_count"] == (None if partial else 1)
    assert public["passed"] == 0 and public["evaluation_completed"] == 0
    assert actual["supported"] == 0 and actual["unverified"] == 1
    assert all("semantic_conflict_count" not in r for r in public["cases"][1:])
    for text in (md, html):
        assert "claim_48" in text and effect["code"] in text
        assert "Original model" in text and "Host effect" in text
        assert "add no supported assertion" in text
    assert json.loads(archive_path.read_text())["report"] == before


def test_private_prepare_validate_freezes_pair_and_rejects_resealed_strip(
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
    historical = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    historical.parent.mkdir(parents=True)
    historical.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable source fixture."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    cfg = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    root = tmp_path / "prepared"
    campaign.prepare(
        root,
        cfg,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        **FLAGS,
        semantic_conflict_policy=POLICY,
    )
    manifest = campaign.validate(root, cfg)
    assert FIELDS.isdisjoint(manifest["options"])
    assert all(manifest["writing_parameters"][k] == manifest[k] for k in FIELDS)
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == manifest["source_hashes"][str(source.relative_to(tmp_path))]
    )
    for plan in manifest["budget_plans"].values():
        assert all(plan["envelope"][k] == manifest[k] for k in FIELDS)
    manifest.pop("semantic_conflict_identity")
    manifest["identity"] = campaign.digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        campaign.validate(root, cfg)
