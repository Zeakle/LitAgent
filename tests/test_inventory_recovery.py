"""Verify bounded re-extraction through the SDK and real report consumers."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
import test_located_evaluation as located
from test_evaluation_support_aliases import long_passages  # noqa: F401
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_flat_writer_pipeline import FlatGateway
from test_writer_v2 import case  # noqa: F401

from litagent.eval.inventory_recovery import CONTRACT, inventory_recovery_feedback
from litagent.quality import digest
from litagent.safety.budget import CostBudget


class RecoveryGateway(FlatGateway):
    """Script extraction defects at HTTP, preserving real budget and evaluation."""

    def __init__(self, case, fault="once"):
        super().__init__(case)
        self.recovery_fault = fault
        self.inventories = 0
        self.first_claim = None
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                self.inventories += 1
                fid = next(iter(value["fragments"]))
                if self.inventories == 1 or fault in {"repeat", "same"}:
                    value["fragments"][fid]["assertions"][0][
                        "claim_text"
                    ] += f" Composite extraction defect {self.inventories if fault == 'repeat' else 1}."
                if "inventory_recovery" in payload and fault == "json":
                    response.content = '{"fragments":'
                    return response
            if "analyses" in value and (self.inventories == 1 or fault == "repeat"):
                cid = payload["assertions"][0]["claim_id"]
                value["issues"] = [
                    {
                        "code": "non_atomic",
                        "anchor_ids": [cid],
                        "explanation": "Controlled non-atomic extraction, not report evidence.",
                    }
                ]
            if "cited_materials" in payload and fault in {"science", "science_fail"}:
                by_id = {c["claim_id"]: c for c in payload["assertions"]}
                for d in value["decisions"]:
                    if "FLOPs by" in by_id[d["claim_id"]]["claim_text"]:
                        d.update(
                            status="unsupported",
                            reason_code="insufficient_support",
                            support_basis=[],
                            explanation="No cited FLOPs result.",
                        )
                        for aspect in d["aspect_checks"]:
                            aspect.update(supported=False, support_ids=[])
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    async def __call__(self, request):
        body = json.loads(request.content)
        payload = json.loads(body.get("input", body.get("messages"))[1]["content"])
        if "inventory_recovery" in payload and self.recovery_fault == "timeout":
            raise httpx.ReadTimeout("Controlled recovery timeout", request=request)
        if "inventory_recovery" in payload and self.recovery_fault == "cancel":
            raise asyncio.CancelledError()
        response = await super().__call__(request)
        if "inventory_recovery" in payload and self.recovery_fault == "unknown":
            value = response.json()
            value.pop("usage", None)
            return httpx.Response(200, json=value)
        return response

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if self.recovery_fault in {"science", "science_fail"}:
            if payload.get("requested_body_section") == "methods":
                value["units"][0]["text"] = "Relation networks reduce FLOPs by 44%."
            if "targets" in payload:
                for patch in value["patches"]:
                    patch["text"] = (
                        "Relation networks reduce FLOPs by 45%."
                        if self.recovery_fault == "science_fail"
                        else "In the cited few-shot setting, Relation networks learn a comparison function."
                    )
                    patch["fact_card_refs"] = [self.case.by_eid["b"]]
        return value


async def execute(
    case, tmp_path, monkeypatch, fault="once", enabled=True, classification_review=False
):
    """Enable only the server-side experimental flag; do not stub consumers."""
    gateway = RecoveryGateway(case, fault)
    monkeypatch.setattr(located, "FindingsGateway", lambda *_: gateway)
    original_methodology = located.build_methodology

    def recorded_methodology(query, upstream, **kwargs):
        tasks = {
            "__search_queries__": {"search_fixture": "relation networks"},
            "search_fixture": list(case.ledger.values()),
            "dedup": list(case.ledger.values()),
            "relevance_gate": list(case.ledger.values()),
        }
        kwargs["source_outcomes"] = [
            {
                "task_id": "search_fixture",
                "source": "offline-public-fixture",
                "query": "relation networks",
                "status": "completed",
                "result_count": len(case.ledger),
            }
        ]
        return original_methodology(query, tasks, **kwargs)

    monkeypatch.setattr(located, "build_methodology", recorded_methodology)
    report, gateway = await located.run_case(
        case,
        tmp_path,
        monkeypatch,
        "recovery",
        inventory_recovery=enabled,
        classification_review=classification_review,
        allow_uncertain=fault in {"timeout", "unknown"},
    )
    (tmp_path / "result.json").write_text(json.dumps(report))
    return report, gateway


@pytest.mark.asyncio
async def test_reextracts_full_unchanged_report_and_repeats_every_support_check(
    case, tmp_path, monkeypatch
):
    report, gateway = await execute(case, tmp_path, monkeypatch)
    history = report["metadata"]["evaluation_history"]
    first = history["initial_evaluation"]["faithfulness"]["details"]
    final = report["evaluation"]["faithfulness"]["details"]
    assert first["inventory_check"]["status"] == "failed"
    assert final["inventory_check"]["status"] == "passed"
    assert first["source_report_sha256"] == final["source_report_sha256"]
    assert final["coverage"]["complete"]
    assert report["delivery"]["publishable"]
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert gateway.inventories == 2
    calls = gateway.evaluator.base.calls
    inventories = [p for p in calls if "allowed_citation_aliases" in p]
    assert inventories[0]["source_units"] == inventories[1]["source_units"]
    assert inventories[1]["inventory_recovery"]["contract"] == CONTRACT
    recovered_wire = next(
        row for row in gateway.sent if "inventory_recovery" in row["payload"]
    )
    assert recovered_wire["request"]["model"] == "lite"
    assert recovered_wire["request"]["max_tokens"] == 16384
    assert recovered_wire["request"]["thinking"] == {"type": "disabled"}
    assert {
        f["source_fragment_id"]
        for f in recovered_wire["payload"]["inventory_recovery"]["findings"]
    }.issubset(
        {
            f["source_fragment_id"]
            for u in recovered_wire["payload"]["source_units"]
            for f in u["source_fragments"]
        }
    )
    assert len([p for p in calls if "cited_materials" in p]) == 2
    for row in gateway.sent:
        if "cited_materials" in row["payload"]:
            assert set(row["payload"]["writer_envelope"]) == {
                "claim_json_bytes",
                "claim_max_json_bytes",
                "diagnostic_json_bytes",
            }
    assert not any("targets" in row["payload"] for row in gateway.sent)
    ledger = json.loads(
        Path(str(tmp_path / "repair.json") + ".budget-ledger.json").read_text()
    )
    entries = list(ledger["entries"].values())
    assert sum(e["stage"] == "inventory" for e in entries) == 1
    assert (
        sum(e["stage"] == "repair" and "inventory_recovery" in e for e in entries) == 1
    )
    assert ledger["capacity_allocation"]["round"]["number"] == 2
    assert CostBudget.from_snapshot(ledger).snapshot() == ledger
    assert history["attempts"][0]["kind"] == "inventory_recovery"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,expected", [("repeat", 2), ("same", 1), ("json", 1)])
async def test_failed_reextract_never_bypasses_gates_or_gets_a_third_attempt(
    case, tmp_path, monkeypatch, fault, expected
):
    report, gateway = await execute(case, tmp_path, monkeypatch, fault)
    assert not report["delivery"]["publishable"]
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    assert gateway.inventories == expected + 1
    assert not any("targets" in row["payload"] for row in gateway.sent)
    if fault == "same":
        assert (
            len([p for p in gateway.evaluator.base.calls if "cited_materials" in p])
            == 1
        )
        assert report["evaluation"]["faithfulness"]["skipped"]


@pytest.mark.asyncio
async def test_old_envelope_does_not_gain_recovery_requests(
    case, tmp_path, monkeypatch
):
    report, gateway = await execute(case, tmp_path, monkeypatch, enabled=False)
    assert gateway.inventories == 1
    assert not report["metadata"]["targeted_repair_state"]["attempts"]
    assert not report["delivery"]["publishable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["science", "science_fail"])
@pytest.mark.parametrize("classification_review", [False, True])
async def test_inventory_recovery_and_prose_repair_share_exactly_two_slots(
    case, tmp_path, monkeypatch, fault, classification_review
):
    report, gateway = await execute(
        case, tmp_path, monkeypatch, fault, classification_review=classification_review
    )
    assert gateway.inventories == 3
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["purpose"] == CONTRACT
    assert "purpose" not in attempts[1] or attempts[1]["purpose"] != CONTRACT
    assert sum("targets" in row["payload"] for row in gateway.sent) == 1
    first = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    middle = report["metadata"]["evaluation_history"]["attempts"][0][
        "candidate_evaluation"
    ]["faithfulness"]["details"]
    assert first["source_report_sha256"] == middle["source_report_sha256"]
    assert middle["inventory_check"]["status"] == "passed"
    assert middle["unsupported_count"] > 0
    if fault == "science_fail":
        assert "FLOPs by 44%" in report["survey"]
        assert "FLOPs by 45%" not in report["survey"]
        assert not report["delivery"]["publishable"]
    else:
        assert "FLOPs by" not in report["survey"]
        assert (
            report["evaluation"]["faithfulness"]["details"]["strict_acceptance"][
                "status"
            ]
            == "passed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["timeout", "unknown"])
async def test_unknown_inventory_response_holds_funds_and_never_replays(
    case, tmp_path, monkeypatch, fault
):
    report, gateway = await execute(case, tmp_path, monkeypatch, fault)
    assert not report["delivery"]["publishable"]
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 1
    ledger = json.loads(
        Path(str(tmp_path / "repair.json") + ".budget-ledger.json").read_text()
    )
    assert ledger["usage_uncertain"]
    assert ledger["stop_reason"] == "strict_usage_unknown"
    recovery = next(e for e in ledger["entries"].values() if "inventory_recovery" in e)
    assert recovery["status"] == "interrupted_no_replay"
    restored = CostBudget.from_snapshot(ledger)
    assert restored.snapshot()["usage_uncertain"]
    assert restored.stop_reason == "strict_usage_unknown"
    assert gateway.inventories <= 2


def diagnostic_fixture():
    """Build exact host locations without depending on a private run archive."""
    import hashlib

    from litagent.eval.scoped import build_source_fragments
    from litagent.evidence import extract_evidence_refs_ordered
    from litagent.report_render import build_report_source_units

    survey = "## Methods\n\nMethod A changes embeddings and Method B adapts distance. [E:a]\n\nMethod A changes embeddings and Method B adapts distance. [E:b]\n"
    units = build_report_source_units(survey)
    claims = []
    for u in units:
        fragment = build_source_fragments(u)[0]
        claims.append(
            dict(
                claim_id=f"claim_{len(claims)}",
                kind="scientific",
                claim_text=fragment["source_quote"],
                source_unit_id=u.unit_id,
                source_fragment_id=fragment["source_fragment_id"],
                source_start=u.start + fragment["start"],
                source_end=u.start + fragment["end"],
                source_quote=fragment["source_quote"],
                evidence_ids=extract_evidence_refs_ordered(fragment["source_quote"]),
                atomic=False,
                classification_valid=True,
                source_faithful=True,
                explanation="Non-atomic.",
            )
        )
    return survey, dict(
        assessment_stage="host_validated",
        inventory_check={"status": "failed"},
        stages={"global_check": "completed"},
        source_report_sha256=hashlib.sha256(survey.encode()).hexdigest(),
        assertions=claims,
        global_issues=[],
    )


def test_identical_text_different_positions_and_citations_keeps_both():
    survey, details = diagnostic_fixture()
    feedback = inventory_recovery_feedback(survey, details)
    assert len(feedback["findings"]) == 2
    assert sum(len(f["claim_ids"]) for f in feedback["findings"]) == 2
    assert len({f["source_fragment_id"] for f in feedback["findings"]}) == 2


@pytest.mark.parametrize(
    "field",
    [
        "source_start",
        "source_unit_id",
        "source_fragment_id",
        "evidence_ids",
        "source_report_sha256",
    ],
)
def test_stale_or_forged_source_binding_never_authorizes_recovery(field):
    survey, details = diagnostic_fixture()
    if field == "source_report_sha256":
        details[field] = "a" * 64
    else:
        details["assertions"][0][field] = {
            "source_start": 0,
            "source_unit_id": "fake",
            "source_fragment_id": "fake",
            "evidence_ids": ["b"],
        }[field]
    with pytest.raises(ValueError, match="identity_invalid"):
        inventory_recovery_feedback(survey, details)


@pytest.mark.parametrize("flag", ["classification_valid", "source_faithful"])
def test_non_atomic_recovery_cannot_relabel_classification_or_source_errors(flag):
    survey, details = diagnostic_fixture()
    details["assertions"][0][flag] = False
    assert inventory_recovery_feedback(survey, details) is None


@pytest.mark.asyncio
async def test_cancelled_recovery_persists_consumed_attempt_and_unknown_hold(
    case, tmp_path, monkeypatch
):
    with pytest.raises(asyncio.CancelledError):
        await execute(case, tmp_path, monkeypatch, "cancel")
    shared = json.loads((tmp_path / "repair.json").read_text())["state"]
    assert len(shared["attempts"]) == 1
    assert shared["attempts"][0]["status"] == "cancelled"
    budget = json.loads(
        Path(str(tmp_path / "repair.json") + ".budget-ledger.json").read_text()
    )
    assert budget["usage_uncertain"]
    assert any(
        e["status"] == "interrupted_no_replay" and "inventory_recovery" in e
        for e in budget["entries"].values()
    )
    from litagent.quality import load_targeted_repairs

    assert (
        len(
            load_targeted_repairs(tmp_path / "repair.json", "located-offline")[
                "attempts"
            ]
        )
        == 1
    )


def test_recovery_marker_changes_new_plan_only_and_invalid_policy_is_rejected(case):
    from litagent.writing_v2 import writer_budget_plan

    old = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    new = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
    )
    assert old["identity"] != new["identity"]
    assert "inventory_recovery_contract" not in old["envelope"]
    assert new["envelope"]["inventory_recovery_contract"] == CONTRACT
    assert old["planned_tokens"] == new["planned_tokens"]
    assert old["repair_limit"] == new["repair_limit"] == 2
    for policy in ["planned-budget-v5", "planned-budget-v6"]:
        with pytest.raises(ValueError, match="recovery_policy"):
            writer_budget_plan(
                case.packet,
                case.aliases,
                methodology={},
                budget_policy=policy,
                inventory_recovery=True,
            )


@pytest.mark.parametrize("value", [False, None, [], {}, 1, "enabled"])
def test_cli_recovery_option_rejects_untyped_flags(value):
    from scripts.run_frozen_writing import ROUND_PROFILE, inventory_recovery_options

    with pytest.raises(ValueError, match="recovery_configuration"):
        inventory_recovery_options(
            {"writing_profile": ROUND_PROFILE, "inventory_recovery": value}
        )


def test_cli_server_option_is_explicit_and_old_manifests_keep_empty_kwargs():
    from scripts.run_frozen_writing import (
        FLAT_PROFILE,
        ROUND_PROFILE,
        inventory_recovery_options,
    )

    assert inventory_recovery_options({"writing_profile": ROUND_PROFILE}) == {}
    assert inventory_recovery_options(
        {"writing_profile": ROUND_PROFILE, "inventory_recovery": True}
    ) == {"inventory_recovery": True}
    with pytest.raises(ValueError, match="recovery_configuration"):
        inventory_recovery_options(
            {"writing_profile": FLAT_PROFILE, "inventory_recovery": True}
        )


@pytest.mark.asyncio
async def test_recovery_receipts_and_pending_ownership_cannot_be_forged(
    case, tmp_path, monkeypatch
):
    report, _ = await execute(case, tmp_path, monkeypatch)
    snapshot = json.loads(
        Path(str(tmp_path / "repair.json") + ".budget-ledger.json").read_text()
    )
    rid = next(k for k, e in snapshot["entries"].items() if "inventory_recovery" in e)
    for field in ["parent_allocation", "source_report_sha256", "contract"]:
        value = copy.deepcopy(snapshot)
        value["entries"][rid]["inventory_recovery"][field] = "bad"
        with pytest.raises(ValueError):
            CostBudget.from_snapshot(value)
    value = copy.deepcopy(snapshot)
    del value["entries"][rid]["inventory_recovery"]
    with pytest.raises(ValueError):
        CostBudget.from_snapshot(value)
    value = copy.deepcopy(snapshot)
    value["entries"][rid]["stage"] = "inventory"
    with pytest.raises(ValueError):
        CostBudget.from_snapshot(value)


@pytest.mark.asyncio
async def test_pending_unknown_recovery_binding_is_monotonic_on_restore(
    case, tmp_path, monkeypatch
):
    await execute(case, tmp_path, monkeypatch, "unknown")
    snapshot = json.loads(
        Path(str(tmp_path / "repair.json") + ".budget-ledger.json").read_text()
    )
    owner = CostBudget.from_snapshot(snapshot)
    changed = copy.deepcopy(snapshot)
    next(e for e in changed["entries"].values() if "inventory_recovery" in e)[
        "inventory_recovery"
    ]["feedback_sha256"] = ("a" * 64)
    with pytest.raises(ValueError, match="restore_regression"):
        owner.restore(changed)
    assert owner.snapshot() == snapshot


def test_cli_history_labels_inventory_recovery_without_inventing_a_prose_revision():
    from scripts.run_frozen_writing import evaluation_attempt_label

    assert (
        evaluation_attempt_label({"kind": "inventory_recovery"}, 0)
        == "inventory_recovery_1"
    )
    assert evaluation_attempt_label({}, 1) == "repair_2"


def test_alias_batch_branch_keeps_host_plan_out_but_all_original_sources_in(
    case, long_passages
):
    """Replay the large alias branch that escaped the small SDK fixture."""
    from litagent.eval.batched import batch_prompt
    from litagent.eval.scoped import build_evaluation_request
    from litagent.writing_v2 import writer_budget_plan

    payload = copy.deepcopy(long_passages)
    payload["writer_envelope"] = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
    )["envelope"]
    before = copy.deepcopy(payload)
    messages, schema = build_evaluation_request(batch_prompt(payload), payload)
    wire = json.loads(messages[1]["content"])
    assert wire["assessment_input_contract"] == payload["assessment_input_contract"]
    assert set(wire["writer_envelope"]) == {
        "claim_json_bytes",
        "claim_max_json_bytes",
        "diagnostic_json_bytes",
    }
    assert wire["original_supports"] == payload["original_supports"]
    assert len(wire["assertions"]) == len(payload["assertions"])
    assert payload == before
