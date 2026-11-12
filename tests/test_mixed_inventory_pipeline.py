"""Exercise mixed-inventory recovery through HTTP and real report consumers."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from model_config_helpers import load_previous_default_config as load_config
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.eval.inventory_recovery import inventory_identity
from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
from litagent.quality import digest, load_targeted_repairs, strict_acceptance
from litagent.review_structure import build_methodology
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget
from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

MIXED_CONTRACT = "bounded-inventory-recovery-v2"
MIXED_FINDINGS = (
    ("non_atomic", "Independent mechanisms require separate assertions."),
    ("classification", "The scientific premise was classified as organization."),
    ("source_fidelity", "The assertion matches its original text; no drift was found."),
    ("missing_assertion", "All propositions were extracted; no omission was found."),
)


class MixedInventoryGateway(FlatGateway):
    """Script only external replies, preserving actual SDK and host validation."""

    def __init__(self, case, fault="once"):
        super().__init__(case)
        self.mixed_fault = fault
        self.declined = fault == "wrong_basis"
        self.inventories = 0
        self.repair_count = 0
        self.recovery_requests = 0
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                self.inventories += 1
                first = next(iter(value["fragments"].values()))
                if self.has_defect:
                    version = (
                        self.inventories
                        if fault == "repeat"
                        else (2 if fault == "cycle" and self.inventories == 2 else 1)
                    )
                    first["assertions"][0][
                        "claim_text"
                    ] += f" Extraction-only variant {version}."
                    first["assertions"][0]["kind"] = "report_structure"
                if self.inventories > 1 and fault == "empty":
                    first.update(assertions=[], exclusion_reason="No assertions.")
                if self.inventories > 1 and fault == "escape":
                    for fragment in value["fragments"].values():
                        for claim in fragment["assertions"]:
                            claim["kind"] = "report_structure"
                if "inventory_recovery" in payload and fault == "json":
                    response.content = '{"fragments":'
                    return response
            if "cited_materials" in payload:
                claims = {c["claim_id"]: c for c in payload["assertions"]}
                for decision in value["decisions"]:
                    claim = claims[decision["claim_id"]]
                    if "Extraction-only variant" in claim["claim_text"]:
                        decision.update(
                            correct_kind="scientific",
                            source_faithful=False,
                            atomic=False,
                        )
                        if "classification_valid" in decision:
                            decision["classification_valid"] = False
                    if fault == "escape" and self.inventories > 1:
                        decision["correct_kind"] = "scientific"
                        if "classification_valid" in decision:
                            decision["classification_valid"] = False
                    if "FLOPs by" in claim["claim_text"]:
                        decision.update(
                            status="unsupported",
                            reason_code="insufficient_support",
                            support_basis=[],
                            explanation="The cited source reports no FLOPs reduction.",
                        )
                        for aspect in decision["aspect_checks"]:
                            aspect.update(supported=False, support_ids=[])
                    if fault == "wrong_basis" and self.inventories > 1:
                        aliases = {e["evidence_alias"] for e in claim["evidence"]}
                        foreign = next(
                            (
                                pid
                                for pid, passage in payload["support_passages"].items()
                                if passage[0] not in aliases
                                and passage[1] != passage[0]
                            ),
                            None,
                        )
                        if foreign is not None:
                            decision["support_basis"] = [{"support_id": foreign}]
                            for aspect in decision["aspect_checks"]:
                                aspect["support_ids"] = [foreign]
            if "analyses" in value:
                if self.has_defect or (self.inventories > 1 and fault == "escape"):
                    cid = payload["assertions"][0]["claim_id"]
                    fid = payload["assertions"][0]["source_fragment_id"]
                    uid = next(
                        u["source_unit_id"]
                        for u in payload["source_units"]
                        if any(
                            f["source_fragment_id"] == fid
                            for f in u["source_fragments"]
                        )
                    )
                    value["issues"] = [
                        {
                            "code": code,
                            "anchor_ids": [uid if code == "missing_assertion" else cid],
                            "explanation": explanation,
                        }
                        for code, explanation in MIXED_FINDINGS
                    ]
                    if fault == "feedback_capacity":
                        # This aggregate legal-diagnostic fixture tests host input
                        # admission, not achievable provider output or real usage.
                        value["issues"] = [
                            {
                                "code": code,
                                "anchor_ids": [
                                    uid if code == "missing_assertion" else cid
                                ],
                                "explanation": (
                                    f"Concern {index}: {explanation} "
                                    + (
                                        "Keep the complete condition, source ownership, "
                                        "negation and original proposition for independent "
                                        "re-extraction; this advisory concern is not evidence. "
                                        * 20
                                    )
                                )[:2000],
                            }
                            for index in range(32)
                            for code, explanation in [MIXED_FINDINGS[index % 4]]
                        ]
                if self.inventories > 1 and fault == "empty":
                    value["units"][0]["complete"] = False
                science = {
                    c["claim_id"]
                    for c in payload["assertions"]
                    if c["kind"] == "scientific"
                }
                for row in [
                    *value["analyses"],
                    value["limitations"],
                    *value["coverage"],
                ]:
                    row["claim_ids"] = [c for c in row["claim_ids"] if c in science]
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    @property
    def has_defect(self):
        """Keep first-inventory faults distinct from deliberately persistent ones."""
        return self.inventories == 1 or self.mixed_fault in {"same", "repeat", "cycle"}

    async def __call__(self, request):
        body = json.loads(request.content)
        payload = json.loads(body.get("input", body.get("messages"))[1]["content"])
        recovering = "inventory_recovery" in payload
        if recovering:
            self.recovery_requests += 1
        if recovering and self.mixed_fault == "timeout":
            raise httpx.ReadTimeout(
                "Controlled mixed recovery timeout", request=request
            )
        if recovering and self.mixed_fault == "cancel":
            raise asyncio.CancelledError()
        response = await super().__call__(request)
        if recovering and self.mixed_fault == "unknown":
            value = response.json()
            value.pop("usage", None)
            return httpx.Response(200, json=value)
        return response

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if self.mixed_fault in {"science", "science_fail"}:
            if payload.get("requested_body_section") == "methods":
                value["units"][0]["text"] = "Relation networks reduce FLOPs by 44%."
            if "targets" in payload:
                self.repair_count += 1
                for patch in value["patches"]:
                    patch["text"] = (
                        "Relation networks reduce FLOPs by 45%."
                        if self.mixed_fault == "science_fail"
                        else "In the cited few-shot setting, Relation networks learn a comparison function."
                    )
                    patch["fact_card_refs"] = [self.case.by_eid["b"]]
        return value


async def execute(
    case, tmp_path, monkeypatch, fault="once", *, mixed=True, gateway=None
):
    """Run actual frozen workers, SDK, evaluator, repair state and report outlet."""
    gateway = gateway or MixedInventoryGateway(case, fault)
    owner, _, _ = await owner_for(gateway)
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.review_max_tokens = 4096
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**kwargs):
        owner._model = kwargs["model"]
        owner._cost_budget = kwargs["cost_budget"]
        owner._trace_hook = kwargs["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    records = {
        "__search_queries__": {"search_fixture": "relation networks"},
        "search_fixture": list(case.ledger.values()),
        "dedup": list(case.ledger.values()),
        "relevance_gate": list(case.ledger.values()),
    }
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            records,
            source_outcomes=[
                {
                    "task_id": "search_fixture",
                    "source": "offline-public-fixture",
                    "query": "relation networks",
                    "status": "completed",
                    "result_count": len(case.ledger),
                }
            ],
            started_at=1700000000,
        ),
        "fact_cards_sha256": digest(case.cards),
    }
    before = copy.deepcopy(bundle)
    flags = dict(inventory_recovery=True, classification_review=True)
    if mixed:
        flags["mixed_inventory_recovery"] = True
    plan = plan_frozen_budget(bundle, budget_policy="planned-budget-v7", **flags)
    try:
        report = await run_frozen(
            agent,
            bundle,
            "mixed-inventory-offline",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **flags,
        )
        assert bundle == before
        (tmp_path / "result.json").write_text(json.dumps(report))
        return report, gateway
    finally:
        await agent.cleanup()


def details(report):
    """Resolve the final actual v3 product rather than a mocked metric."""
    return report["evaluation"]["faithfulness"]["details"]


def ledger(tmp_path):
    """Read the persisted budget, including unknown holds."""
    return json.loads(
        Path(str(tmp_path / "repair.json") + ".budget-ledger.json").read_text()
    )


@pytest.mark.asyncio
async def test_mixed_recovery_reextracts_same_report_and_rechecks_every_scientific_claim(
    case, tmp_path, monkeypatch
):
    report, gateway = await execute(case, tmp_path, monkeypatch)
    first = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    final = details(report)
    assert first["inventory_check"]["status"] == "failed"
    assert {i["code"] for i in first["global_issues"]} == {c for c, _ in MIXED_FINDINGS}
    assert final["source_report_sha256"] == first["source_report_sha256"]
    assert any(c["source_faithful"] is False for c in first["assertions"])
    assert any(c["classification_valid"] is False for c in first["assertions"])
    assert final["inventory_check"]["status"] == "passed"
    assert final["coverage"]["complete"]
    assert strict_acceptance(final)["status"] == "passed"
    assert report["delivery"]["publishable"]
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 1 and attempts[0]["purpose"] == MIXED_CONTRACT
    calls = gateway.evaluator.base.calls
    inventories = [p for p in calls if "allowed_citation_aliases" in p]
    assert gateway.inventories == len(inventories) == 2
    before_units, recovered_units = (
        inventories[0]["source_units"],
        inventories[1]["source_units"],
    )
    assert len(before_units) == len(recovered_units)
    for index, (before, recovered) in enumerate(
        zip(before_units, recovered_units, strict=True), 1
    ):
        # v2 aliases the exact ordered host directory; every prose/citation and
        # fragment range identity must remain identical after that one mapping.
        assert recovered == {**before, "source_unit_id": f"v{index}"}
    assert inventories[1]["inventory_recovery"]["contract"] == MIXED_CONTRACT
    recovery = next(p for p in gateway.sent if "inventory_recovery" in p["payload"])
    feedback_text = json.dumps(recovery["payload"]["inventory_recovery"])
    for _, explanation in MIXED_FINDINGS:
        assert explanation in feedback_text
    batches = [p for p in calls if "cited_materials" in p]
    assert len(batches) == 2
    assert {c["source_fragment_id"] for c in batches[0]["assertions"]} == {
        c["source_fragment_id"] for c in batches[1]["assertions"]
    }
    assert batches[0]["original_supports"] == batches[1]["original_supports"]
    assert len([p for p in calls if "analyses" in p["schema"]["properties"]]) == 2
    assert not any("targets" in row["payload"] for row in gateway.sent)
    assert CostBudget.from_snapshot(ledger(tmp_path)).snapshot() == ledger(tmp_path)


@pytest.mark.asyncio
async def test_old_v1_does_not_gain_mixed_recovery_or_reclassify_previous_failures(
    case, tmp_path, monkeypatch
):
    report, gateway = await execute(case, tmp_path, monkeypatch, mixed=False)
    assert gateway.inventories == 1
    assert not report["metadata"]["targeted_repair_state"]["attempts"]
    assert details(report)["inventory_check"]["status"] == "failed"
    assert not report["delivery"]["publishable"]


@pytest.mark.asyncio
async def test_recovered_inventory_cannot_borrow_another_papers_support(
    case, tmp_path, monkeypatch
):
    case.body["taxonomy"][0]["sentences"][0] = {
        "text": "On DeiT-S, EViT reduces computation.",
        "fact_card_refs": [case.by_eid["a"]],
    }
    report, gateway = await execute(case, tmp_path, monkeypatch, "wrong_basis")
    history = report["metadata"]["evaluation_history"]
    recovered = history["attempts"][0]["candidate_evaluation"]["faithfulness"][
        "details"
    ]
    assert recovered["inventory_check"]["status"] == "passed"
    assert recovered["unverified_count"] > 0
    assert any(
        c.get("support_validation", {}).get("issue")
        == "support_basis_ownership_invalid"
        for c in recovered["assertions"]
    )
    assert not report["delivery"]["publishable"]
    assert (
        gateway.inventories == 2
        and len(report["metadata"]["targeted_repair_state"]["attempts"]) <= 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["science", "science_fail"])
async def test_mixed_recovery_and_scientific_patch_share_only_two_attempts(
    case, tmp_path, monkeypatch, fault
):
    report, gateway = await execute(case, tmp_path, monkeypatch, fault)
    history = report["metadata"]["evaluation_history"]
    first = history["initial_evaluation"]["faithfulness"]["details"]
    middle = history["attempts"][0]["candidate_evaluation"]["faithfulness"]["details"]
    assert first["source_report_sha256"] == middle["source_report_sha256"]
    assert middle["inventory_check"]["status"] == "passed"
    assert middle["unsupported_count"] > 0
    rejected = [c for c in middle["assertions"] if "FLOPs by" in c["claim_text"]]
    assert rejected and all(
        c["source_faithful"] is True and c["classification_valid"] is True
        for c in rejected
    )
    assert gateway.repair_count == 1 and gateway.inventories == 3
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["purpose"] == MIXED_CONTRACT
    assert sum("targets" in p["payload"] for p in gateway.sent) == 1
    final = details(report)
    if fault == "science":
        assert final["coverage"]["complete"]
        assert final["supported_count"] == final["factual_claim_count"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert report["delivery"]["publishable"]
    else:
        assert not report["delivery"]["publishable"]
        assert "FLOPs by 44%" in report["survey"]
        assert "FLOPs by 45%" not in report["survey"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["same", "repeat", "cycle", "empty", "escape", "json"]
)
async def test_no_progress_cycle_omission_or_reclassification_never_delivers_or_retries_forever(
    case, tmp_path, monkeypatch, fault
):
    report, gateway = await execute(case, tmp_path, monkeypatch, fault)
    assert not report["delivery"]["publishable"]
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert 1 <= len(attempts) <= 2
    assert gateway.inventories <= 3
    assert not any("targets" in row["payload"] for row in gateway.sent)
    if fault == "same":
        assert len(attempts) == 1
        assert (
            len([p for p in gateway.evaluator.base.calls if "cited_materials" in p])
            == 1
        )
        assert details(report)["skipped_reason"] == "inventory_recovery_no_progress"
    assert (
        load_targeted_repairs(tmp_path / "repair.json", "mixed-inventory-offline")[
            "attempts"
        ]
        == attempts
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["timeout", "unknown"])
async def test_unknown_recovery_consumes_one_attempt_and_keeps_unknown_reservation(
    case, tmp_path, monkeypatch, fault
):
    report, gateway = await execute(case, tmp_path, monkeypatch, fault)
    assert not report["delivery"]["publishable"]
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    snapshot = ledger(tmp_path)
    assert (
        snapshot["usage_uncertain"]
        and snapshot["stop_reason"] == "strict_usage_unknown"
    )
    entries = [e for e in snapshot["entries"].values() if "inventory_recovery" in e]
    assert len(entries) == 1 and entries[0]["status"] == "interrupted_no_replay"
    assert entries[0]["inventory_recovery"]["contract"] == MIXED_CONTRACT
    assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
    assert gateway.inventories <= 2
    assert gateway.recovery_requests == 1


@pytest.mark.asyncio
async def test_cancelled_mixed_recovery_keeps_attempt_and_never_sends_another_request(
    case, tmp_path, monkeypatch
):
    gateway = MixedInventoryGateway(case, "cancel")
    with pytest.raises(asyncio.CancelledError):
        await execute(case, tmp_path, monkeypatch, "cancel", gateway=gateway)
    shared = load_targeted_repairs(tmp_path / "repair.json", "mixed-inventory-offline")
    assert (
        len(shared["attempts"]) == 1 and shared["attempts"][0]["status"] == "cancelled"
    )
    snapshot = ledger(tmp_path)
    assert snapshot["usage_uncertain"]
    recovery = [e for e in snapshot["entries"].values() if "inventory_recovery" in e]
    assert len(recovery) == 1 and recovery[0]["status"] == "interrupted_no_replay"
    assert gateway.recovery_requests == 1


@pytest.mark.asyncio
async def test_complete_feedback_capacity_is_a_resource_terminal_before_recovery_send(
    case, tmp_path, monkeypatch
):
    report, gateway = await execute(case, tmp_path, monkeypatch, "feedback_capacity")
    first = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    assert first["assessment_stage"] == "host_validated"
    assert first["stages"]["global_check"] == "completed"
    assert len(first["global_issues"]) == 32
    assert len({f["explanation"] for f in first["global_issues"]}) == 32
    assert all(len(f["explanation"]) == 2000 for f in first["global_issues"])
    assert not report["delivery"]["publishable"]
    assert not report["metadata"]["targeted_repair_state"]["attempts"]
    assert gateway.inventories == 1 and gateway.recovery_requests == 0
    final = details(report)
    assert final["skipped_reason"] == "inventory_recovery_request_capacity_exceeded"
    failure = report["metadata"]["writer_failure"]
    assert failure["family"] == "resource"
    assert failure["code"] == "inventory_recovery_request_capacity_exceeded"
    snapshot = ledger(tmp_path)
    assert not snapshot["usage_uncertain"]
    assert not any("inventory_recovery" in e for e in snapshot["entries"].values())


def test_inventory_identity_ignores_id_only_changes_but_preserves_each_native_position():
    original = [
        {
            "claim_id": "c1",
            "source_fragment_id": "f1",
            "kind": "scientific",
            "claim_text": "A fact.",
            "evidence_ids": ["a"],
        },
        {
            "claim_id": "c2",
            "source_fragment_id": "f2",
            "kind": "scientific",
            "claim_text": "A fact.",
            "evidence_ids": ["b"],
        },
    ]
    renamed = [{**c, "claim_id": f"new{i}"} for i, c in enumerate(original)]
    assert inventory_identity(original) == inventory_identity(renamed)
    assert inventory_identity(original) != inventory_identity(original[:1])
    assert inventory_identity(original) != inventory_identity(list(reversed(original)))
