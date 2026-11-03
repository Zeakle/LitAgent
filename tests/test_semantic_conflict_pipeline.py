"""Exercise conservative conflict recovery through the complete frozen workflow."""

import copy
import json

import httpx
import pytest
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_source_map_pipeline import products, sent_with
from test_source_owner_pipeline import FLAGS as SOURCE_OWNER_FLAGS
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import load_config
from litagent.eval.batched import _restore_batch_support_ids
from litagent.eval.scoped import Assessment
from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
from litagent.quality import digest, strict_acceptance
from litagent.review_structure import build_methodology
from litagent.runner import LitAgent, RunPolicy
from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

POLICY = "host-unverified-continue-v1"
CONTRACT = "host-assessment-conflict-effect-v1"
POSITIVE = "positive_status_with_negative_required_aspect"
NEGATIVE = "negative_status_with_complete_positive_diagnostics"
FLAGS = {**SOURCE_OWNER_FLAGS, "semantic_conflict_policy": POLICY}
CANDIDATE_TEXTS = (
    "As the cited source states, relation networks learn a comparison function.",
    "The cited source describes relation networks learning a comparison function.",
)


class ConflictGateway(FlatGateway):
    """Change provider JSON only while keeping SDK settlement and consumers real."""

    def __init__(self, case, *, code=NEGATIVE, failing_candidates=0, fault=None):
        super().__init__(case)
        self.code = code
        self.failing_candidates = failing_candidates
        self.conflict_fault = fault
        self.repair_count = 0
        self.inventory_calls = 0
        self.global_calls = 0
        self.assessment_calls = 0
        self.injected = []
        self.injected_payloads = []
        self.injected_rounds = set()
        self.order = []
        original = self.evaluator.chat

        async def chat(messages, **options):
            response = await original(messages, **options)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                self.inventory_calls += 1
                self.order.append("inventory")
            if "global_output_contract" in payload:
                self.global_calls += 1
                self.order.append("global")
                if self.conflict_fault == "global_incomplete":
                    value["units"][0]["complete"] = False
            if "cited_materials" in payload:
                self.assessment_calls += 1
                self.order.append("assessment")
                should_inject = (
                    self.repair_count <= self.failing_candidates
                    and self.inventory_calls not in self.injected_rounds
                )
                if should_inject:
                    claims = {
                        claim["claim_id"]: claim for claim in payload["assertions"]
                    }
                    row = next(
                        (
                            item
                            for item in value["decisions"]
                            if item["correct_kind"] == "scientific"
                            and (
                                claims[item["claim_id"]]["claim_text"].startswith(
                                    "methods:"
                                )
                                or any(
                                    text in claims[item["claim_id"]]["claim_text"]
                                    for text in CANDIDATE_TEXTS
                                )
                            )
                        ),
                        None,
                    )
                    if row is not None:
                        row["explanation"] = (
                            "Controlled contradictory diagnostics; not a scientific gold label."
                        )
                        if self.code == NEGATIVE:
                            row.update(
                                status="unsupported", reason_code="insufficient_support"
                            )
                        else:
                            row["aspect_checks"][0].update(
                                supported=False,
                                support_ids=[],
                                explanation="Controlled required-aspect negative.",
                            )
                        if self.conflict_fault == "bad_basis":
                            row["support_basis"][0][
                                "support_id"
                            ] = "outside-claim-basis"
                        elif self.conflict_fault == "extra_aspect":
                            row["aspect_checks"].append(
                                copy.deepcopy(row["aspect_checks"][0])
                            )
                        elif self.conflict_fault == "atomic":
                            row["atomic"] = False
                        elif self.conflict_fault == "fidelity":
                            row["source_faithful"] = False
                        elif self.conflict_fault == "classification":
                            row["correct_kind"] = "report_structure"
                        self.injected.append(copy.deepcopy(row))
                        self.injected_payloads.append(copy.deepcopy(payload))
                        self.injected_rounds.add(self.inventory_calls)
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        """Keep reviewer and all authored units on the actual requested wire."""
        value = super().answer(payload, native_schema)
        if "targets" in payload:
            self.repair_count += 1
            self.order.append("repair")
            for patch in value["patches"]:
                patch["text"] = CANDIDATE_TEXTS[self.repair_count - 1]
                patch["fact_card_refs"] = [self.case.by_eid["b"]]
        return value

    async def __call__(self, request):
        """Simulate unknown or partial responses before SDK usage accounting."""
        wire = json.loads(request.content)
        payload = json.loads(wire.get("input", wire.get("messages"))[1]["content"])
        if (
            self.conflict_fault == "partial"
            and "cited_materials" in payload
            and self.injected
        ):
            self.order.append("assessment_timeout")
            raise httpx.ReadTimeout(
                "Controlled incomplete later batch.", request=request
            )
        before = len(self.injected)
        response = await super().__call__(request)
        if self.conflict_fault == "unknown_usage" and len(self.injected) > before:
            body = response.json()
            body.pop("usage")
            return httpx.Response(200, json=body)
        return response


async def run_conflict_case(
    case, tmp_path, monkeypatch, *, selection="selected", **kwargs
):
    """Run real Synthesis, Reviewer, complete Evaluation and bounded transactions."""
    gateway = ConflictGateway(case, **kwargs)
    owner, _, _ = await owner_for(gateway)
    config = load_config()
    config.capacity = CapacityConfig(
        target_body_units=5,
        max_body_units=5,
        max_summary_units=1,
        initial_assertions=8,
        batch_size=2,
        max_assertions=8,
        max_assessment_batches=4,
    )
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.review_max_tokens = 4096
    config.eval.strict_batch_size = 2
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**options):
        owner._model = options["model"]
        owner._cost_budget = options["cost_budget"]
        owner._trace_hook = options["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {
                "paper_id": item["paper_id"],
                "claims": [item["text"]],
                "evidence_items": [item],
            }
            for item in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            {
                "__search_queries__": {"search_0": "offline relation-network fixture"},
                "search_0": list(case.ledger.values()),
                "dedup": list(case.ledger.values()),
                "relevance_gate": list(case.ledger.values()),
            },
            source_outcomes=[
                {
                    "task_id": "search_0",
                    "source": "arxiv",
                    "status": "success",
                    "result_count": 2,
                }
            ],
            started_at=1700000000,
        ),
        "fact_cards_sha256": digest(case.cards),
    }
    flags = (
        FLAGS
        if selection == "selected"
        else SOURCE_OWNER_FLAGS if selection == "v2" else {}
    )
    before = copy.deepcopy(bundle)
    plan = plan_frozen_budget(
        bundle, budget_policy="planned-budget-v7", capacity=config.capacity, **flags
    )
    assert plan["status"] == "admitted"
    try:
        report = await run_frozen(
            agent,
            bundle,
            "semantic-conflict-offline",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **flags,
        )
        assert bundle == before
        return report, gateway, agent._cost_budget.snapshot()
    finally:
        await agent.cleanup()


def assert_effect_preserves_original(details, gateway, code):
    """Require the typed host result to coexist with the exact received decision."""
    affected = [
        row for row in details["assertions"] if row.get("assessment_conflict_effect")
    ]
    assert len(affected) == 1
    row = affected[0]
    effect = row["assessment_conflict_effect"]
    assert row["status"] == "unverifiable"
    assert set(effect) == {
        "contract",
        "policy",
        "policy_identity",
        "code",
        "original_status",
        "original_reason_code",
        "decision_sha256",
        "binding_sha256",
    }
    assert (
        effect["contract"] == CONTRACT
        and effect["policy"] == POLICY
        and effect["code"] == code
    )
    original = gateway.injected[0]
    assert effect["original_status"] == original["status"]
    assert effect["original_reason_code"] == original["reason_code"]
    saved = [
        decision
        for batch in details["stages"]["batches"]
        for decision in batch["decisions"]
        if decision["claim_id"] == row["claim_id"]
    ]
    payload = gateway.injected_payloads[0]
    aliases = {}
    for alias, passage in payload["support_passages"].items():
        evidence, source, start, end = passage
        assert start == 0 and end == len(payload["original_supports"][source]["text"])
        aliases[alias] = evidence + ":full_original"
    received = Assessment.model_validate(
        {
            "units": [],
            "decisions": [{**original, "classification_valid": True}],
            "conflicts": [],
        }
    )
    canonical = _restore_batch_support_ids(received, aliases).decisions[0].model_dump()
    assert saved == [canonical]
    assert effect["decision_sha256"] == digest(canonical)
    assert details["unverified_count"] >= 1
    assert strict_acceptance(details)["status"] == "failed"
    assert details.get("score") is None
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [NEGATIVE, POSITIVE])
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_conflict_requires_full_diagnosis_then_two_shared_repairs_and_reassessment(
    case, tmp_path, monkeypatch, code, failing_candidates
):
    """Preserve both contradictory signals until a complete new candidate passes."""
    report, gateway, snapshot = await run_conflict_case(
        case, tmp_path, monkeypatch, code=code, failing_candidates=failing_candidates
    )
    state, journal, ledger = products(tmp_path)
    initial_metric = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]
    initial = initial_metric["details"]
    assert_effect_preserves_original(initial, gateway, code)
    assert initial_metric["skipped"] is True
    assert initial["coverage"]["complete"] is False
    assert initial["stages"]["global_check"] == "completed"
    assert all(batch["status"] == "completed" for batch in initial["stages"]["batches"])
    expected = min(2, failing_candidates + 1)
    assert gateway.repair_count == len(state["state"]["attempts"]) == expected
    assert gateway.inventory_calls == gateway.global_calls == expected + 1
    assert len(sent_with(gateway, "body_units")) == expected + 1
    assert gateway.order.index("global") < gateway.order.index("repair")
    assert len(journal["candidates"]) == expected
    assert not any(row["committed"] for row in journal["candidates"][:-1])
    assert journal["candidates"][-1]["committed"] is (failing_candidates < 2)
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    final_metric = report["evaluation"]["faithfulness"]
    final = final_metric["details"]
    if failing_candidates < 2:
        assert final["coverage"]["complete"] and final_metric["score"] == 1.0
        assert final["unverified_count"] == final["unsupported_count"] == 0
        assert not any(
            row.get("assessment_conflict_effect") for row in final["assertions"]
        )
    else:
        assert final_metric["skipped"] is True
        assert final["coverage"]["complete"] is False
        assert not any(text in report["survey"] for text in CANDIDATE_TEXTS)
    assert snapshot["usage_uncertain"] is False
    assert ledger["requests_sent"] == len(gateway.sent)
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
    progress = [
        json.loads(path.read_text())
        for path in (tmp_path / "evaluation").glob("*/progress.json")
    ]
    preserved = [
        json.loads(request["response"])
        for item in progress
        for request in item["progress"]["requests"]
        if "response" in request
    ]
    assert all(
        any(original in raw.get("decisions", []) for raw in preserved)
        for original in gateway.injected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["none", "v2"])
@pytest.mark.parametrize("code", [NEGATIVE, POSITIVE])
async def test_unselected_legacy_conflict_still_stops_without_repair(
    case, tmp_path, monkeypatch, selection, code
):
    """The new selection cannot reinterpret old None or V2 runtime scopes."""
    report, gateway, snapshot = await run_conflict_case(
        case, tmp_path, monkeypatch, selection=selection, code=code
    )
    assert not report["delivery"]["publishable"]
    assert gateway.injected and gateway.repair_count == 0
    details = report["evaluation"]["faithfulness"]["details"]
    assert "evaluation_decision_consistency_invalid" in json.dumps(details)
    assert not any(
        row.get("assessment_conflict_effect") for row in details.get("assertions", [])
    )
    assert not snapshot["usage_uncertain"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "bad_basis",
        "extra_aspect",
        "atomic",
        "fidelity",
        "classification",
        "unknown_usage",
        "partial",
        "global_incomplete",
    ],
)
async def test_conflict_does_not_authorize_repair_with_invalid_or_incomplete_evidence(
    case, tmp_path, monkeypatch, fault
):
    """Other failures retain their own veto instead of inheriting conflict eligibility."""
    report, gateway, snapshot = await run_conflict_case(
        case, tmp_path, monkeypatch, fault=fault
    )
    assert not report["delivery"]["publishable"]
    assert gateway.repair_count == 0
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    if fault in {"atomic", "fidelity", "classification"}:
        assert all(
            attempt.get("purpose") == "bounded-inventory-recovery-v2"
            for attempt in attempts
        )
    else:
        assert not attempts
    assert not sent_with(gateway, "targets")
    details = report["evaluation"]["faithfulness"]["details"]
    assert details.get("score") is None
    if fault in {"unknown_usage", "partial"}:
        assert snapshot["usage_uncertain"]
    assert not any(
        row.get("committed")
        for row in json.loads((tmp_path / "repair.json.writer-v2.json").read_text())[
            "candidates"
        ]
    )
