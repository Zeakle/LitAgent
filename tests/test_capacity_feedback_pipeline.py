"""Exercise changing capacity envelopes through the real frozen SDK pipeline.

Inventories in this module are synthetic transport fixtures, not scientific gold.
Only HTTP is replaced: allocation, receipts, feedback replay, reassessment and
automatic delivery all run through their production consumers.
"""

import copy
import json

import httpx
import pytest
from model_config_helpers import load_previous_default_config as load_config
from test_flat_writer_pipeline import FlatGateway
from test_guidance_wire_pipeline import assert_complete_model_wire
from test_native_responses import owner_for
from test_source_map_pipeline import products, sent_with
from test_support_receipts_pipeline import SELECTION
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
from litagent.quality import digest
from litagent.review_structure import build_methodology
from litagent.runner import LitAgent, RunPolicy
from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

FLAGS = {
    **SELECTION,
    "support_validation_revision": "host-support-receipts-v2",
    "repair_mapping_contract": "capacity-source-map-v3",
    "repair_guidance_contract": "host-structured-repair-v1",
    "repair_guidance_wire_revision": "host-unit-ref-guidance-v1",
}


class CapacityFeedbackGateway(FlatGateway):
    """Provide synthetic counts and controlled support failures over real HTTP."""

    def __init__(
        self, case, *, fail_second=False, unknown_second=False, repeat_wrong=False
    ):
        super().__init__(case)
        self.fail_second = fail_second
        self.unknown_second = unknown_second
        self.repeat_wrong = repeat_wrong
        self.repairs = 0
        self.inventories = []
        self.assessments = []
        self.globals = []
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                # The first full inventory exceeds the declared operational
                # ceiling. Actual choose_refinement must allocate 176/11.
                count = 188 if not self.inventories else 101
                groups = [g for g in value["fragments"].values() if g["assertions"]]
                wrong = {
                    index
                    for index, group in enumerate(groups)
                    if "Evo-ViT" in group["assertions"][0]["claim_text"]
                    and not self.repeat_wrong
                }
                # Keep one actual scientific failure, rather than turning the
                # same wrong sentence into fourteen synthetic repair issues.
                # All 101 inventory candidates still receive full assessment.
                quotient, remainder = divmod(
                    count - len(wrong), len(groups) - len(wrong)
                )
                regular = 0
                for index, group in enumerate(groups):
                    seed = copy.deepcopy(group["assertions"][0])
                    group["assertions"] = []
                    size = 1 if index in wrong else quotient + (regular < remainder)
                    regular += index not in wrong
                    for ordinal in range(size):
                        claim = copy.deepcopy(seed)
                        claim["claim_text"] += (
                            " Synthetic inventory scope "
                            + chr(97 + index)
                            + chr(97 + ordinal)
                            + "."
                        )
                        group["assertions"].append(claim)
                self.inventories.append(count)
            elif "cited_materials" in payload:
                self.assessments.append(copy.deepcopy(payload))
                claims = {row["claim_id"]: row for row in payload["assertions"]}
                for decision in value["decisions"]:
                    if "Evo-ViT" not in claims[decision["claim_id"]]["claim_text"]:
                        continue
                    decision.update(
                        status="unsupported",
                        reason_code="attribution_mismatch",
                        support_basis=[],
                        explanation="The cited EViT original cannot establish this Evo-ViT attribution.",
                    )
                    for aspect in decision["aspect_checks"]:
                        aspect.update(supported=False, support_ids=[])
            else:
                self.globals.append(copy.deepcopy(payload))
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        """Return source declarations, leaving all local indexes to the host."""
        if payload.get("cause") != "capacity":
            return super().answer(payload, native_schema)
        self.repairs += 1
        patches, deleted = [], []
        for tid, target in payload["targets"].items():
            section, paragraph, start = target["path"]
            originals = payload["current_body"][section][paragraph]["sentences"]
            for offset, uid in enumerate(target["original_unit_ids"]):
                index = start + offset
                if section == "methods" and index == 1:
                    deleted.append(
                        {"unit_id": uid, "reason": "Remove the exact redundant unit."}
                    )
                    continue
                unit = copy.deepcopy(originals[index])
                if section == "methods":
                    failed = self.repairs == 1 or self.fail_second
                    unit.update(
                        text=(
                            (
                                "Evo-ViT learns a comparison function."
                                if self.repairs == 1
                                else "Evo-ViT learns a comparison function in another setting."
                            )
                            if failed
                            else "Relation networks learn a comparison function."
                        ),
                        fact_card_refs=[self.case.by_eid["a" if failed else "b"]],
                    )
                patches.append({"target_id": tid, **unit, "source_unit_ids": [uid]})
        return {
            "outcome": "patched",
            "explanation": "Controlled source-map candidate; science is independently checked.",
            "patches": patches,
            "deleted_sources": deleted,
        }

    async def __call__(self, request):
        """An unknown second response keeps its actual reservation occupied."""
        body = json.loads(request.content)
        payload = json.loads(body.get("input", body.get("messages"))[1]["content"])
        response = await super().__call__(request)
        if (
            self.unknown_second
            and payload.get("cause") == "capacity"
            and self.repairs == 2
        ):
            value = response.json()
            value.pop("usage")
            return httpx.Response(200, json=value)
        return response


async def run_changed_capacity(case, tmp_path, monkeypatch, gateway):
    """Keep the frozen runner intact without changing the old max140 helper."""
    case.body["methods"][0]["sentences"].append(
        copy.deepcopy(case.body["methods"][0]["sentences"][0])
    )
    owner, _, _ = await owner_for(gateway)
    config = load_config()
    config.capacity = CapacityConfig(max_assertions=176)
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
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
    papers = [{"paper_id": "evit"}, {"paper_id": "relation"}]
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "synthetic-capacity-feedback-fixture",
        "requirements": case.packet["requirements"],
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            {
                "__search_queries__": {"search_0": "offline relation-network fixture"},
                "search_0": papers,
                "dedup": papers,
                "relevance_gate": papers,
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
    plan = plan_frozen_budget(
        bundle, budget_policy="planned-budget-v7", capacity=config.capacity, **FLAGS
    )
    assert plan["status"] == "admitted"
    assert plan["envelope"]["model_assertions"] == 128
    try:
        return await run_frozen(
            agent,
            bundle,
            "synthetic-changing-capacity",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **FLAGS,
        )
    finally:
        await agent.cleanup()


def assert_changed_capacity_path(case, gateway, journal, ledger):
    """Prove that the real prior receipt has a different envelope from current."""
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == gateway.repairs == 2
    assert_complete_model_wire(repairs[0], prior=False)
    assert_complete_model_wire(repairs[1], prior=True)
    for request in repairs:
        current = request["payload"]["current_body"]
        assert current == case.body
        assert "Evo-ViT" not in json.dumps(current)
        for source in case.packet["sources"].values():
            assert source["source_quote"] in json.dumps(request["request"])
    first = journal["candidates"][0]
    assert first["committed"] is False
    assert first["writer_document"]["budget_envelope"]["model_assertions"] == 176
    assert first["evaluation"]["faithfulness"]["details"]["unsupported_count"] > 0
    witness = journal["calls"]["repair_1"]["guidance_wire_record"][
        "prior_replay_witness"
    ]
    assert witness["document"]["budget_envelope"]["model_assertions"] == 128
    assert (
        witness["prior"]["writer_document"]["budget_envelope"]["model_assertions"]
        == 176
    )
    assert witness["prior"]["report_sha256"] == first["report_sha256"]
    replay = journal["calls"]["repair_1"]["guidance_wire_record"]
    diagnostic = replay["arguments"]["capacity_diagnostic"]
    current_issues = repairs[1]["payload"]["repair_guidance"]["current"]["issues"]
    capacity_issues = [
        row[1]
        for row in current_issues
        if row[0] == 1 and row[1]["category"] == "capacity"
    ]
    assert len(capacity_issues) == 1
    current_fields = capacity_issues[0]["host_fields"]
    for key in ("target_assertions", "target_batches", "allocation_identity"):
        assert current_fields[key] == diagnostic[key]
    assert diagnostic["target_assertions"] == 128
    assert diagnostic["target_batches"] == 8
    assert (
        diagnostic["allocation_identity"]
        == ledger["capacity_revisions"][1]["allocation"]["identity"]
    )
    local_capacity = [
        row["evaluation_details"]["stages"]["capacity"]
        for row in journal["structured_capacity_inputs"].values()
    ]
    assert any(
        row["target_assertions"] == 176 and row["actual_assertions"] == 188
        for row in local_capacity
    )
    revisions = ledger["capacity_revisions"]
    assert revisions[0]["arguments"]["mode"] == "refine"
    assert revisions[0]["arguments"]["assertions"] == 176
    assert revisions[0]["arguments"]["batches"] == 11
    assert ledger["plan"]["envelope"]["model_assertions"] == 128
    assert ledger["requests_sent"] == len(gateway.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_second", [False, True])
async def test_capacity_candidate_failure_replays_against_original_envelope(
    case, tmp_path, monkeypatch, fail_second
):
    """Both edits receive full review; only a fully accepted candidate commits."""
    gateway = CapacityFeedbackGateway(case, fail_second=fail_second)
    report = await run_changed_capacity(case, tmp_path, monkeypatch, gateway)
    state, journal, ledger = products(tmp_path)
    assert_changed_capacity_path(case, gateway, journal, ledger)
    assert len(state["state"]["attempts"]) == 2
    assert gateway.inventories == [188, 101, 101]
    assert len(sent_with(gateway, "body_units")) == 3
    assert len(gateway.assessments) == 14
    assert len(gateway.globals) == 2
    assert ledger["usage_uncertain"] is False
    assert all(e["status"] == "settled" for e in ledger["entries"].values())
    assert report["delivery"]["publishable"] is not fail_second
    if fail_second:
        assert not any(c["committed"] for c in journal["candidates"])
        assert report["metadata"]["writer_document"]["body"] == case.body
    else:
        details = report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"] is True
        assert details["supported_count"] == details["factual_claim_count"]
        assert details["unsupported_count"] == details["unverified_count"] == 0
        assert "Evo-ViT" not in report["survey"]
        assert journal["candidates"][-1]["committed"] is True


@pytest.mark.asyncio
async def test_unknown_second_capacity_repair_holds_usage_and_keeps_original(
    case, tmp_path, monkeypatch
):
    """The second unknown response is neither committed nor silently resent."""
    gateway = CapacityFeedbackGateway(case, unknown_second=True)
    report = await run_changed_capacity(case, tmp_path, monkeypatch, gateway)
    state, journal, ledger = products(tmp_path)
    assert_changed_capacity_path(case, gateway, journal, ledger)
    assert len(state["state"]["attempts"]) == 2
    assert gateway.inventories == [188, 101]
    assert len(sent_with(gateway, "body_units")) == 2
    assert ledger["usage_uncertain"] is True
    assert ledger["stop_reason"] == "strict_usage_unknown"
    occupied = [e for e in ledger["entries"].values() if e["status"] != "settled"]
    assert len(occupied) == 1
    assert occupied[0]["stage"] == "repair"
    assert occupied[0]["reserved_tokens"] > 0
    assert not any(c["committed"] for c in journal["candidates"])
    assert report["delivery"]["publishable"] is False
    assert report["metadata"]["writer_document"]["body"] == case.body


@pytest.mark.asyncio
async def test_repeated_negative_inventory_keeps_full_guidance_and_refuses_overflow(
    case, tmp_path, monkeypatch
):
    """Fourteen complete negative issues must not be shortened to obtain a retry."""
    gateway = CapacityFeedbackGateway(case, repeat_wrong=True)
    report = await run_changed_capacity(case, tmp_path, monkeypatch, gateway)
    state, journal, ledger = products(tmp_path)
    assert gateway.inventories == [188, 101]
    assert gateway.repairs == len(sent_with(gateway, "targets")) == 1
    assert len(state["state"]["attempts"]) == 1
    assert len(gateway.assessments) == 7
    assert len(gateway.globals) == 1
    candidate = journal["candidates"][0]
    details = candidate["evaluation"]["faithfulness"]["details"]
    assert candidate["committed"] is False
    assert details["coverage"]["complete"] is True
    assert details["coverage"]["assertions"] == 101
    assert details["unsupported_count"] == 14
    failure = journal["failure"]
    assert failure["code"] == "writer_guidance_unrepairable_capacity"
    assert failure["diagnostics"][0]["actual"] > 12288
    assert failure["diagnostics"][0]["maximum"] == 12288
    assert ledger["usage_uncertain"] is False
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
    assert ledger["requests_sent"] == len(gateway.sent)
    assert report["delivery"]["publishable"] is False
    assert report["metadata"]["writer_document"]["body"] == case.body
