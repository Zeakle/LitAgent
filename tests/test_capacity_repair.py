"""Exercise explicit capacity edit accounting and the real writer transport."""

import copy
import json

import pytest
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_repair import capacity_targets, validate_edit_records
from litagent.writing_v2 import WriterError, writer_budget_plan


class CapacityGateway(FlatGateway):
    """Remove only a duplicated unit, recording all other originals explicitly."""

    def answer(self, payload, native_schema):
        if payload.get("cause") != "capacity":
            return super().answer(payload, native_schema)
        patches, records = [], []
        for tid, target in payload["targets"].items():
            section, pi, start = target["path"]
            original = payload["current_body"][section][pi]["sentences"]
            replacement = 0
            for i in range(start, target["end"]):
                removed = section == "methods" and i == 1
                records.append(
                    {
                        "unit_id": f"writer:{section}:{pi}:{i}",
                        "action": "removed" if removed else "retained",
                        "reason": (
                            "Repeated finding."
                            if removed
                            else "Core source-bound content."
                        ),
                        "replacement_indexes": [] if removed else [replacement],
                    }
                )
                if not removed:
                    patches.append({"target_id": tid, **copy.deepcopy(original[i])})
                    replacement += 1
        result = {
            "outcome": "patched",
            "explanation": "Removed the exact redundant unit.",
            "patches": patches,
            "removed_targets": [],
            "edit_records": records,
        }
        if "removed_targets" not in native_schema["properties"]:
            result.pop("removed_targets")
            for record in records:
                record.pop("action")
        return result


@pytest.mark.asyncio
async def test_capacity_patch_uses_real_sdk_and_shared_attempt_counter(case, tmp_path):
    case.body["methods"][0]["sentences"].append(
        copy.deepcopy(case.body["methods"][0]["sentences"][0])
    )
    gateway = CapacityGateway(case)
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v6"
    )
    budget.activate_plan(plan)
    session = make_session(owner, case, tmp_path)
    targets = capacity_targets(
        case.body, {"reason": "capacity_remaining_workflow_unaffordable"}
    )
    original = copy.deepcopy(case.body)
    try:
        candidate = await session.patch(
            original, targets, cause="capacity", claim_count=144
        )
        assert original == case.body
        assert len(candidate["methods"][0]["sentences"]) == 1
        assert len(session.state["attempts"]) == 1
        assert session.state["attempts"][0]["cause"] == "capacity"
        assert len(session.state["attempts"][0]["edit_records"]) == 6
        assert session.state["attempts"][0]["edit_records"][3]["action"] == "removed"
        assert len(gateway.sent) == 1
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "fault", ["omitted", "false_retention", "silent_removal", "new_unaccounted"]
)
def test_edit_records_cannot_hide_removed_or_new_content(case, fault):
    targets = capacity_targets(case.body, {})
    patches = [
        {"target_id": tid, "sentences": copy.deepcopy(t["original"])}
        for tid, t in targets.items()
    ]
    records = [
        {
            "unit_id": f"writer:{t['path'][0]}:{t['path'][1]}:0",
            "action": "retained",
            "reason": "Preserved.",
            "replacement_indexes": [0],
        }
        for t in targets.values()
    ]
    result = {"patches": patches, "edit_records": records}
    if fault == "omitted":
        records.pop()
    elif fault == "false_retention":
        patches[0]["sentences"][0]["text"] = "An unrelated unsupported fact."
    elif fault == "silent_removal":
        records[0]["action"] = "removed"
    else:
        patches[0]["sentences"].append(copy.deepcopy(patches[0]["sentences"][0]))
    with pytest.raises(WriterError, match="writer_capacity"):
        validate_edit_records(result, targets)


class OverflowInventoryGateway(CapacityGateway):
    """Return a full oversized first inventory, then ordinary independent ones."""

    def __init__(self, case, **kwargs):
        super().__init__(case, **kwargs)
        self.inventory_calls = 0
        original = self.evaluator.chat

        async def chat(messages, **options):
            response = await original(messages, **options)
            if "allowed_citation_aliases" not in json.loads(messages[1]["content"]):
                return response
            self.inventory_calls += 1
            if self.inventory_calls == 1:
                value = json.loads(response.content)
                groups = [g for g in value["fragments"].values() if g["assertions"]]
                count, extra = divmod(144, len(groups))
                for index, group in enumerate(groups):
                    original_assertion = group["assertions"][0]
                    group["assertions"] = []
                    for j in range(count + (index < extra)):
                        candidate = copy.deepcopy(original_assertion)
                        # These are explicitly unverified extraction candidates;
                        # exact duplicate assertions are a separate hard error.
                        candidate["claim_text"] += (
                            " Candidate scope " + chr(97 + index) + chr(97 + j) + "."
                        )
                        group["assertions"].append(candidate)
                response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


async def run_capacity_case(
    case,
    tmp_path,
    monkeypatch,
    *,
    unsupported=False,
    budget_policy="planned-budget-v7",
    gateway=None,
    allow_uncertain=False,
):
    """Exercise the real frozen consumers with a bounded external gateway."""
    from model_config_helpers import load_previous_default_config as load_config

    from litagent.capacity import CapacityConfig
    from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
    from litagent.quality import digest
    from litagent.review_structure import build_methodology
    from litagent.runner import LitAgent, RunPolicy
    from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

    case.body["methods"][0]["sentences"].append(
        copy.deepcopy(case.body["methods"][0]["sentences"][0])
    )
    gateway = gateway or OverflowInventoryGateway(case, unsupported=unsupported)
    owner, _, _ = await owner_for(gateway)
    config = load_config()
    config.capacity = CapacityConfig(max_assertions=140)
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
            {
                "__search_queries__": {"search_0": "offline relation-network fixture"},
                "search_0": [{"paper_id": "evit"}, {"paper_id": "relation"}],
                "dedup": [{"paper_id": "evit"}, {"paper_id": "relation"}],
                "relevance_gate": [{"paper_id": "evit"}, {"paper_id": "relation"}],
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
        bundle, budget_policy=budget_policy, capacity=config.capacity
    )
    try:
        result = await run_frozen(
            agent,
            bundle,
            "capacity-fallback",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy=budget_policy,
            expected_budget_plan=plan,
        )
        if not allow_uncertain:
            assert not agent._cost_budget.snapshot()["usage_uncertain"]
        return result, gateway
    finally:
        await agent.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize("budget_policy", ["planned-budget-v6", "planned-budget-v7"])
async def test_capacity_fallback_complete_worker_chain_and_atomic_delivery(
    case, tmp_path, monkeypatch, unsupported, budget_policy
):
    """Keep both legacy and round budgets subject to full scientific gates."""
    result, gateway = await run_capacity_case(
        case,
        tmp_path,
        monkeypatch,
        unsupported=unsupported,
        budget_policy=budget_policy,
    )
    state = result["metadata"]["targeted_repair_state"]
    assert 1 <= len(state["attempts"]) <= 2
    assert state["attempts"][0]["cause"] == "capacity"
    assert result["delivery"]["publishable"] is not unsupported
    if not unsupported:
        details = result["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["unsupported_count"] == details["unverified_count"] == 0
    else:
        assert result["metadata"]["evidence_rewrite_attempts"][0]["committed"] is False
        assert gateway.inventory_calls == 2


def test_merged_originals_are_rewritten_and_removed_requires_no_index(case):
    """Replay 026's removed-with-index pattern without accepting its semantics."""
    from litagent.capacity_repair import ACTION_CONTRACT, ACTION_INSTRUCTION
    from litagent.writing_v2 import RANGE_WRITING_PLAN, repair_request_input

    body = copy.deepcopy(case.body)
    body["methods"][0]["sentences"].append(
        copy.deepcopy(body["methods"][0]["sentences"][0])
    )
    targets = {"r1": capacity_targets(body, {})["r3"]}
    original = targets["r1"]["original"]
    result = {
        "patches": [{"target_id": "r1", "sentences": [original[0]]}],
        "edit_records": [
            {
                "unit_id": "writer:methods:0:0",
                "action": "rewritten",
                "reason": "Combined supported premises.",
                "replacement_indexes": [0],
            },
            {
                "unit_id": "writer:methods:0:1",
                "action": "rewritten",
                "reason": "Merged into the same replacement.",
                "replacement_indexes": [0],
            },
        ],
    }
    validate_edit_records(result, targets)
    result["edit_records"][1]["action"] = "removed"
    with pytest.raises(WriterError, match="writer_capacity_removal_invalid"):
        validate_edit_records(result, targets)
    envelope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v6"
    )["envelope"]
    system, _, _ = repair_request_input(
        case.packet,
        body,
        targets,
        case.aliases,
        cause="capacity",
        previous_failure=None,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=envelope,
        wire_contract="flat-writer-v1",
    )
    assert ACTION_CONTRACT in system and ACTION_INSTRUCTION in system
    assert "replacement_indexes MUST be []" in system
