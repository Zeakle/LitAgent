"""Separate complete request capacity from malformed protocols before any send."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from test_adaptive_capacity import settle_initial
from test_joint_round_escrow import _joint_plan
from test_native_responses import envelope, owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_rounds import plan_round, request_proof
from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.output_capacity import configured_output_tokens
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.safety.planned_budget import PlannedBudget


def expanded_owner(case):
    """Retain both future repairs and the shipped expanded output reservations."""
    outputs = configured_output_tokens(load_config("config/expanded-capacity.yaml"))
    _, plan = _joint_plan(case, output_tokens=outputs)
    owner = PlannedBudget(plan)
    settle_initial(owner, maximum=True)
    return owner


def complete_requests(owner, stage, amount):
    """Measure actual serialized messages rather than inventing token receipts."""
    proofs = []
    for name in ("assessment", "global"):
        output = owner.plan["stages"][name]["output_tokens"]
        messages = [{"role": "user", "content": "complete fixture original"}]
        if name == stage:
            padding = amount - strict_request_reservation(messages, output)
            assert padding >= 0
            messages[0]["content"] += "x" * padding
        proofs.append(request_proof(name, messages, output, None))
    return proofs


@pytest.mark.parametrize("stage", ["assessment", "global"])
@pytest.mark.parametrize("boundary", ["stage", "context"])
def test_valid_oversize_request_is_capacity_without_authorizing_a_send(
    case, stage, boundary
):
    owner = expanded_owner(case)
    cap = (
        owner.plan["stages"][stage]["request_cap"]
        if boundary == "stage"
        else owner.plan["context_reservation_limit"]
    )
    requests = complete_requests(owner, stage, cap + 1)
    before = owner.snapshot()
    with pytest.raises(ValueError, match="^capacity_round_request_exceeded$") as caught:
        plan_round(
            owner,
            assertions=1,
            batches=1,
            requests=requests,
            report_sha256="a" * 64,
        )
    assert caught.value.diagnostics[0]["stage"] == stage
    assert caught.value.diagnostics[0]["reserved_tokens"] == cap + 1
    assert owner.snapshot() == before
    assert CostBudget.from_snapshot(before).snapshot() == before


@pytest.mark.asyncio
async def test_real_056_complete_inputs_reach_bounded_capacity_edit_without_payment(
    tmp_path, monkeypatch
):
    """Replay actual provider bytes; replace HTTP only, never the real consumers."""
    from litagent.frozen_writing import plan_frozen_budget, run_frozen
    from litagent.runner import LitAgent, RunPolicy
    from litagent.safety.planned_accounting import validate_accounting_snapshot
    from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

    root = Path(__file__).parents[1]
    suite = root / "artifacts/acceptance/frozen-writer-expanded-20261104-056"
    archive = root / "artifacts/runs" / (suite.name + "-few-shot-metric.json")
    journal_path = (
        root
        / "artifacts/targeted-repairs"
        / (suite.name + "-few-shot-metric.json.writer-v2.json")
    )
    if not all(p.exists() for p in (suite / "manifest.json", archive, journal_path)):
        pytest.skip(
            "Private 056 responses remain local; portable boundary tests still run"
        )
    protected = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            suite / "manifest.json",
            suite / "state.json",
            archive,
            journal_path,
        )
    }
    frozen = json.loads((suite / "manifest.json").read_text())
    bundle = copy.deepcopy(frozen["cases"][0]["bundle"])
    original = json.loads(archive.read_text())["report"]
    journal = json.loads(journal_path.read_text())
    inventory = original["evaluation"]["faithfulness"]["details"]["stages"]["requests"][
        0
    ]
    sent = []

    async def peer(request):
        """Use recorded full outputs, then explicitly decline one authorized edit."""
        body = json.loads(request.content)
        messages = body.get("input", body.get("messages"))
        payload = json.loads(messages[1]["content"])
        sent.append(payload)
        if request.url.path.endswith("/responses"):
            section = payload.get("requested_body_section")
            key = (
                "body:" + section
                if section
                else "summary_initial" if "body_units" in payload else "reviewer"
            )
            if payload.get("cause") == "capacity":
                text = json.dumps(
                    {
                        "outcome": "declined",
                        "explanation": "Offline refusal; no candidate is scientifically accepted.",
                        "patches": [],
                        "deleted_sources": [],
                    }
                )
                usage = {"prompt_tokens": 20, "completion_tokens": 100}
            else:
                text, usage = (
                    journal["calls"][key]["raw"],
                    journal["calls"][key]["usage"],
                )
            response = envelope(text, model=body["model"])
            response["output"] = response["output"][1:]
            response["usage"] = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            return httpx.Response(200, json=response)
        assert "allowed_citation_aliases" in payload
        return httpx.Response(
            200,
            json={
                "id": "offline-056-inventory",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": inventory["response"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": inventory["usage"],
            },
        )

    owner, _, _ = await owner_for(peer)
    config = load_config(str(root / "config/expanded-capacity.yaml"))
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
    flags = profile_flags(PROFILE)
    plan = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        output_tokens=configured_output_tokens(config),
        **flags,
    )
    try:
        report = await run_frozen(
            agent,
            bundle,
            "offline-complete-056-resource-recovery",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **flags,
        )
        details = report["metadata"]["evaluation_history"]["initial_evaluation"][
            "faithfulness"
        ]["details"]
        assert details["skipped_reason"] == "evaluation_capacity_refinement_required"
        capacity = details["stages"]["capacity"]
        assert capacity["actual_assertions"] == 172
        assert capacity["global_reserved_tokens"] == 122077
        assert capacity["status"] == "needs_refinement"
        assert "122077 > 120000" in capacity["explanation"]
        assert not report["delivery"]["publishable"]
        edits = [p for p in sent if p.get("cause") == "capacity"]
        assert len(edits) == 1
        assert all(
            token in json.dumps(edits[0])
            for token in ("global_request_reservation", "122077", "120000")
        )
        assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
        assert not any("cited_materials" in p for p in sent)
        validate_accounting_snapshot(agent._cost_budget.snapshot(), expected_plan=plan)
    finally:
        await agent.cleanup()
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == sha
        for p, sha in protected.items()
    )


def test_global_exactly_at_locked_cap_remains_legal(case):
    owner = expanded_owner(case)
    cap = owner.plan["stages"]["global"]["request_cap"]
    allocation = plan_round(
        owner,
        assertions=1,
        batches=1,
        requests=complete_requests(owner, "global", cap),
        report_sha256="a" * 64,
    )
    assert allocation["round"]["remaining_repairs"] == 2
    assert allocation["round"]["requests"][-1]["reserved_tokens"] == cap
    assert not owner._stage_entries("assessment")


@pytest.mark.parametrize("fault", ["hash", "output", "bool", "extra", "stage"])
def test_capacity_cannot_hide_a_later_malformed_proof(case, fault):
    owner = expanded_owner(case)
    proofs = complete_requests(
        owner, "assessment", owner.plan["stages"]["assessment"]["request_cap"] + 1
    )
    bad = proofs[-1]
    if fault == "hash":
        bad["wire_sha256"] = "not-a-host-hash"
    elif fault == "output":
        bad["output_tokens"] -= 1
    elif fault == "bool":
        bad["reserved_tokens"] = True
    elif fault == "extra":
        bad["client_override"] = True
    else:
        bad["stage"] = "assessment"
    before = copy.deepcopy(owner.snapshot())
    with pytest.raises(ValueError, match="^capacity_round_requests_invalid$"):
        plan_round(
            owner,
            assertions=1,
            batches=1,
            requests=proofs,
            report_sha256="a" * 64,
        )
    assert owner.snapshot() == before
