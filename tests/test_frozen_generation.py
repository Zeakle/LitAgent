"""Exercise immutable one-shot generation and cumulative accounting offline."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_strict_quality import draft, to_native

from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import LLMResponse
from litagent.quality import build_fact_cards, digest
from litagent.runner import LitAgent
from scripts.run_frozen_generation import (
    FrozenSelection,
    file_sha,
    prepare,
    run_trial,
    settle,
)


def prepared(tmp_path, *, source_metadata=None):
    """Freeze real card/selection shapes beside an intact paid-history receipt."""
    root = tmp_path / "trial"
    root.mkdir()
    evidence = {
        "a": {
            "evidence_id": "a",
            "paper_id": "evit",
            "text": "EViT improves speed.",
            "supporting_text": "EViT improves speed by 50% on DeiT-S.",
            "chunk_key": "abstract",
        }
    }
    if source_metadata:
        evidence["a"]["source_locator"] = "paper=evit;chunk=abstract"
    cards = build_fact_cards(evidence)
    selection = EvidenceSelection(
        candidate_count=1,
        selected_items=evidence,
        section_evidence_ids={
            key: ["a"]
            for key in (
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            )
        },
        method_by_section={
            key: "lexical_fallback"
            for key in (
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            )
        },
        omitted_count=0,
        estimated_tokens=10,
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "run_id": "previous",
                "query": "EViT",
                "status": "completed",
                "usage": {"total_tokens": 100},
                "events": [
                    {
                        "event": "llm.start",
                        "data": {"operation_id": "old-request", "task_id": "extract"},
                    },
                    {
                        "event": "llm.complete",
                        "data": {
                            "operation_id": "old-request",
                            "task_id": "extract",
                            "prompt_tokens": 99,
                            "completion_tokens": 1,
                            "total_tokens": 100,
                            "usage_available": True,
                        },
                    },
                ],
                "nodes": {
                    "worker:extract": {
                        "kind": "worker",
                        "task_id": "extract",
                        "output": [
                            {
                                "paper_id": "evit",
                                "claims": ["EViT improves speed."],
                                "evidence_items": list(evidence.values()),
                            }
                        ],
                    }
                },
            }
        )
    )
    if source_metadata:
        saved = json.loads(source.read_text())
        saved["nodes"]["worker:dedup"] = {
            "kind": "worker",
            "task_id": "dedup",
            "output": [{"paper_id": "evit", **source_metadata}],
        }
        source.write_text(json.dumps(saved))
    inputs = {
        "question": "EViT",
        "source_artifact": str(source),
        "source_artifact_sha256": file_sha(source),
        "source_run_id": "previous",
        "fact_cards": cards,
        "fact_cards_sha256": digest(cards),
        "evidence_selection": selection.to_dict(),
        "quality_policy": "strict_auto_v1",
        "report_language": "zh-CN",
        "review_structure": "academic_v2",
        "coverage_requirements": [],
        "review_methodology": {},
        "benchmark_labels_in_generation": False,
    }
    (root / "input-snapshot.json").write_text(json.dumps(inputs))
    history = tmp_path / "history.json"
    costs = {
        "survey": {"tokens": 100, "requests": 1, "usage_complete": True},
        **{
            s: {"tokens": 0, "requests": 0, "usage_complete": True}
            for s in ("judge", "jev")
        },
    }
    history.write_text(
        json.dumps(
            {
                "status": "stopped",
                "observations": [
                    {
                        "costs": costs,
                        "run_id": "previous",
                        "artifact_path": str(source),
                        "artifact_sha256": "sha256:" + file_sha(source),
                    }
                ],
                "budgets": {
                    s: {"charged": c["tokens"], "requests": c["requests"]}
                    for s, c in costs.items()
                },
            }
        )
    )
    grant = tmp_path / "grant.json"
    grant.write_text(
        json.dumps(
            {"actor": "user", "text": "授权新18轮，后续不要我审批，我全都默认授权"}
        )
    )
    (root / "protected-before.json").write_text(
        json.dumps({"hashes": {str(source): file_sha(source)}})
    )
    config = load_config()
    plan = prepare(root, config, "new-trial", history, grant)
    return root, config, plan, inputs, history, grant


@pytest.mark.asyncio
async def test_dry_run_has_no_transport_or_checkpoint(tmp_path):
    root, config, plan, *_ = prepared(tmp_path)
    factory = Mock(side_effect=AssertionError("No paid transport"))
    result = await run_trial(root, config, plan, client_factory=factory)
    assert result["paid_calls"] == 0
    assert not factory.called and not (root / "checkpoint.json").exists()


@pytest.mark.asyncio
async def test_started_attempt_never_replays_or_refunds(tmp_path):
    root, config, plan, *_ = prepared(tmp_path)
    reserved = settle(plan, {s: {} for s in plan["limits"]})
    (root / "checkpoint.json").write_text(
        json.dumps(
            {
                "binding": plan["binding"],
                "status": "started_no_replay",
                "cumulative_reserved": reserved,
            }
        )
    )
    factory = Mock(side_effect=AssertionError("No repeat transport"))
    result = await run_trial(root, config, plan, live=True, client_factory=factory)
    assert result["status"] == "interrupted"
    assert result["cumulative_reserved"]["survey"]["charged"] == 500100
    assert not factory.called


@pytest.mark.asyncio
async def test_changed_input_refused_before_transport(tmp_path):
    root, config, plan, inputs, *_ = prepared(tmp_path)
    inputs["question"] = "another question"
    (root / "input-snapshot.json").write_text(json.dumps(inputs))
    factory = Mock()
    with pytest.raises(ValueError, match="identity_changed"):
        await run_trial(root, config, plan, live=True, client_factory=factory)
    assert not factory.called


@pytest.mark.asyncio
async def test_changed_budget_config_refused_before_transport(tmp_path):
    root, config, plan, *_ = prepared(tmp_path)
    config.eval.strict_max_claims = 64
    factory = Mock()
    with pytest.raises(ValueError, match="configuration_changed"):
        await run_trial(root, config, plan, live=True, client_factory=factory)
    assert not factory.called


@pytest.mark.asyncio
async def test_frozen_selector_refuses_wrong_paper_and_keeps_original(tmp_path):
    *_, inputs, history, grant = prepared(tmp_path)
    selected = inputs["evidence_selection"]["selected_items"]
    changed = copy.deepcopy(selected)
    changed["a"]["paper_id"] = "evo-vit"
    with pytest.raises(ValueError, match="selection_changed"):
        await FrozenSelection(inputs).select("EViT", changed)
    actual = await FrozenSelection(inputs).select("EViT", selected)
    assert actual.selected_items == selected


@pytest.mark.parametrize("mode", ["unknown", "mismatch", "boolean"])
def test_historical_unknown_or_inconsistent_usage_cannot_be_reset(tmp_path, mode):
    root, config, plan, inputs, history, grant = prepared(tmp_path)
    old = json.loads(history.read_text())
    cost = old["observations"][0]["costs"]["survey"]
    cost["usage_complete" if mode == "unknown" else "tokens"] = (
        False if mode == "unknown" else (True if mode == "boolean" else 99)
    )
    history.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="history_usage_invalid"):
        prepare(root, config, "another", history, grant)


def test_unknown_usage_retains_full_allowance_and_prior_charge(tmp_path):
    root, config, plan, *_ = prepared(tmp_path)
    actual = settle(plan, {s: {} for s in plan["limits"]})
    assert actual["survey"]["charged"] == 500100
    assert actual["judge"]["charged"] == 100000
    assert actual["jev"]["charged"] == 200000
    assert actual["survey"]["usage_complete"] is False


@pytest.mark.parametrize(
    "cost",
    [
        {"tokens": 500001, "requests": 1, "usage_complete": True},
        {"tokens": 1, "requests": 65, "usage_complete": True},
        {"tokens": 1, "requests": True, "usage_complete": True},
    ],
)
def test_settlement_rejects_over_limit_or_invalid_types(tmp_path, cost):
    root, config, plan, *_ = prepared(tmp_path)
    costs = {
        s: {"tokens": 0, "requests": 0, "usage_complete": True} for s in plan["limits"]
    }
    costs["survey"] = cost
    with pytest.raises(ValueError, match="budget_exceeded|usage_invalid"):
        settle(plan, costs)


@pytest.mark.asyncio
async def test_real_synthesis_quote_failure_stops_before_evaluation_and_auditors(
    tmp_path, monkeypatch
):
    root, config, plan, inputs, *_ = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    payload = to_native(draft(inputs["fact_cards"], "zh-CN"), inputs["fact_cards"])
    payload["sections"]["methods"]["first_block"]["first_sentence"][
        "support_quotes"
    ] = ["Invented source clause"]
    client = SimpleNamespace(
        _cost_budget=None, _model=config.llm.model, close=AsyncMock()
    )

    async def chat(*args, **kwargs):
        client._cost_budget.begin_request()
        client.hook(
            "llm.start",
            {
                "operation_id": "test-op",
                "task_id": "adversarial_review.synthesis.r1",
                "messages": args[0],
            },
        )
        client._cost_budget.record({"prompt_tokens": 9, "completion_tokens": 1})
        client.hook(
            "llm.complete",
            {
                "operation_id": "test-op",
                "task_id": "adversarial_review.synthesis.r1",
                "content": json.dumps(payload),
                "prompt_tokens": 9,
                "completion_tokens": 1,
                "total_tokens": 10,
                "usage_available": True,
            },
        )
        return LLMResponse(
            json.dumps(payload), usage={"prompt_tokens": 9, "completion_tokens": 1}
        )

    client.chat = AsyncMock(side_effect=chat)

    def factory(**kwargs):
        client._cost_budget = kwargs["cost_budget"]
        client.hook = kwargs["trace_hook"]
        return client

    service = SimpleNamespace(
        capabilities=lambda: {"judge_available": True, "jev_available": True},
        judge_config=lambda: SimpleNamespace(
            base_url=config.llm.base_url, model=config.llm.model
        ),
        submit=AsyncMock(),
        shutdown=AsyncMock(),
    )
    result = await run_trial(
        root,
        config,
        plan,
        live=True,
        client_factory=factory,
        service_factory=lambda *args, **kwargs: service,
    )
    assert result["status"] == "failed"
    assert result["costs"]["survey"]["tokens"] == 10
    assert result["cumulative_after"]["survey"]["charged"] == 110
    assert client.chat.await_count == 1 and not service.submit.called
    artifact = json.loads((tmp_path / "artifacts/runs/new-trial.json").read_text())
    assert artifact["status"] == "failed" and artifact["report"] is None
    assert artifact["events"][-1]["event"] == "survey.error"
    assert "Invented source clause" in (root / "events.jsonl").read_text()


@pytest.mark.asyncio
async def test_compiled_draft_uses_strict_identity_and_unverified_cannot_audit(
    tmp_path, monkeypatch
):
    root, config, plan, inputs, *_ = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    payload = to_native(draft(inputs["fact_cards"], "zh-CN"), inputs["fact_cards"])
    client = SimpleNamespace(
        _cost_budget=None, _model=config.llm.model, close=AsyncMock()
    )

    async def chat(*args, **kwargs):
        client._cost_budget.begin_request()
        client._cost_budget.record({"prompt_tokens": 9, "completion_tokens": 1})
        return LLMResponse(
            json.dumps(payload), usage={"prompt_tokens": 9, "completion_tokens": 1}
        )

    client.chat = AsyncMock(side_effect=chat)

    def factory(**kwargs):
        client._cost_budget = kwargs["cost_budget"]
        return client

    def worker_factory(llm, synthesis, reviewer, **kwargs):
        async def execute(task):
            output = await synthesis.execute(task)
            return {
                **output,
                "final_draft": output["draft"],
                "report_language": "zh-CN",
                "accepted": False,
                "rounds": [],
                "total_rounds": 0,
                "final_score": 0,
            }

        return SimpleNamespace(execute=execute)

    service = SimpleNamespace(
        capabilities=lambda: {"judge_available": True, "jev_available": True},
        judge_config=lambda: SimpleNamespace(
            base_url=config.llm.base_url, model=config.llm.model
        ),
        submit=AsyncMock(),
        shutdown=AsyncMock(),
    )
    monkeypatch.setattr(
        LitAgent,
        "_evaluate",
        AsyncMock(
            return_value={
                metric: {
                    "score": 0,
                    "passed": True,
                    "skipped": True,
                    "details": {"skipped_reason": "test_semantics_not_verified"},
                }
                for metric in (
                    "citation_accuracy",
                    "internal_consistency",
                    "faithfulness",
                )
            }
        ),
    )
    state = await run_trial(
        root,
        config,
        plan,
        live=True,
        client_factory=factory,
        worker_factory=worker_factory,
        service_factory=lambda *args, **kwargs: service,
    )
    artifact = json.loads((tmp_path / "artifacts/runs/new-trial.json").read_text())
    assert state["status"] == "blocked" and not service.submit.called
    assert (
        artifact["config_fingerprint"]
        == artifact["report"]["config_fingerprint"]
        == plan["config"]["fingerprint"]
    )
    assert artifact["report"]["delivery"]["publishable"] is False
    assert client.close.called and service.shutdown.called
