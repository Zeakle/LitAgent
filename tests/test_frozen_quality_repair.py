"""Validate a one-shot frozen repair supervisor without paid providers."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_quality_recovery_paths import completed_unverified

from litagent.config import load_config
from litagent.quality import build_fact_cards, digest
from litagent.runner import RewriteOutcome
from scripts.run_frozen_quality_repair import TrialBudget, prepare, run_trial


def source(tmp_path):
    """Supply a complete source-bound diagnostic and a known historic request."""
    body = "# Review\n\n## Introduction\n\nEViT improves speed. [E:a]"
    cards = build_fact_cards(
        {"a": {"paper_id": "evit", "supporting_text": "EViT improves speed."}}
    )
    data = {
        "run_id": "original",
        "query": "EViT",
        "status": "completed",
        "usage": {"total_tokens": 100},
        "events": [
            {
                "event": "llm.start",
                "data": {
                    "task_id": "query_decomposition",
                    "operation_id": "original-op",
                },
            },
            {
                "event": "llm.complete",
                "data": {
                    "task_id": "query_decomposition",
                    "operation_id": "original-op",
                    "prompt_tokens": 99,
                    "completion_tokens": 1,
                    "total_tokens": 100,
                    "usage_available": True,
                },
            },
        ],
        "nodes": {},
        "report": {
            "survey": body,
            "metadata": {
                "query": "EViT",
                "quality_policy": "strict_auto_v1",
                "review_structure": "academic_v2",
                "report_language": "en",
                "fact_cards": cards,
                "evidence_snapshot_sha256": digest(cards),
                "automatic_acceptance": {"actor": "user", "publishable": True},
                "human_review": {"approved": True},
            },
            "evaluation": completed_unverified(body),
            "quality": {"status": "failed"},
            "delivery": {"publishable": False},
        },
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data))
    return path


def factories(*, committed=False, crash=None, caps=True):
    client = SimpleNamespace(close=AsyncMock())
    service = SimpleNamespace(
        shutdown=AsyncMock(),
        capabilities=lambda: {"judge_available": caps, "jev_available": caps},
        judge_config=lambda: SimpleNamespace(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="doubao-seed-2-0-mini-260428",
        ),
    )
    agent = SimpleNamespace(_automatic_review=AsyncMock())

    async def rewrite(**kwargs):
        if crash:
            raise crash
        if committed:
            kwargs["report_data"]["quality"] = {"status": "passed"}
        return RewriteOutcome(
            True,
            committed,
            "new candidate" if committed else None,
            None,
            None,
            None,
            "committed" if committed else "validation_failed",
        )

    async def automatic(report, *args):
        report["metadata"]["automatic_acceptance"] = {"actor": "system"}
        report["delivery"] = {"publishable": True}

    agent._attempt_evidence_rewrite = AsyncMock(side_effect=rewrite)
    agent._automatic_review.side_effect = automatic
    return {
        "client_factory": Mock(return_value=client),
        "service_factory": Mock(return_value=service),
        "agent_factory": Mock(return_value=agent),
    }, agent


@pytest.mark.asyncio
async def test_dry_run_never_creates_transports_or_output_files(tmp_path):
    src = source(tmp_path)
    deps, _ = factories()
    result = await run_trial(
        src, load_config(), root=tmp_path / "trial", run_id="new-trial", **deps
    )
    assert result["status"] == "planned" and result["paid_calls"] == 0
    assert result["plan"]["source_budget"]["used_tokens"] == 100
    assert result["plan"]["source_budget"]["requests_sent"] == 1
    assert result["plan"]["limits"]["new_survey_requests"] == 7
    assert not (tmp_path / "trial").exists()
    for factory in deps.values():
        factory.assert_not_called()


@pytest.mark.parametrize(
    "fault",
    ["unknown", "missing_response", "failed_request", "total_mismatch", "stale_source"],
)
def test_incomplete_source_usage_or_location_rejected_before_transport(tmp_path, fault):
    src = source(tmp_path)
    data = json.loads(src.read_text())
    if fault == "unknown":
        data["events"][1]["data"]["usage_available"] = False
    elif fault == "missing_response":
        data["events"].pop()
    elif fault == "failed_request":
        data["events"].append({"event": "llm.failed", "data": {}})
    elif fault == "total_mismatch":
        data["usage"]["total_tokens"] = 101
    else:
        data["report"]["survey"] += " Different current report."
    src.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen_trial_"):
        prepare(src, load_config(), "new-trial")


@pytest.mark.parametrize("trial_id", ["../escape", "original", ""])
def test_new_trial_requires_a_new_safe_identity(tmp_path, trial_id):
    with pytest.raises(ValueError, match="frozen_trial_"):
        prepare(source(tmp_path), load_config(), trial_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authorization", [None, {"actor": "system"}, {"actor": "user", "binding": "stale"}]
)
async def test_live_requires_exact_human_authorization_before_client(
    tmp_path, authorization
):
    src = source(tmp_path)
    deps, _ = factories()
    with pytest.raises(ValueError, match="frozen_trial_authorization_required"):
        await run_trial(
            src,
            load_config(),
            root=tmp_path / "trial",
            run_id="new-trial",
            live=True,
            authorization=authorization,
            **deps,
        )
    deps["client_factory"].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_once_only_and_no_inherited_approval_or_failed_candidate_audit(
    tmp_path, committed
):
    src = source(tmp_path)
    before = src.read_bytes()
    cfg = load_config()
    auth = {
        "actor": "user",
        "binding": prepare(src, cfg, "new-trial")["plan"]["binding"],
    }
    deps, agent = factories(committed=committed)
    result = await run_trial(
        src,
        cfg,
        root=tmp_path / "trial",
        run_id="new-trial",
        live=True,
        authorization=auth,
        **deps,
    )
    assert result["status"] == ("completed" if committed else "blocked")
    assert agent._automatic_review.await_count == int(committed)
    assert "human_review" not in result["report"]["metadata"]
    assert (
        result["report"]["metadata"].get("automatic_acceptance", {}).get("actor")
        != "user"
    )
    assert result["report"]["evaluation"] == {}
    assert src.read_bytes() == before
    assert (
        await run_trial(
            src,
            cfg,
            root=tmp_path / "trial",
            run_id="new-trial",
            live=True,
            authorization=auth,
            **deps,
        )
        == result
    )
    deps["client_factory"].assert_called_once()
    agent._attempt_evidence_rewrite.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupted_checkpoint_is_never_replayed(tmp_path):
    src = source(tmp_path)
    cfg = load_config()
    plan = prepare(src, cfg, "new-trial")["plan"]
    root = tmp_path / "trial"
    root.mkdir()
    (root / "checkpoint.json").write_text(
        json.dumps({"binding": plan["binding"], "status": "started_no_replay"})
    )
    deps, _ = factories()
    result = await run_trial(
        src,
        cfg,
        root=root,
        run_id="new-trial",
        live=True,
        authorization={"actor": "user", "binding": plan["binding"]},
        **deps,
    )
    assert result["status"] == "interrupted" and "no_replay" in result["reason"]
    deps["client_factory"].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [OSError("provider failed"), __import__("asyncio").CancelledError()]
)
async def test_failure_and_cancellation_save_terminal_budget_and_release_clients(
    tmp_path, error
):
    src = source(tmp_path)
    cfg = load_config()
    plan = prepare(src, cfg, "new-trial")["plan"]
    root = tmp_path / "trial"
    deps, _ = factories(crash=error)
    args = dict(
        root=root,
        run_id="new-trial",
        live=True,
        authorization={"actor": "user", "binding": plan["binding"]},
        **deps,
    )
    if isinstance(error, __import__("asyncio").CancelledError):
        with pytest.raises(__import__("asyncio").CancelledError):
            await run_trial(src, cfg, **args)
    else:
        await run_trial(src, cfg, **args)
    state = json.loads((root / "checkpoint.json").read_text())
    assert (
        state["status"] in {"failed", "interrupted"}
        and state["budget"]["used_tokens"] == 100
    )
    deps["client_factory"].return_value.close.assert_awaited_once()
    deps["service_factory"].return_value.shutdown.assert_awaited_once()


def test_additional_request_ceiling_does_not_reset_old_usage(tmp_path):
    plan = prepare(source(tmp_path), load_config(), "new-trial")["plan"]
    budget = TrialBudget(plan["source_budget"], new_requests=2)
    budget.begin_request()
    budget.begin_request()
    with pytest.raises(ValueError, match="strict_request_budget_exceeded"):
        budget.begin_request()
    assert budget.used == 100 and budget.snapshot()["requests_sent"] == 3


@pytest.mark.asyncio
async def test_actual_runner_and_transport_boundary_reject_bad_native_candidate_without_audit(
    tmp_path,
):
    from litagent.llm.client import LLMResponse, OpenAICompatibleClient
    from litagent.observability.context import get_task_id

    src = source(tmp_path)
    cfg = load_config()
    plan = prepare(src, cfg, "real-boundary")["plan"]
    calls = []

    def client_factory(**kwargs):
        client = object.__new__(OpenAICompatibleClient)
        client._cost_budget = kwargs["cost_budget"]
        client._trace_hook = kwargs["trace_hook"]
        client._max_tokens = 20000
        client._transport_max_attempts = 1

        async def attempt(messages, **options):
            calls.append((messages, options))
            client._emit(
                "llm.start",
                {
                    "operation_id": "offline-op",
                    "task_id": get_task_id(),
                    "model": "offline",
                },
            )
            usage = {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "usage_available": True,
            }
            client._cost_budget.record(usage)
            client._emit(
                "llm.complete",
                {
                    "operation_id": "offline-op",
                    "task_id": get_task_id(),
                    "model": "offline",
                    "content": "{}",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "usage_available": True,
                },
            )
            return LLMResponse(content="{}", model="offline", usage=usage)

        client._chat_attempt = attempt
        client.close = AsyncMock()
        return client

    deps, _ = factories()
    deps.pop("agent_factory")
    deps["client_factory"] = client_factory
    root = tmp_path / "actual-trial"
    result = await run_trial(
        src,
        cfg,
        root=root,
        run_id="real-boundary",
        live=True,
        authorization={"actor": "user", "binding": plan["binding"]},
        **deps,
    )
    assert result["status"] == "blocked" and len(calls) == 1
    assert result["new_survey_requests"] == 1 and result["new_survey_tokens"] == 2
    assert (
        result["budget"]["requests_sent"] == 2
        and result["budget"]["used_tokens"] == 102
    )
    assert (
        result["report"]["evaluation"] == {}
        and not result["report"]["delivery"]["publishable"]
    )
    candidate = json.loads((root / "candidate.json").read_text())
    assert not candidate["committed"]
    artifact = json.loads((root / "runs" / "real-boundary.json").read_text())
    assert artifact["usage"]["total_tokens"] == 2


@pytest.mark.asyncio
async def test_missing_audit_capability_stops_before_survey_request(tmp_path):
    src = source(tmp_path)
    cfg = load_config()
    deps, agent = factories(caps=False)
    plan = prepare(src, cfg, "new-trial")["plan"]
    result = await run_trial(
        src,
        cfg,
        root=tmp_path / "trial",
        run_id="new-trial",
        live=True,
        authorization={"actor": "user", "binding": plan["binding"]},
        **deps,
    )
    assert result["status"] == "failed" and result["new_survey_requests"] == 0
    agent._attempt_evidence_rewrite.assert_not_awaited()


@pytest.mark.parametrize("limit", [0, 8, True, -1])
def test_invalid_additional_request_limit_is_rejected(tmp_path, limit):
    plan = prepare(source(tmp_path), load_config(), "new-trial")["plan"]
    with pytest.raises(ValueError, match="frozen_trial_request_limit_invalid"):
        TrialBudget(plan["source_budget"], new_requests=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    [
        "candidate.json",
        "input-snapshot.json",
        "runs/new-trial.json",
        "audits/index.sqlite3",
    ],
)
async def test_new_trial_does_not_overwrite_unrelated_output(tmp_path, filename):
    src = source(tmp_path)
    cfg = load_config()
    plan = prepare(src, cfg, "new-trial")["plan"]
    root = tmp_path / "trial"
    path = root / filename
    path.parent.mkdir(parents=True)
    path.write_text("user data")
    deps, _ = factories()
    with pytest.raises(ValueError, match="frozen_trial_output_collision"):
        await run_trial(
            src,
            cfg,
            root=root,
            run_id="new-trial",
            live=True,
            authorization={"actor": "user", "binding": plan["binding"]},
            **deps,
        )
    assert path.read_text() == "user data"
    deps["client_factory"].assert_not_called()


@pytest.mark.parametrize(
    "fault", ["duplicate_start", "duplicate_response", "wrong_operation", "event_total"]
)
def test_source_request_lifecycles_cannot_understate_spending(tmp_path, fault):
    src = source(tmp_path)
    data = json.loads(src.read_text())
    if fault == "duplicate_start":
        data["events"].insert(1, copy.deepcopy(data["events"][0]))
    elif fault == "duplicate_response":
        data["events"].append(copy.deepcopy(data["events"][1]))
    elif fault == "wrong_operation":
        data["events"][1]["data"]["operation_id"] = "other"
    else:
        data["events"][1]["data"]["total_tokens"] = 99
    src.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen_trial_source_"):
        prepare(src, load_config(), "new-trial")
