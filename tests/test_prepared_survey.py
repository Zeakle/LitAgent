"""Verify bounded continuation identities, skipped work, and shared accounting."""

import copy
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from test_preparation_rebuild import source

from litagent.orchestrator.task_graph import TaskStatus
from litagent.safety.budget import CostBudget
from scripts import run_prepared_survey as continuation
from scripts.rebuild_extraction_preparation import rebuild


@pytest_asyncio.fixture
async def prepared(tmp_path):
    config = continuation.effective_config()
    data = source()
    # This synthetic parent uses the current profile before its input is frozen.
    data["graph"]["tasks"]["adversarial_review"][
        "timeout_ms"
    ] = config.adversarial.task_timeout_ms
    data["query"] = (
        "Compare metric-learning approaches for few-shot image classification."
    )
    data["graph"]["tasks"]["adversarial_review"]["input"]["query"] = data["query"]
    extra = copy.deepcopy(data["nodes"]["worker:relevance_gate"]["output"][0])
    extra.update(
        paper_id="fixture:pending", title="Additional public metric-learning study"
    )
    data["nodes"]["worker:relevance_gate"]["output"].append(extra)
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(data))
    receipt = await rebuild(parent, config)
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(receipt))
    return config, parent, path, receipt


@pytest.mark.asyncio
async def test_prepare_only_preserves_spent_budget_and_identifies_pending(
    prepared, tmp_path
):
    config, parent, path, p = prepared
    before = parent.read_bytes()
    root = tmp_path / "prepared-001"
    m = continuation.prepare(root, path, config)
    state = await continuation.run(root, config)
    assert state["status"] == "planned" and state["pending_papers"] == 1
    assert m["new_token_ceiling"] == 499980 and m["new_request_ceiling"] == 63
    assert parent.read_bytes() == before and not (root / "checkpoint.json").exists()
    assert [i["paper_id"] for i in continuation.pending_papers(p)] == [
        "fixture:pending"
    ]
    with pytest.raises(FileExistsError):
        continuation.prepare(root, path, config)


@pytest.mark.asyncio
async def test_prepared_planner_leaves_only_writing_pending(prepared):
    _, _, _, p = prepared
    planner = continuation.FrozenPreparedPlanner(p, p["extractions"])
    graph = await planner.plan(p["query"])
    assert graph.get_task("relevance_gate").status == TaskStatus.DONE
    assert graph.get_task("extract").status == TaskStatus.DONE
    assert [t.task_id for t in graph.get_ready_tasks()] == ["adversarial_review"]
    with pytest.raises(ValueError, match="query_changed"):
        await planner.plan("new question")


def test_budget_restoration_keeps_borrowers_and_request_count():
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.begin_request()
    budget.record({"prompt_tokens": 10, "completion_tokens": 10})
    inherited = budget.snapshot()
    live = CostBudget()
    agent = SimpleNamespace(
        _cost_budget=live,
        _scheduler=SimpleNamespace(_cost_budget=live),
        _stage_llm=lambda role: SimpleNamespace(_cost_budget=live),
    )
    continuation.restore_shared_budget(agent, {"inherited_budget": inherited})
    assert agent._cost_budget is live and live.snapshot()["requests_sent"] == 1
    live.begin_request()
    assert live.snapshot()["requests_sent"] == 2
    assert live.snapshot()["used_tokens"] == 20
    agent._scheduler._cost_budget = CostBudget()
    with pytest.raises(ValueError, match="scheduler_budget"):
        continuation.restore_shared_budget(agent, {"inherited_budget": inherited})


@pytest.mark.asyncio
async def test_started_checkpoint_never_creates_another_agent(
    prepared, tmp_path, monkeypatch
):
    config, _, path, _ = prepared
    root = tmp_path / "prepared-002"
    m = continuation.prepare(root, path, config)
    (root / "checkpoint.json").write_text(
        json.dumps({"status": "started_no_replay", "binding": m["binding"]})
    )
    monkeypatch.setattr(
        continuation, "LitAgent", lambda *a, **kw: pytest.fail("must not create agent")
    )
    state = await continuation.run(root, config, live=True)
    assert state["status"] == "interrupted_no_replay"
    assert await continuation.run(root, config, live=True) == state


@pytest.mark.asyncio
async def test_parent_mutation_is_rejected_before_agent_creation(prepared, tmp_path):
    config, parent, path, _ = prepared
    root = tmp_path / "prepared-003"
    continuation.prepare(root, path, config)
    parent.write_text(parent.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen_source_changed"):
        await continuation.run(root, config, live=True)


def test_one_parent_has_one_paid_child(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = {"parent_sha256": "b" * 64, "binding": "a" * 64}
    continuation.claim_owner(tmp_path / "child-1", manifest)
    continuation.claim_owner(tmp_path / "child-1", manifest)
    with pytest.raises(ValueError, match="already_owned"):
        continuation.claim_owner(
            tmp_path / "child-2", {**manifest, "binding": "c" * 64}
        )


@pytest.mark.asyncio
async def test_no_additional_fulltext_loader_does_not_mutate_candidates(prepared):
    _, _, _, p = prepared
    pending = continuation.pending_papers(p)
    before = copy.deepcopy(pending)
    actual = await continuation.NoAdditionalFulltext().enrich(pending, [])
    assert pending == before and actual[0]["chunks"] == pending[0]["chunks"]
    assert actual[0]["fulltext_acquisition"]["full_document_reviewed"] is False


@pytest.mark.asyncio
async def test_real_execution_reaches_sdk_extraction_and_pro_writer_without_false_pass(
    prepared, tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    import httpx
    from openai import AsyncOpenAI
    from test_extraction_record_identity import real_case

    from litagent.runner import Infra, LitAgent

    config, _, path, _ = prepared
    root = tmp_path / "prepared-real-consumer"
    monkeypatch.setenv("LLM_API_KEY", "offline-test")
    monkeypatch.setattr(continuation, "OWNER_ROOT", tmp_path / "owners")
    import litagent.report_audit as report_audit_module
    from litagent.observability.recorder import ArchiveRepository
    from litagent.report_audit import atomic_json as real_atomic_json

    monkeypatch.setattr(
        continuation, "ArchiveRepository", lambda: ArchiveRepository(tmp_path / "runs")
    )

    def save_local(path, value):
        if str(path).startswith("artifacts/automatic-acceptance/"):
            path = tmp_path / "acceptances" / path.name
        return real_atomic_json(path, value)

    monkeypatch.setattr(report_audit_module, "atomic_json", save_local)
    from litagent.extraction_verification import ExtractionVerifier

    original_verifier_init = ExtractionVerifier.__init__

    def verifier_init(self, root=None, **kwargs):
        return original_verifier_init(self, tmp_path / "extraction-receipts", **kwargs)

    monkeypatch.setattr(ExtractionVerifier, "__init__", verifier_init)

    monkeypatch.setattr(LitAgent, "_connect_infra", AsyncMock(return_value=Infra()))

    class OfflineTrace:
        trace_id = "0" * 32
        trace_url = "http://offline.invalid/trace"

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(continuation, "LangFuseTracer", OfflineTrace)
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = (
            json.dumps(real_case()["response"])
            if payload["model"].endswith("lite-260915")
            else "{}"
        )
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "created": 1,
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    monkeypatch.setattr(
        "litagent.llm.client.AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    continuation.prepare(root, path, config)
    result = await continuation.run(root, config, live=True)
    assert result.get("reason_type") not in {"NameError", "TypeError", "AttributeError"}
    assert requests and requests[0]["model"] == "doubao-seed-2-1-lite-260915"
    assert any(p["model"] == "doubao-seed-2-1-pro-260915" for p in requests)
    assert result["status"] in {"blocked", "failed_no_replay"}
    assert result["incremental_cost"]["requests"] == len(requests)
    assert result["budget"]["requests_sent"] == 1 + len(requests)
    assert result["budget"]["used_tokens"] == 20 + 20 * len(requests)
    assert (
        json.loads((root / "completed-preparation.json").read_text())["extractions"][1][
            "paper_id"
        ]
        == "fixture:pending"
    )
    count = len(requests)
    assert await continuation.run(root, config, live=True) == result
    assert len(requests) == count


@pytest.mark.asyncio
async def test_zero_call_owner_release_retains_history_and_allows_corrected_child(
    prepared, tmp_path, monkeypatch
):
    config, _, path, _ = prepared
    monkeypatch.setattr(continuation, "OWNER_ROOT", tmp_path / "owners")
    root = tmp_path / "zero-call-child"
    manifest = continuation.prepare(root, path, config)
    continuation.claim_owner(root, manifest)
    checkpoint = {
        "binding": manifest["binding"],
        "status": "failed_no_replay",
        "incremental_cost": {"tokens": 0, "requests": 0, "usage_complete": True},
        "budget": manifest["inherited_budget"],
    }
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))
    (root / "events.jsonl").write_text("")
    before = (root / "checkpoint.json").read_bytes()
    proof = continuation.release_zero_call_owner(root)
    assert proof["status"] == "released_before_model_call"
    assert (root / "checkpoint.json").read_bytes() == before
    new = {**manifest, "binding": "c" * 64}
    continuation.claim_owner(tmp_path / "corrected-child", new)
    assert (
        tmp_path / "owners/zero-call-history" / f"{manifest['binding']}.json"
    ).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["request_started", "budget_increment", "unknown_usage"]
)
async def test_any_model_activity_or_uncertainty_prevents_owner_release(
    prepared, tmp_path, monkeypatch, fault
):
    config, _, path, _ = prepared
    monkeypatch.setattr(continuation, "OWNER_ROOT", tmp_path / "owners")
    root = tmp_path / "not-zero"
    m = continuation.prepare(root, path, config)
    continuation.claim_owner(root, m)
    state = {
        "binding": m["binding"],
        "status": "failed_no_replay",
        "budget": copy.deepcopy(m["inherited_budget"]),
        "incremental_cost": {"tokens": 0, "requests": 0, "usage_complete": True},
    }
    events = []
    if fault == "request_started":
        events = [{"event": "llm.start", "data": {}}]
    if fault == "budget_increment":
        state["budget"]["requests_sent"] += 1
    if fault == "unknown_usage":
        state["incremental_cost"]["usage_complete"] = False
    (root / "checkpoint.json").write_text(json.dumps(state))
    (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="zero_call"):
        continuation.release_zero_call_owner(root)
