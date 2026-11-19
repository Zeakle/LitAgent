"""Exercise active API owners and actual browser consumers before archival."""

import asyncio
import copy
import json

import httpx
import pytest
from test_execution_evidence_inspection import run_js
from test_extraction_admission import dual_transport
from test_extraction_verification import selection

from litagent import api
from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.config import AgentConfig, AppConfig, LoggingConfig, ObservabilityConfig
from litagent.demo import project_flow_demo_view
from litagent.extraction_verification import ADMISSION_VERSION, ExtractionVerifier
from litagent.interaction_store import SqliteReviewRepository
from litagent.interactive_service import InteractiveRunService
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.workbench import RunOptions


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """Isolate API state without opening network services or provider clients."""
    config = AppConfig(
        agent=AgentConfig(),
        logging=LoggingConfig(),
        observability=ObservabilityConfig(enabled=False),
    )
    archive = ArchiveRepository(tmp_path / "archive")
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "offline")
    for key, value in {
        "tasks": {},
        "flow_runs": {},
        "flow_samples": {},
        "flow_archive_index": {},
        "flow_repository": archive,
        "config": config,
        "survey_semaphore": asyncio.Semaphore(1),
    }.items():
        monkeypatch.setattr(api.app.state, key, value, raising=False)
    return config, archive


def emit_work(hook, receipt=None):
    """Use scheduler and subspan event contracts, including worker.input."""
    graph = TaskGraph()
    graph.add_task(SubTask("search", "Search", "search", {"query": "query"}))
    graph.add_task(SubTask("extract", "Extract", "extractor"), ["search"])
    hook.capture_graph(graph)
    hook(
        "subspan.start",
        {
            "task_id": "query_decomposition",
            "name": "planner.query_decomposition",
            "input": {"query": "query"},
        },
    )
    hook(
        "subspan.end",
        {
            "task_id": "query_decomposition",
            "output": {"sub_queries": ["expanded query"]},
        },
    )
    hook(
        "worker.input",
        {"task_id": "extract", "agent_type": "extractor", "input": {"papers": ["p1"]}},
    )
    hook("worker.start", {"task_id": "extract", "agent_type": "extractor"})
    output = [
        {"evidence_items": list((receipt or {}).get("candidate_ledger", {}).values())}
    ]
    if receipt:
        output[0]["extraction_verification"] = receipt
    hook("worker.complete", {"task_id": "extract", "output": output})
    graph.mark_done("search", [{"paper_id": "p1"}])
    graph.mark_done("extract", output)
    hook.capture_graph_state(graph)
    return output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["queued", "running", "cancelling", "failed", "cancelled", "completed"]
)
async def test_ordinary_owner_views_and_history_work_without_report(runtime, status):
    """Both view routes expose the same in-memory I/O, even without a report."""
    _, archive = runtime
    recorder = RunRecorder("ordinary", "query", archive)
    output = emit_work(recorder)
    entry = api.SurveyTaskEntry(
        "ordinary", "query", status, status, api.CancellationToken(), recorder
    )
    entry.error = "offline_failure" if status == "failed" else None
    api.app.state.tasks["ordinary"] = entry
    before = recorder.snapshot()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        flow = await client.get("/flow-demo/runs/ordinary/view")
        common = await client.get("/survey/ordinary/view")
        history = (await client.get("/flow-demo/runs")).json()
        missing = await client.get("/flow-demo/runs/missing/view")
    assert flow.status_code == common.status_code == 200
    assert missing.status_code == 404
    data = flow.json()
    assert common.json()["graph"] == data["dag"]
    assert common.json()["capabilities"]["can_download_artifact"] is False
    assert data["summary"]["status"] == status
    assert data["summary"]["error"] == entry.error
    assert data["dag"]["dependencies"]["extract"] == ["search"]
    assert data["dag"]["nodes"]["worker:extract"]["input"] == {"papers": ["p1"]}
    assert data["dag"]["nodes"]["worker:extract"]["output"] == output
    assert common.json()["capabilities"]["can_cancel"] == (
        status in {"queued", "running", "cancelling"}
    )
    assert history[0]["source"] == (
        "live" if status in {"queued", "running", "cancelling"} else "archive"
    )
    assert recorder.snapshot() == before
    assert archive.get("ordinary") is None


@pytest.mark.asyncio
async def test_real_post_owns_running_snapshot_then_terminal_sse(runtime, monkeypatch):
    """Submit through the actual noninteractive route and observe before finish."""
    _, archive = runtime
    emitted, release = asyncio.Event(), asyncio.Event()

    class OfflineAgent:
        """Pause only the external execution boundary; keep API ownership real."""

        def __init__(self, config, *, trace_hook):
            self.hook = trace_hook

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def run(self, query, **kwargs):
            emit_work(self.hook)
            emitted.set()
            await release.wait()
            return {"survey": "# Offline result", "metadata": {}}

    monkeypatch.setattr(api, "LitAgent", OfflineAgent)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/survey", json={"query": "query", "quality_policy": "legacy"}
        )
        assert response.status_code == 202
        run_id = response.json()["task_id"]
        entry = api.app.state.tasks[run_id]
        try:
            await asyncio.wait_for(emitted.wait(), 3)
            data = (await client.get(f"/flow-demo/runs/{run_id}/view")).json()
            assert data["summary"]["status"] == "running"
            assert data["dag"]["nodes"]["worker:extract"]["status"] == "completed"
            assert archive.get(run_id) is None
        finally:
            release.set()
            await asyncio.wait_for(entry.task, 3)
        terminal = (await client.get(f"/survey/{run_id}/view")).json()
        assert terminal["summary"]["status"] == "completed"
        assert terminal["capabilities"]["can_download_artifact"] is True
        assert terminal["report"]["survey"] == "# Offline result"
        events = await client.get(f"/survey/{run_id}/events")
        assert "event: trace" in events.text and "event: done" in events.text
        assert "expanded query" in events.text


@pytest.mark.asyncio
async def test_interactive_live_and_replayed_nodes_feed_real_browser(
    runtime, tmp_path, monkeypatch
):
    """Read live and restarted SQLite owners with a real offline Jev receipt."""
    config, archive = runtime
    repository = SqliteReviewRepository(tmp_path / "interactive")
    service = InteractiveRunService(config, repository, archive_repository=archive)
    await repository.initialize()
    await repository.create_run("interactive", "query", RunOptions(human_review=True))
    monkeypatch.setattr(api.app.state, "interactive_service", service)
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test")
    captured = []
    async with httpx.AsyncClient(transport=dual_transport(captured)) as client:
        verifier = ExtractionVerifier(
            tmp_path / "jev",
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        _, receipt = await verifier.verify_candidates(
            {"a": selection().selected_items["a"]}, run_id="interactive"
        )
    hook = await service._trace_hook("interactive", "query")
    output = emit_work(hook, receipt)
    plan = {"contract": "planned-budget-v7", "plan_identity": "a" * 64}
    repair = {"status": "pending", "reason": "offline-check"}
    hook("budget.plan", {"budget_plan": plan})
    hook("targeted_repair.state", {"task_id": "extract", "state": repair})
    recorder = service._recorders["interactive"]
    pipeline = {"stages": {"extraction": "completed", "writing": "pending"}}
    recorder.set_extraction_verification(receipt, pipeline)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        live = await client.get("/survey/interactive/view")
        assert live.status_code == 200
        data = live.json()
        assert data["graph"]["dependencies"]["extract"] == ["search"]
        assert data["graph"]["nodes"]["worker:extract"]["output"] == output
        assert data["pipeline_verification"] == {**pipeline, "writing_diagnostics": []}
        assert data["budget_plan"] == plan
        assert data["targeted_repair_state"] == repair
        assert data["display_warnings"] == []
        assert not data["delivery"]
        assert not data["capabilities"]["can_download_artifact"]
        assert archive.get("interactive") is None
        restarted = InteractiveRunService(
            config, repository, archive_repository=archive
        )
        monkeypatch.setattr(api.app.state, "interactive_service", restarted)
        replay = (await client.get("/survey/interactive/view")).json()
    assert replay["graph"]["nodes"]["worker:extract"]["input"] == {"papers": ["p1"]}
    assert replay["graph"]["nodes"]["worker:extract"]["output"] == output
    assert replay["graph"]["tasks"]["extract"]["agent_type"] == "extractor"
    assert replay["graph"]["dependencies"] == {}
    assert replay["report"]["metadata"]["extraction_verification"] == receipt
    assert replay["budget_plan"] == plan
    assert replay["targeted_repair_state"] == repair
    assert archive.get("interactive") is None
    for payload in [data, replay]:
        run_js(
            "const apiPayload = "
            + json.dumps(payload)
            + r""";
const {adaptInteractiveView} = await load('adapters.js');
const {buildExecutionPresentation} = await load('execution-model.js');
const {buildEvidenceLedger} = await load('evidence-model.js');
const adapted = adaptInteractiveView(apiPayload, {key:'interactive:interactive',runId:'interactive',source:'interactive'});
const ledger = buildEvidenceLedger(adapted);
assert.equal(ledger.length,1);
assert.equal(ledger[0].inspection.state,'accepted');
assert.equal(ledger[0].inspection.decisions.length,2);
assert.equal(adapted.graph.nodes['worker:extract'].input.papers[0],'p1');
const presentation = buildExecutionPresentation(adapted, {locale:'en'});
assert(presentation.tasks.some(task=>task.id==='extract'));
assert(!presentation.observations.some(node=>node.id==='system:delivery'));
"""
        )


def test_pre_report_receipt_conflicts_and_terminal_delivery_are_explicit():
    """Never choose between conflicting receipts or claim active delivery finished."""
    receipt = {"status": "completed", "source_run_id": "r", "candidate_ledger": {}}
    artifact = {
        "run_id": "r",
        "status": "running",
        "nodes": {"worker:extract": {"output": [{"extraction_verification": receipt}]}},
    }
    original = copy.deepcopy(artifact)
    live = project_flow_demo_view(artifact, source="live")
    assert live["report"]["metadata"]["extraction_verification"] == receipt
    assert live["delivery"] == {}
    assert artifact == original
    artifact["status"] = "failed"
    assert (
        project_flow_demo_view(artifact, source="archive")["delivery"]["status"]
        == "blocked"
    )
    artifact["nodes"]["worker:extract"]["output"].append(
        {"extraction_verification": {**receipt, "status": "disabled"}}
    )
    assert (
        "extraction_verification"
        not in project_flow_demo_view(artifact, source="archive")["report"]["metadata"]
    )


def test_eventsource_reconnect_keeps_stream_open_until_terminal():
    """Transient errors retain browser reconnection; terminal events close once."""
    run_js(
        r"""
class OfflineStream {
  constructor(url) {this.url=url;this.handlers={};this.closed=0;}
  addEventListener(name,handler) {this.handlers[name]=handler;}
  close(){this.closed++;}
}
globalThis.EventSource=OfflineStream;
const {openEventStream}=await load('api-client.js');
let errors=0,settled=0;
const stream=openEventStream('run one',{onError(){errors++;},onSettle(){settled++;}});
stream.onerror(); assert.equal(errors,1); assert.equal(stream.closed,0);
assert.equal(stream.url,'/survey/run%20one/events');
stream.handlers.done({type:'done'});assert.equal(settled,1);assert.equal(stream.closed,1);
"""
    )


@pytest.mark.asyncio
async def test_presentation_mode_does_not_expose_private_active_owner(
    runtime, monkeypatch
):
    """An ordinary live task is not implicitly added to presentation mode."""
    _, archive = runtime
    recorder = RunRecorder("private", "private query", archive)
    emit_work(recorder)
    api.app.state.tasks["private"] = api.SurveyTaskEntry(
        "private",
        "private query",
        "running",
        "extract",
        api.CancellationToken(),
        recorder,
    )
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "presentation")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.get("/flow-demo/runs/private/view")
        assert response.status_code == 404
        assert not any(
            item["run_id"] == "private"
            for item in (await client.get("/flow-demo/runs")).json()
        )


@pytest.mark.asyncio
async def test_flow_event_alias_waits_for_ordinary_owner_terminal(runtime):
    """A live ordinary stream must not settle after sending its first snapshot."""
    _, archive = runtime
    recorder = RunRecorder("ordinary", "query", archive)
    emit_work(recorder)
    entry = api.SurveyTaskEntry(
        "ordinary", "query", "running", "extract", api.CancellationToken(), recorder
    )
    api.app.state.tasks["ordinary"] = entry
    response = await api.stream_flow_demo_events("ordinary")
    stream = response.body_iterator
    for _ in recorder.snapshot()["events"]:
        assert "event: trace" in await anext(stream)
    pending = asyncio.create_task(anext(stream))
    try:
        await asyncio.sleep(0)
        assert not pending.done()
        entry.status = "completed"
        assert "event: done" in await asyncio.wait_for(pending, 1)
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
