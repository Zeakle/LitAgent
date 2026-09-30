"""Verify Phase 15.9 execution projections and graph interaction contracts."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path("src/litagent/static/flow-demo")


def _run_module(module: str, expression: str) -> object:
    """Execute a browser-independent ES module contract with local Node."""
    if shutil.which("node") is None:
        pytest.skip("Node is unavailable for the ES module contract test")
    uri = (ROOT / module).resolve().as_uri()
    script = (
        f"import * as subject from {json.dumps(uri)}; "
        f"console.log(JSON.stringify({expression}));"
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_execution_presentation_uses_real_outputs_and_preserves_dependencies() -> None:
    result = _run_module(
        "execution-model.js",
        "subject.buildExecutionPresentation({graph: {tasks: {"
        "search: {task_id: 'search', agent_type: 'search', status: 'done'}, "
        "dedup: {task_id: 'dedup', agent_type: 'dedup', status: 'failed'}}, "
        "dependencies: {search: [], dedup: ['search']}, nodes: {"
        "'worker:search': {id: 'worker:search', kind: 'worker', task_id: 'search', "
        "name: 'search', status: 'completed', elapsed_ms: 1200, output: ["
        "{paper_id: 'p1'}, {paper_id: 'p1'}, {paper_id: 'p2'}]}, "
        "'worker:dedup': {id: 'worker:dedup', kind: 'worker', task_id: 'dedup', "
        "name: 'dedup', status: 'failed', error: 'bad input', output: null}}}, "
        "events: [], evidence: []})",
    )

    tasks = {task["id"]: task for task in result["tasks"]}
    assert result["dependencies"] == [{"from": "search", "to": "dedup"}]
    assert tasks["search"]["status"] == "completed"
    assert tasks["search"]["durationMs"] == 1200
    assert tasks["search"]["summaryFacts"] == [
        {
            "kind": "paper_count",
            "value": 2,
            "labelKey": "papersFound",
            "sourcePath": "graph.nodes.worker:search.output",
        }
    ]
    assert tasks["dedup"]["status"] == "failed"
    assert tasks["dedup"]["error"] == "bad input"


def test_stage_state_keeps_failures_and_active_work_visible_together() -> None:
    result = _run_module(
        "execution-model.js",
        "subject.aggregateStageState(["
        "{status: 'completed'}, {status: 'running'}, {status: 'failed'}, "
        "{status: null}, {status: 'skipped'}])",
    )

    assert result["counts"] == {
        "completed": 1,
        "running": 1,
        "failed": 1,
        "unknown": 1,
        "skipped": 1,
    }
    assert result["active"] is True
    assert result["hasFailures"] is True
    assert result["terminal"] is False
    assert result["unknownCount"] == 1


def test_stage_overview_collapses_tasks_without_losing_source_edges() -> None:
    result = _run_module(
        "execution-model.js",
        "subject.buildStageOverview({tasks: ["
        "{id: 's1', stageId: 'discovery', status: 'completed'}, "
        "{id: 's2', stageId: 'discovery', status: 'completed'}, "
        "{id: 'd', stageId: 'evidence', status: 'completed'}], "
        "stages: [{id: 'discovery', memberTaskIds: ['s1','s2'], memberObservationIds: []}, "
        "{id: 'evidence', memberTaskIds: ['d'], memberObservationIds: []}], "
        "dependencies: [{from: 's1', to: 'd'}, {from: 's2', to: 'd'}], "
        "observations: []}, [])",
    )

    assert [node["id"] for node in result["nodes"]] == [
        "stage:discovery",
        "stage:evidence",
    ]
    assert result["edges"] == [
        {
            "from": "stage:discovery",
            "to": "stage:evidence",
            "sourceEdges": [
                {"from": "s1", "to": "d"},
                {"from": "s2", "to": "d"},
            ],
        }
    ]


def test_layout_fit_and_neighborhood_are_deterministic() -> None:
    result = _run_module(
        "execution-view.js",
        "(() => { const graph = subject.buildExecutionGraph("
        "{a:{task_id:'a'}, b:{task_id:'b'}, c:{task_id:'c'}}, "
        "{a:[], b:['a'], c:['b']}, {}); "
        "const layout = subject.layoutExecutionGraph(graph, {direction:'LR'}); "
        "return {fit: subject.fitGraphToViewport(layout.bounds, {width: 600, height: 300}), "
        "near: subject.getNodeNeighborhood(graph, 'b'), "
        "positions: layout.nodes.map(n => [n.id,n.x,n.y])}; })()",
    )

    assert 0.25 <= result["fit"]["scale"] <= 1
    assert set(result["near"]["upstreamIds"]) == {"a"}
    assert set(result["near"]["downstreamIds"]) == {"c"}
    assert set(result["near"]["edgeIds"]) == {"a>b", "b>c"}
    assert result["positions"][0][1] < result["positions"][1][1]
    assert result["positions"][1][1] < result["positions"][2][1]


def test_call_timeline_keeps_retries_and_marks_orphans() -> None:
    result = _run_module(
        "execution-timeline.js",
        "subject.buildCallTimeline({observations: ["
        "{id:'llm:1', kind:'llm', taskId:'writer', status:'completed', "
        "startedAt:'2026-01-01T00:00:00Z', completedAt:'2026-01-01T00:00:02Z'}], "
        "events: ["
        "{sequence:1,event:'tool.retry',at:'2026-01-01T00:00:01Z',data:{task_id:'writer',operation_id:'tool:1',attempt:1}},"
        "{sequence:2,event:'tool.retry',at:'2026-01-01T00:00:02Z',data:{task_id:'writer',operation_id:'tool:1',attempt:2}},"
        "{sequence:3,event:'tool.complete',at:'2026-01-01T00:00:03Z',data:{task_id:'missing',operation_id:'tool:orphan'}}]})",
    )

    retries = [item for item in result["items"] if item["event"] == "tool.retry"]
    assert [item["attempt"] for item in retries] == [1, 2]
    assert any(
        item["orphan"] for item in result["items"] if item["taskId"] == "missing"
    )
    assert result["total"] == 4


def test_call_timeline_collapses_paired_lifecycle_events_into_observation() -> None:
    result = _run_module(
        "execution-timeline.js",
        "subject.buildCallTimeline({observations: ["
        "{id:'llm:op-1', kind:'llm', taskId:'writer', operationId:'op-1', "
        "status:'completed', startedAt:'2026-01-01T00:00:00Z'}], events: ["
        "{sequence:1,event:'llm.start',at:'2026-01-01T00:00:00Z',"
        "data:{task_id:'writer',operation_id:'op-1'}},"
        "{sequence:2,event:'llm.complete',at:'2026-01-01T00:00:02Z',"
        "data:{task_id:'writer',operation_id:'op-1'}},"
        "{sequence:3,event:'llm.retry',at:'2026-01-01T00:00:01Z',"
        "data:{task_id:'writer',operation_id:'op-1',attempt:2}}]})",
    )

    assert [item["event"] for item in result["items"]] == [
        "llm.completed",
        "llm.retry",
    ]


def test_call_timeline_pairs_worker_input_and_subspan_without_operation_id() -> None:
    result = _run_module(
        "execution-timeline.js",
        "subject.buildCallTimeline({observations: ["
        "{id:'worker:search-1', kind:'worker', taskId:'search-1', "
        "status:'completed', startedAt:'2026-01-01T00:00:00Z'},"
        "{id:'subspan:query', kind:'subspan', taskId:'query', "
        "status:'completed', startedAt:'2026-01-01T00:00:00Z'}], events: ["
        "{sequence:1,event:'worker.input',data:{task_id:'search-1'}},"
        "{sequence:2,event:'worker.start',data:{task_id:'search-1'}},"
        "{sequence:3,event:'worker.complete',data:{task_id:'search-1'}},"
        "{sequence:4,event:'subspan.start',data:{task_id:'query'}},"
        "{sequence:5,event:'subspan.end',data:{task_id:'query'}}]})",
    )

    assert [item["event"] for item in result["items"]] == [
        "subspan.completed",
        "worker.completed",
    ]


def test_delivery_stage_separates_policy_completion_from_blocked_outcome() -> None:
    result = _run_module(
        "execution-model.js",
        "subject.buildExecutionPresentation({graph:{tasks:{},dependencies:{},nodes:{}},"
        "delivery:{status:'blocked',publishable:false,reason_codes:['quality_failed']},"
        "quality:{status:'failed'}})",
    )

    delivery = next(stage for stage in result["stages"] if stage["id"] == "delivery")
    observation = next(
        item for item in result["observations"] if item["id"] == "system:delivery"
    )
    assert delivery["state"]["counts"] == {"completed": 1}
    assert observation["status"] == "completed"
    assert observation["output"]["status"] == "blocked"


def test_execution_ui_state_is_isolated_per_run() -> None:
    result = _run_module(
        "state.js",
        "(() => { const store = subject.createWorkbenchStore(); "
        "store.setExecutionUi('run-a', {mode:'dag', scale:0.7}); "
        "store.setExecutionUi('run-b', {mode:'timeline'}); "
        "return {a:store.getExecutionUi('run-a'), b:store.getExecutionUi('run-b')}; })()",
    )

    assert result["a"]["mode"] == "dag"
    assert result["a"]["scale"] == 0.7
    assert result["b"]["mode"] == "timeline"
    assert result["b"]["scale"] == 1


def test_phase15_9_shell_exposes_execution_modes_and_graph_tools() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    for element_id in (
        "execution-mode",
        "stage-overview",
        "execution-graph",
        "execution-timeline",
        "graph-zoom-in",
        "graph-zoom-out",
        "graph-fit",
        "graph-focus",
        "graph-fullscreen",
    ):
        assert f'id="{element_id}"' in html

    assert html.count('data-i18n-label="') >= 7


def test_execution_i18n_includes_graph_controls_and_duration_unit() -> None:
    result = _run_module(
        "i18n.js",
        "(() => ({zh: ['zoomOut','fitGraph','secondsShort'].map("
        "subject.translator('zh-CN')), en: ['zoomOut','fitGraph','secondsShort'].map("
        "subject.translator('en'))}))()",
    )

    assert result == {
        "zh": ["缩小", "适配画布", "秒"],
        "en": ["Zoom out", "Fit graph", "s"],
    }
