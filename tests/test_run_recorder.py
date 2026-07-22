import json

from litagent.observability.recorder import ArchiveRepository, RedactingTraceHook, RunRecorder
from litagent.orchestrator.task_graph import SubTask, TaskGraph


def test_recorder_pairs_lifecycle_events_and_persists_full_payload(tmp_path):
    repository = ArchiveRepository(tmp_path)
    recorder = RunRecorder("run-1", "few-shot learning in computer vision", repository)

    recorder("worker.input", {
        "task_id": "search_arxiv_q0",
        "agent_type": "search",
        "input": {"query": "few-shot", "source": "arxiv"},
    })
    recorder("worker.start", {"task_id": "search_arxiv_q0", "agent_type": "search"})
    recorder("tool.start", {"operation_id": "tool-1", "task_id": "search_arxiv_q0", "name": "search_arxiv"})
    recorder("tool.complete", {"operation_id": "tool-1", "task_id": "search_arxiv_q0", "name": "search_arxiv", "output": [{"title": "Paper"}]})
    recorder("worker.complete", {
        "task_id": "search_arxiv_q0",
        "agent_type": "search",
        "output": [{"title": "Paper"}],
    })
    artifact = recorder.finalize({"survey": "draft", "delivery": {"status": "ready"}})

    assert artifact["status"] == "completed"
    assert artifact["version"] == 2
    assert artifact["nodes"]["worker:search_arxiv_q0"]["input"]["query"] == "few-shot"
    assert artifact["nodes"]["tool:tool-1"]["output"] == [{"title": "Paper"}]
    assert repository.path_for("run-1").is_file()

    restored = repository.get("run-1")
    assert restored is not None
    assert restored["report"]["survey"] == "draft"
    assert json.loads(repository.path_for("run-1").read_text(encoding="utf-8"))["run_id"] == "run-1"


def test_recorder_syncs_final_graph_and_execution_summary(tmp_path):
    repository = ArchiveRepository(tmp_path)
    recorder = RunRecorder("run-graph", "query", repository)
    graph = TaskGraph()
    graph.add_task(SubTask(task_id="search", description="s", agent_type="search"))
    graph.add_task(SubTask(task_id="extract", description="e", agent_type="extractor"),
                   depends_on=["search"])
    recorder.capture_graph(graph)

    graph.mark_done("search", [{"title": "Paper"}])
    graph.mark_failed("extract", "worker_timeout")
    recorder.capture_graph_state(graph)
    artifact = recorder.finalize({"survey": "partial", "quality": {"status": "failed"},
                                  "delivery": {"status": "partial"}})

    assert artifact["graph"]["tasks"]["search"]["status"] == "done"
    assert artifact["graph"]["tasks"]["extract"]["status"] == "failed"
    assert artifact["graph"]["tasks"]["extract"]["error"] == "worker_timeout"
    assert artifact["task_statuses"] == {"done": 1, "failed": 1}
    assert artifact["quality"] == {"status": "failed"}
    assert artifact["delivery"] == {"status": "partial"}


def test_completed_artifact_rejects_stale_graph_state(tmp_path):
    recorder = RunRecorder("run-stale", "query", ArchiveRepository(tmp_path))
    graph = TaskGraph()
    graph.add_task(SubTask(task_id="pending", description="p", agent_type="search"))
    recorder.capture_graph(graph)

    artifact = recorder.finalize({"survey": "draft"})

    assert artifact["status"] == "failed"
    assert artifact["error"] == "incomplete_graph_state"


def test_recorder_maps_rag_memory_and_usage_payloads(tmp_path):
    recorder = RunRecorder("run-io", "query", ArchiveRepository(tmp_path))
    recorder("rag.search.start", {
        "operation_id": "rag-1", "task_id": "recall", "query": "few shot", "top_k": 5,
    })
    recorder("rag.search.complete", {
        "operation_id": "rag-1", "task_id": "recall", "count": 1,
        "results": [{"title": "Paper", "score": 0.9}], "elapsed_ms": 12,
    })
    recorder("memory.write.start", {
        "operation_id": "mem-1", "task_id": "cleanup", "layer": "episodic",
        "source": "survey",
    })
    recorder("memory.write.complete", {
        "operation_id": "mem-1", "task_id": "cleanup", "source": "survey",
        "duration_ms": 4, "elapsed_ms": 4,
    })
    recorder("llm.start", {
        "operation_id": "llm-1", "task_id": "planner", "model": "model",
        "messages": [{"role": "user", "content": "query"}], "max_tokens": 10,
    })
    recorder("llm.complete", {
        "operation_id": "llm-1", "task_id": "planner", "model": "model",
        "content": "answer", "prompt_tokens": 7, "completion_tokens": 3,
        "total_tokens": 10, "elapsed_ms": 8,
    })

    artifact = recorder.finalize({"survey": "answer"})

    assert artifact["nodes"]["rag.search:rag-1"]["input"] == {
        "query": "few shot", "top_k": 5,
    }
    assert artifact["nodes"]["rag.search:rag-1"]["output"]["results"][0]["title"] == "Paper"
    assert artifact["nodes"]["memory.write:mem-1"]["input"]["layer"] == "episodic"
    assert artifact["nodes"]["memory.write:mem-1"]["output"]["duration_ms"] == 4
    assert artifact["usage"] == {
        "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
    }
    assert artifact["elapsed_ms"] >= 0


def test_worker_elapsed_is_derived_when_terminal_event_omits_it(tmp_path):
    recorder = RunRecorder("run-worker-time", "query", ArchiveRepository(tmp_path))
    recorder._apply_event(
        "worker.start",
        {"task_id": "extract", "agent_type": "extractor"},
        "2026-07-22T07:00:00.000000+00:00",
    )
    recorder._apply_event(
        "worker.complete",
        {"task_id": "extract", "agent_type": "extractor", "output": []},
        "2026-07-22T07:00:01.250000+00:00",
    )

    assert recorder.snapshot()["nodes"]["worker:extract"]["elapsed_ms"] == 1250


def test_tool_call_llm_completion_is_recorded_as_output(tmp_path):
    recorder = RunRecorder("run-tool-call", "query", ArchiveRepository(tmp_path))
    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "load_skill", "arguments": {"name": "cv"}},
    }]
    recorder("llm.start", {
        "operation_id": "llm-tool", "task_id": "synthesis", "model": "model",
        "messages": [{"role": "user", "content": "query"}],
    })
    recorder("llm.complete", {
        "operation_id": "llm-tool", "task_id": "synthesis", "model": "model",
        "content": "", "tool_calls": tool_calls, "prompt_tokens": 3,
        "completion_tokens": 2, "total_tokens": 5, "elapsed_ms": 10,
    })

    node = recorder.snapshot()["nodes"]["llm:llm-tool"]
    assert node["output"] == {"content": "", "tool_calls": tool_calls}


def test_repository_reads_legacy_v1_archive_without_rewriting(tmp_path):
    repository = ArchiveRepository(tmp_path)
    legacy = {"version": 1, "run_id": "legacy", "events": [], "completed_at": "2026-01-01"}
    repository.path_for("legacy").write_text(json.dumps(legacy), encoding="utf-8")

    assert repository.get("legacy") == legacy
    assert json.loads(repository.path_for("legacy").read_text(encoding="utf-8")) == legacy


def test_repository_ignores_invalid_archives_and_sorts_newest_first(tmp_path):
    repository = ArchiveRepository(tmp_path)
    repository.save({"run_id": "older", "completed_at": "2026-01-01T00:00:00+00:00"})
    repository.save({"run_id": "newer", "completed_at": "2026-01-02T00:00:00+00:00"})
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    assert [item["run_id"] for item in repository.list()] == ["newer", "older"]
    assert repository.get("missing") is None


def test_redacting_sink_preserves_full_payload_but_removes_secrets():
    captured = []
    hook = RedactingTraceHook(lambda event, data: captured.append((event, data)))

    hook("llm.start", {
        "operation_id": "llm-1", "task_id": "planner", "model": "test",
        "messages": [{"role": "user", "content": "private prompt"}],
        "max_tokens": 100,
        "headers": {"Authorization": "Bearer top-secret", "X-Request-ID": "req-1"},
    })
    hook("tool.complete", {
        "operation_id": "tool-1", "task_id": "search", "output": {"raw": "paper"},
        "elapsed_ms": 10, "output_size": 1, "total_tokens": 12,
        "api_key": "sk-1234567890",
    })

    assert captured[0][1]["messages"][0]["content"] == "private prompt"
    assert captured[0][1]["max_tokens"] == 100
    assert "Authorization" not in captured[0][1]["headers"]
    assert captured[0][1]["headers"]["X-Request-ID"] == "req-1"
    assert captured[1][1]["output"] == {"raw": "paper"}
    assert captured[1][1]["total_tokens"] == 12
    assert "api_key" not in captured[1][1]


def test_redacting_sink_redacts_secret_values_under_innocent_keys():
    captured = []
    hook = RedactingTraceHook(lambda event, data: captured.append(data))

    hook("survey.error", {
        "error": "request failed with Authorization: Bearer abcdefghijklmnop",
        "prompt_tokens": 4,
    })

    assert captured[0]["error"] == "<redacted>"
    assert captured[0]["prompt_tokens"] == 4


def test_redacting_sink_removes_session_tokens_but_keeps_token_metrics():
    captured = []
    hook = RedactingTraceHook(lambda event, data: captured.append(data))

    hook("llm.complete", {
        "session_token": "credential",
        "nested": {"service_access_token": "credential"},
        "max_tokens": 100,
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "total_tokens": 50,
    })

    assert "session_token" not in captured[0]
    assert captured[0]["nested"] == {}
    assert captured[0]["max_tokens"] == 100
    assert captured[0]["total_tokens"] == 50


def test_redacting_sink_redacts_structured_tool_call_arguments():
    captured = []
    hook = RedactingTraceHook(lambda event, data: captured.append(data))

    hook("llm.complete", {
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": {"query": "few shot", "api_key": "secret"},
            },
        }],
    })

    arguments = captured[0]["tool_calls"][0]["function"]["arguments"]
    assert arguments["query"] == "few shot"
    assert "api_key" not in arguments
