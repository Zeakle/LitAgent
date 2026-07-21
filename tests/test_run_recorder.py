import json

from litagent.observability.recorder import ArchiveRepository, RedactingTraceHook, RunRecorder


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
    assert artifact["nodes"]["worker:search_arxiv_q0"]["input"]["query"] == "few-shot"
    assert artifact["nodes"]["tool:tool-1"]["output"] == [{"title": "Paper"}]
    assert repository.path_for("run-1").is_file()

    restored = repository.get("run-1")
    assert restored is not None
    assert restored["report"]["survey"] == "draft"
    assert json.loads(repository.path_for("run-1").read_text(encoding="utf-8"))["run_id"] == "run-1"


def test_repository_ignores_invalid_archives_and_sorts_newest_first(tmp_path):
    repository = ArchiveRepository(tmp_path)
    repository.save({"run_id": "older", "completed_at": "2026-01-01T00:00:00+00:00"})
    repository.save({"run_id": "newer", "completed_at": "2026-01-02T00:00:00+00:00"})
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    assert [item["run_id"] for item in repository.list()] == ["newer", "older"]
    assert repository.get("missing") is None


def test_redacting_sink_excludes_full_prompt_and_tool_payloads():
    captured = []
    hook = RedactingTraceHook(lambda event, data: captured.append((event, data)))

    hook("llm.start", {
        "operation_id": "llm-1", "task_id": "planner", "model": "test",
        "messages": [{"role": "user", "content": "private prompt"}],
    })
    hook("tool.complete", {
        "operation_id": "tool-1", "task_id": "search", "output": {"raw": "paper"},
        "elapsed_ms": 10, "output_size": 1,
    })

    assert captured[0][1] == {"operation_id": "llm-1", "task_id": "planner", "model": "test"}
    assert captured[1][1] == {
        "operation_id": "tool-1", "task_id": "search", "elapsed_ms": 10, "output_size": 1
    }
