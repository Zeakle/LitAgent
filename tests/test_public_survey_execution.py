"""Exercise privacy and lifecycle pairing in public Survey execution views."""

import json

from scripts.export_survey_benchmark_runs import project_execution


def test_public_projection_excludes_private_payloads_and_preserves_repeated_spans():
    private = "private-execution-canary"
    original = "replay-few-shot-metric-external-abstract-r1"
    run_id = original + "-retry-0123456789ab"
    started = "2026-11-26T04:00:00+00:00"
    finished = "2026-11-26T04:00:02+00:00"
    row = {
        "run_id": run_id,
        "case_id": "few-shot-metric",
        "profile_id": "external-abstract",
        "repetition": 1,
        "source": "retry",
        "status": "failed",
        "delivery": "blocked",
        "timing_scope": "summary_checkpoint",
        "downstream_ms": 2000,
        "evaluation": {"faithfulness": {"status": "unverified", "score": 0}},
        "usage": {"total_tokens": 123, "usage_uncertain": True},
        "stages_ms": {"evaluation.post_rewrite": [400, 500]},
        "reasons": ["AccountOverdueError " + private],
    }

    def event(name, at=started, **data):
        return {"event": name, "at": at, "data": {**data, "input": private}}

    def span(name, at):
        return event(name, at, task_id="evaluation.post_rewrite", name_override=private)

    actions = [event("benchmark.summary_checkpoint", source_run_id=original)]
    for _ in range(2):
        start = span("subspan.start", started)
        start["data"]["name"] = "evaluation.post_rewrite"
        actions.extend((start, span("subspan.end", finished)))
    actions.extend(
        event(name, task_id="extract", operation_id=private, request_id=private)
        for name in ("llm.start", "llm.complete")
    )
    archive = {
        "started_at": started,
        "completed_at": finished,
        "query": private,
        "report": private,
        "trace": private,
        "graph": {
            "tasks": {
                "extract": {
                    "agent_type": "extractor",
                    "status": "done",
                    "input": private,
                }
            },
            "dependencies": {"extract": []},
        },
        "nodes": {
            "worker:extract": {
                "kind": "worker",
                "task_id": "extract",
                "status": "completed",
                "metadata": {"checkpoint_source_run_id": original, "secret": private},
            },
            f"llm:{private}": {
                "kind": "llm",
                "task_id": "extract",
                "status": "completed",
                "elapsed_ms": 400,
                "name": private,
                "output": private,
            },
        },
        "events": actions,
    }

    exported = project_execution(row, archive, {original, run_id})
    assert private not in json.dumps(exported)
    nodes = exported["graph"]["nodes"]
    assert nodes["worker:extract"]["elapsed_ms"] is None
    assert nodes["worker:extract"]["metadata"]["checkpoint_source_run_id"] == original
    assert nodes["llm:op-1"]["elapsed_ms"] == 400
    assert [
        exported["events"][index]["data"]["operation_id"]
        for index in (1, 2, 3, 4, 5, 6)
    ] == [
        "span-1",
        "span-1",
        "span-2",
        "span-2",
        "op-1",
        "op-1",
    ]
    metrics = exported["metrics"]
    assert metrics["faithfulness"]["score"] is None
    assert metrics["usage"]["reported_tokens"] == 123
    assert metrics["usage"]["billed_tokens"] is None
    assert metrics["stages_ms"]["evaluation.post_rewrite"] == [400, 500]
    assert metrics["duration_basis"] == "summary_checkpoint"
