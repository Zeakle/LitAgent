"""Exercise privacy and lifecycle pairing in public Survey execution views."""

import json
from hashlib import sha256

from scripts.export_survey_benchmark_runs import project_execution
from scripts.survey_benchmark_public_content import project_report


def test_public_projection_excludes_private_payloads_and_preserves_repeated_spans():
    private = "private-execution-canary"
    markdown = "# Example Survey\n\nExact archived text.\n"
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
        "report_sha256": sha256(markdown.encode()).hexdigest(),
    }

    def event(name, at=started, **data):
        return {"event": name, "at": at, "data": {**data, "input": private}}

    def span(name, at):
        return event(name, at, task_id="evaluation.post_rewrite", name_override=private)

    actions = [event("benchmark.summary_checkpoint", source_run_id=original)]
    for index in range(2):
        start = span("subspan.start", started)
        start["data"]["name"] = "evaluation.post_rewrite"
        start["data"]["input"] = {"assertion_count": index + 1, "messages": private}
        end = span("subspan.end", finished)
        end["data"]["output"] = {"faithfulness": {"score": (index + 1) / 2}}
        actions.extend((start, end))
    actions.extend(
        event(
            name,
            task_id="extract",
            operation_id=private,
            request_id=private,
            total_tokens=123,
        )
        for name in ("llm.start", "llm.complete")
    )
    actions.append(
        event(
            "fulltext.complete",
            finished,
            task_id="extract",
            paper_count=2,
            storage_path=private,
        )
    )
    archive = {
        "started_at": started,
        "completed_at": finished,
        "query": private,
        "report": {"survey": markdown, "private_metadata": private},
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
                "input": {"query": "public query", "api_key": private},
                "output": [
                    {"paper_id": "arxiv:1901.00001", "chunks": [private]},
                ],
            },
            f"llm:{private}": {
                "kind": "llm",
                "task_id": "extract",
                "status": "completed",
                "elapsed_ms": 400,
                "name": private,
                "input": {"model": "model-id", "messages": [private]},
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
    assert exported["report"]["markdown"] == markdown
    assert exported["report"]["source_sha256"] == row["report_sha256"]
    assert "<h1" in exported["report"]["html"]
    step_io = exported["step_io"]
    assert step_io["worker:extract"] == {
        "status": "reused",
        "input": {"query": "public query"},
        "output": [{"paper_id": "arxiv:1901.00001"}],
        "omitted": ["input", "output"],
    }
    assert step_io["subspan:span-1"]["input"] == {"assertion_count": 1}
    assert step_io["subspan:span-1"]["output"] == {"faithfulness": {"score": 0.5}}
    assert step_io["subspan:span-2"]["input"] == {"assertion_count": 2}
    assert step_io["subspan:span-2"]["output"] == {"faithfulness": {"score": 1.0}}
    assert step_io["llm:op-1"] == {
        "status": "recorded",
        "input": {"model": "model-id"},
        "output": {"total_tokens": 123},
        "omitted": ["input", "output"],
    }
    assert step_io["event:sequence:8"]["output"] == {
        "paper_count": 2,
        "task_id": "extract",
    }
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


def test_generation_failure_notice_is_not_report_prose():
    report = {
        "survey": "> The survey could not be completed.\n",
        "partial": True,
        "evaluation": {
            "faithfulness": {"details": {"skipped_reason": "final_report_unavailable"}}
        },
    }
    assert (
        project_report(
            {"report": report}, sha256(report["survey"].encode()).hexdigest()
        )
        is None
    )


def test_public_report_uses_ieee_numbers_and_references_without_mutating_source():
    markdown = "# Survey\n\nFirst [E:a]. Second [E:b]. First again [E:a2].\n"
    papers = [
        {
            "paper_id": "arxiv:2010.11929",
            "title": "A Study",
            "author_details": [{"given": "Alice", "family": "Chen"}],
            "year": 2020,
        },
        {
            "paper_id": "arxiv:2202.01301",
            "title": "The Second Study",
            "authors": ["Taylor Smith"],
            "year": 2022,
        },
    ]
    archive = {
        "run_id": "ieee-public",
        "report": {"survey": markdown},
        "nodes": {
            "worker:dedup": {"output": papers},
            "worker:extract": {
                "output": [
                    {
                        "evidence_items": [
                            {"evidence_id": eid, "paper_id": papers[index]["paper_id"]}
                            for eid, index in (("a", 0), ("b", 1), ("a2", 0))
                        ]
                    }
                ],
            },
        },
    }
    digest = sha256(markdown.encode()).hexdigest()

    report = project_report(archive, digest)

    assert report["markdown"] == markdown
    assert report["source_sha256"] == digest
    html = report["html"]
    assert 'First <a href="#citation-1">[1]</a>' in html
    assert 'Second <a href="#citation-2">[2]</a>' in html
    assert 'First again <a href="#citation-3">[1]</a>' in html
    assert '<h2 id="references">References</h2>' in html
    assert "<p>[1]" in html and "A. Chen, “A Study,”" in html
    assert "<p>[2]" in html and "Taylor Smith, “The Second Study,”" in html
    assert "[E:" not in html
