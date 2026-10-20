"""Check frozen preparation recovery without granting a new paid budget."""

import copy
import json

import pytest
from test_extraction_record_identity import real_case

from litagent.agents.extraction_strategy import build_extraction_context
from litagent.config import load_config
from litagent.context.templates import wrap_xml
from litagent.evidence import collect_ledger
from litagent.orchestrator.task_graph import TaskGraph, TaskStatus
from scripts.rebuild_extraction_preparation import known_preparation_usage, rebuild


def source():
    fixture = real_case()
    paper = fixture["paper"]
    context, _ = build_extraction_context(paper)
    opening = {
        "operation_id": "op-1",
        "task_id": "extract",
        "model": fixture["response_model"],
        "messages": [
            {"role": "system", "content": "Original public extraction contract"},
            {"role": "user", "content": wrap_xml("paper", context)},
        ],
    }
    completion = {
        **opening,
        "actual_model": fixture["response_model"],
        "finish_reason": "stop",
        "usage_available": True,
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20,
        "content": json.dumps(fixture["response"]),
    }
    graph = {
        "tasks": {},
        "dependencies": {
            "relevance_gate": [],
            "extract": ["relevance_gate"],
            "adversarial_review": ["extract"],
        },
    }
    for tid, status, kind in [
        ("relevance_gate", "done", "relevance_gate"),
        ("extract", "failed", "extractor"),
        ("adversarial_review", "skipped", "adversarial_review"),
    ]:
        graph["tasks"][tid] = {
            "task_id": tid,
            "description": tid,
            "agent_type": kind,
            "input": {},
            "status": status,
            "timeout_ms": 120000,
        }
    graph["tasks"]["extract"]["error"] = "extraction_contract_invalid:duplicate_record"
    return {
        "run_id": "public-parent",
        "query": "Compare metric-learning methods",
        "graph": graph,
        "nodes": {"worker:relevance_gate": {"output": [paper]}},
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        "events": [
            {"event": "llm.start", "data": opening},
            {"event": "llm.complete", "data": completion},
        ],
        "report": {"metadata": {"search_source_outcomes": []}},
    }


@pytest.mark.asyncio
async def test_received_preparation_rebuild_keeps_parent_and_actual_fields(tmp_path):
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(source()))
    before = path.read_bytes()
    result = await rebuild(path, load_config())
    assert path.read_bytes() == before
    assert result["inherited_budget"]["used_tokens"] == 20
    assert result["inherited_budget"]["requests_sent"] == 1
    assert result["inherited_budget"]["max_tokens"] == 500000
    assert result["inherited_budget"]["stop_reason"] is None
    graph = TaskGraph.from_snapshot(result["graph"])
    assert graph.get_task("extract").status == TaskStatus.DONE
    assert graph.get_task("adversarial_review").status == TaskStatus.PENDING
    ledger = collect_ledger(result["extractions"])
    assert len(ledger) == 3 and {
        i["fact_fields"]["value"] for i in ledger.values()
    } == {"8.5%", "4.8%", "76.7%"}
    assert all(
        c["semantic_status"] == "not_verified" for c in result["fact_cards"].values()
    )
    assert result["scope"]["source_failure_retained"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "mismatched_task",
        "mismatched_model",
        "duplicate",
        "missing_event",
        "aggregate_drift",
    ],
)
def test_unknown_or_unreconciled_received_usage_is_not_reused(fault):
    data = source()
    if fault == "unknown":
        data["events"].append({"event": "llm.failed", "data": {}})
    if fault == "mismatched_task":
        data["events"][1]["data"]["task_id"] = "synthesis"
    if fault == "mismatched_model":
        data["events"][0]["data"]["model"] = "other"
    if fault == "duplicate":
        data["events"].append(copy.deepcopy(data["events"][1]))
    if fault == "missing_event":
        data["events"].pop()
    if fault == "aggregate_drift":
        data["usage"]["total_tokens"] = 99
    with pytest.raises(ValueError, match="preparation_"):
        known_preparation_usage(data)


@pytest.mark.asyncio
async def test_changed_original_context_is_rejected_without_pdf_download(tmp_path):
    data = source()
    data["events"][0]["data"]["messages"][1]["content"] += "altered source"
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source_unrecoverable"):
        await rebuild(path, load_config())


def test_preparation_accounting_uses_original_stage_and_restores_caller_context():
    from litagent.observability.context import get_task_id, reset_task_id, set_task_id

    token = set_task_id("evaluation.reconstruction_caller")
    try:
        _, _, budget = known_preparation_usage(source())
        assert budget["phase_usage"] == [20, 0, 0]
        assert get_task_id() == "evaluation.reconstruction_caller"
    finally:
        reset_task_id(token)


@pytest.mark.asyncio
async def test_rebuild_keeps_candidate_ranking_when_responses_complete_in_reverse(
    tmp_path,
):
    data = source()
    second = copy.deepcopy(data["nodes"]["worker:relevance_gate"]["output"][0])
    second.update(paper_id="fixture:second", title="Second public source")
    data["nodes"]["worker:relevance_gate"]["output"].append(second)
    opening = copy.deepcopy(data["events"][0])
    opening["data"]["operation_id"] = "op-2"
    context, _ = build_extraction_context(second)
    opening["data"]["messages"][1]["content"] = wrap_xml("paper", context)
    completion = copy.deepcopy(data["events"][1])
    completion["data"]["operation_id"] = "op-2"
    completion["data"]["messages"] = copy.deepcopy(opening["data"]["messages"])
    data["events"] = [data["events"][0], opening, completion, data["events"][1]]
    data["usage"] = {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(data))
    result = await rebuild(path, load_config())
    assert [item["paper_id"] for item in result["extractions"]] == [
        "fixture:tadam",
        "fixture:second",
    ]
    assert result["inherited_budget"]["requests_sent"] == 2
