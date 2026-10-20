"""Verify received-draft continuation without paid transport or counterfeit scores."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.observability.recorder import ArchiveRepository
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import (
    GroundedWritingError,
    _native_unit,
    build_fact_cards,
)
from litagent.quality import digest as quality_digest
from litagent.quality import (
    grounded_messages,
    native_task_identity,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report_traced,
    validate_received_native,
)
from litagent.report_audit import atomic_json, digest
from litagent.runner import Infra, LitAgent
from litagent.safety.budget import CostBudget, strict_request_reservation
from scripts import run_prepared_survey
from scripts import run_received_native as recovery


@pytest.fixture
def received_parent(tmp_path):
    """Use real public prose with synthetic accounting solely for boundary tests."""
    f = json.loads(
        (Path(__file__).parent / "fixtures/native-mixed-errors-real.json").read_text()
    )
    config = recovery.effective_config()
    parent = tmp_path / "known-parent"
    parent.mkdir()
    by_paper = {}
    for item in f["selected_items"].values():
        ext = by_paper.setdefault(
            item["paper_id"],
            {
                "paper_id": item["paper_id"],
                "title": item["paper_title"],
                "evidence_items": [],
                "claims": [],
            },
        )
        ext["evidence_items"].append(item)
        ext["claims"].append(item["text"])
    xs = list(by_paper.values())
    graph = TaskGraph()
    graph.add_task(SubTask("relevance_gate", "relevant papers", "relevance_gate"))
    graph.mark_running("relevance_gate")
    graph.mark_done(
        "relevance_gate", [{"paper_id": x["paper_id"], "title": x["title"]} for x in xs]
    )
    graph.add_task(
        SubTask("extract", "extract", "extractor"), depends_on=["relevance_gate"]
    )
    graph.mark_running("extract")
    graph.mark_done("extract", xs)
    data = {
        "query": f["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "review_structure": "academic_v2",
        "report_language": "zh-CN",
        "verification": recovery.options().verification.model_dump(mode="json"),
        "requested_methods": [],
        "coverage_constraints": [r["text"] for r in f["coverage_requirements"][1:]],
        "coverage_requirements": f["coverage_requirements"],
        "search_started_at": 1,
        "upstream_results": {"extract": xs},
        "methodology_results": {**graph.get_results(), "__search_queries__": {}},
    }
    graph.add_task(
        SubTask(
            "adversarial_review",
            "write",
            "adversarial_review",
            input_data=copy.deepcopy(data),
        ),
        depends_on=["extract"],
    )
    original = {
        "query": f["query"],
        "scope": {"relevant_papers": len(xs)},
        "graph": graph.to_snapshot(),
        "asset_hashes": {},
        "search_source_outcomes": [],
    }
    original["binding"] = digest(original)
    preparation = tmp_path / "preparation.json"
    atomic_json(preparation, original)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    before = budget.snapshot()
    before.update(
        used_tokens=100, call_count=1, requests_sent=1, phase_usage=[100, 0, 0]
    )
    after = {
        **before,
        "used_tokens": 160,
        "call_count": 3,
        "requests_sent": 3,
        "phase_usage": [100, 60, 0],
        "strict_phase": 1,
        "stop_reason": "strict_worker_failed",
    }
    frozen = parent / "frozen-files/pyproject.toml"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(Path("pyproject.toml").read_bytes())
    m = {
        "run_id": parent.name,
        "config": build_config_summary(config, "strict_auto_v2"),
        "inherited_budget": before,
        "hashes": {str(Path("pyproject.toml").resolve()): recovery.file_sha(frozen)},
        "preparation_path": str(preparation),
        "preparation_sha256": recovery.file_sha(preparation),
    }
    m["binding"] = digest(m)
    atomic_json(parent / "manifest.json", m)
    checkpoint = {
        "binding": m["binding"],
        "status": "failed_no_replay",
        "budget": after,
        "incremental_cost": {"requests": 2, "tokens": 60, "usage_complete": True},
    }
    atomic_json(parent / "checkpoint.json", checkpoint)
    raw_selection = {
        "candidate_count": len(f["selected_items"]),
        "selected_items": f["selected_items"],
        "section_evidence_ids": {
            k: v
            for k, v in f["selection"]["section_evidence_ids"].items()
            if k not in {"abstract", "conclusion"}
        },
        "method_by_section": {
            k: "cross_encoder"
            for k in f["selection"]["section_evidence_ids"]
            if k not in {"abstract", "conclusion"}
        },
        "omitted_count": 0,
        "estimated_tokens": 10000,
        "ranking_decisions": [],
    }
    selection = EvidenceSelection.from_dict(raw_selection)
    for key in ("abstract", "conclusion"):
        selection.section_evidence_ids[key] = list(selection.selected_items)
        selection.method_by_section[key] = "selected_union"
    events = [
        {
            "event": "worker.input",
            "data": {"task_id": "adversarial_review", "input": data},
        },
        {"event": "evidence.retrieve.complete", "data": raw_selection},
    ]
    for i, content in enumerate([f["candidate"], f["repair_response"]]):
        start = {
            "operation_id": f"op{i}",
            "task_id": "adversarial_review:synthesis:r1",
            "model": config.llm.model_for("generation"),
        }
        events.extend(
            [
                {"event": "llm.start", "data": start},
                {
                    "event": "llm.complete",
                    "data": {
                        **start,
                        "content": content,
                        "actual_model": start["model"],
                        "finish_reason": "stop",
                        "usage_available": True,
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                    },
                },
            ]
        )
    (parent / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )
    state = {
        "version": "shared-targeted-repair-limit-v1",
        "run_id": parent.name,
        "limit": 2,
        "attempts": [
            {
                "index": 0,
                "stage": "native_initial",
                "status": "failed",
                "input_sha256": "a" * 64,
                "response_sha256": hashlib.sha256(
                    f["repair_response"].encode()
                ).hexdigest(),
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "usage_available": True,
                },
            }
        ],
    }
    atomic_json(
        parent / "targeted-repair-state.json",
        {
            "state": state,
            "native_candidate": f["candidate"],
            "native_response": f["repair_response"],
            "native_binding": f["repair_binding"],
        },
    )
    atomic_json(parent / "completed-preparation.json", {"extractions": xs})
    cards = build_fact_cards(selection.selected_items)
    with pytest.raises(GroundedWritingError) as failed:
        render_grounded_report_traced(
            f["candidate"],
            cards,
            "academic_v2",
            "zh-CN",
            evidence_selection=selection,
            require_native=True,
            require_full_sources=True,
            require_scope_notices=True,
            analysis_mode=True,
        )
    messages, _ = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            "zh-CN",
            f["query"],
            coverage=f["coverage_requirements"],
            evidence_selection=selection,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    request, schema, binding = prepare_native_repair_request(
        messages,
        f["candidate"],
        failed.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language="zh-CN",
        evidence_selection=selection,
    )
    value = {
        "version": "received-native-recovery-input-v1",
        "parent_root": str(parent),
        "parent_hashes": {
            n: recovery.file_sha(parent / n)
            for n in [
                "manifest.json",
                "checkpoint.json",
                "events.jsonl",
                "targeted-repair-state.json",
                "completed-preparation.json",
            ]
        },
        "config": build_config_summary(config, "strict_auto_v2"),
        "options": recovery.options().model_dump(mode="json"),
        "source_manifest_binding": m["binding"],
        "source_run_id": parent.name,
        "inherited_budget": after,
        "remaining_requests": 61,
        "remaining_tokens": 499840,
        "remaining_repairs": 1,
        "candidate": f["candidate"],
        "source_repair_state": state,
        "fact_cards": cards,
        "selection": selection.to_dict(),
        "source_input": data,
        "diagnostics": failed.value.diagnostics,
        "native_messages": request,
        "native_schema": schema,
        "native_binding": binding,
        "native_reservation": strict_request_reservation(request, 20000, schema),
    }
    value["binding"] = digest(value)
    path = tmp_path / "received.json"
    atomic_json(path, value)
    return config, path, value, parent


@pytest.mark.parametrize(
    "fault",
    ["refund", "selection", "candidate", "unknown", "model", "old_code", "shape"],
)
def test_source_accounting_and_identity_mismatch_prevents_execution(
    received_parent, fault
):
    config, path, value, parent = received_parent
    if fault == "refund":
        value["inherited_budget"]["used_tokens"] = 0
    elif fault == "selection":
        value["selection"]["section_evidence_ids"]["methods"] = []
    elif fault == "candidate":
        value["candidate"] += " "
    elif fault == "old_code":
        (parent / "frozen-files/pyproject.toml").write_text("changed")
    elif fault == "shape":
        del value["source_input"]
    else:
        events = [
            json.loads(l) for l in (parent / "events.jsonl").read_text().splitlines()
        ]
        completed = next(e["data"] for e in events if e["event"] == "llm.complete")
        completed["usage_available" if fault == "unknown" else "actual_model"] = (
            False if fault == "unknown" else "wrong-model"
        )
        (parent / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        value["parent_hashes"]["events.jsonl"] = recovery.file_sha(
            parent / "events.jsonl"
        )
    value["binding"] = digest({k: v for k, v in value.items() if k != "binding"})
    atomic_json(path, value)
    with pytest.raises(ValueError):
        recovery.read_input(path, config)


@pytest.mark.asyncio
async def test_corrupted_executable_snapshot_cannot_start(received_parent, tmp_path):
    config, path, _, _ = received_parent
    root = tmp_path / "snapshot-child"
    manifest = recovery.prepare(root, path, config)
    original = Path(next(iter(manifest["hashes"])))
    recovery.snapshot_path(root, original).write_text("changed")
    with pytest.raises(ValueError, match="native_frozen_snapshot_changed"):
        await recovery.run(root, config, live=True)
    assert not (root / "checkpoint.json").exists()


@pytest.mark.asyncio
async def test_dry_run_and_interrupted_identity_send_no_request(
    received_parent, tmp_path, monkeypatch
):
    config, path, _, _ = received_parent
    root = tmp_path / "child"
    m = recovery.prepare(root, path, config)
    assert (await recovery.run(root, config))["status"] == "planned"
    assert m["review_rounds"] == 1 and m["inherited_budget"]["used_tokens"] == 160
    atomic_json(
        root / "checkpoint.json",
        {"binding": m["binding"], "status": "started_no_replay"},
    )
    monkeypatch.setattr(
        recovery, "LitAgent", lambda *a, **kw: pytest.fail("no second process")
    )
    assert (await recovery.run(root, config, live=True))[
        "status"
    ] == "interrupted_no_replay"


@pytest.mark.parametrize("fault", ["task", "counter", "source", "run"])
def test_worker_receipt_cannot_change_the_received_input(received_parent, fault):
    _, _, value, _ = received_parent
    receipt = recovery.resume_receipt(value, "new-run")
    task = copy.deepcopy(value["source_input"])
    if fault == "task":
        task["query"] += " changed"
    elif fault == "counter":
        receipt["repair_state"]["attempts"] = []
    elif fault == "source":
        receipt["source_run_id"] = []
    else:
        receipt["run_id"] = "wrong"
    receipt["binding_sha256"] = quality_digest(
        {k: v for k, v in receipt.items() if k != "binding_sha256"}
    )
    with pytest.raises(ValueError):
        validate_received_native(receipt, task, "new-run")


@pytest.mark.asyncio
@pytest.mark.parametrize("codec_pass", [False, True])
async def test_real_entry_uses_only_remaining_pro_repair_and_never_delivers_invalid_evaluation(
    received_parent, tmp_path, monkeypatch, codec_pass
):
    config, path, value, _ = received_parent
    monkeypatch.setenv("LLM_API_KEY", "offline")
    monkeypatch.setattr(run_prepared_survey, "OWNER_ROOT", tmp_path / "owners")
    monkeypatch.setattr(LitAgent, "_connect_infra", AsyncMock(return_value=Infra()))
    monkeypatch.setattr(
        recovery, "ArchiveRepository", lambda: ArchiveRepository(tmp_path / "runs")
    )
    import litagent.report_audit as audit_module
    from litagent.extraction_verification import ExtractionVerifier

    original_init = ExtractionVerifier.__init__
    monkeypatch.setattr(
        ExtractionVerifier,
        "__init__",
        lambda self, root=None, **kw: original_init(
            self, tmp_path / "extraction", **kw
        ),
    )

    def save_local(p, v):
        if str(p).startswith("artifacts/"):
            p = tmp_path / "isolated" / p
        return atomic_json(p, v)

    monkeypatch.setattr(audit_module, "atomic_json", save_local)

    class Trace:
        trace_id = "0" * 32
        trace_url = "http://offline.invalid/trace"

        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(recovery, "LangFuseTracer", Trace)
    requests = []

    def respond(request):
        p = json.loads(request.content)
        requests.append(p)
        if len(requests) == 1:
            assert p["model"] == config.llm.model_for("generation")
            assert (
                p["response_format"]["json_schema"]["name"] == "native_unit_repair_v5"
            )
            packet = json.loads(p["messages"][1]["content"])
            assert packet["current_candidate"] == json.loads(value["candidate"])
            content = {
                "can_repair": codec_pass,
                "patches": {
                    k: {
                        "text": _native_unit(packet["current_candidate"], t["path"])[
                            "text"
                        ],
                        "kind": "scientific",
                        "fact_card_ids": t["allowed_fact_card_ids"][:1],
                        "scope_notice_id": "none",
                    }
                    for k, t in packet["repair_targets"].items()
                },
            }
        else:
            assert p["model"] == config.llm.model_for("evaluation")
            content = {
                "score": 0.95,
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
            if len(requests) > 2:
                content = {}  # Invalid inventory must stop, never a fake pass.
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "created": 0,
                "object": "chat.completion",
                "model": p["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content),
                        },
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
        lambda **kw: AsyncOpenAI(
            **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ),
    )
    root = tmp_path / "live-child"
    recovery.prepare(root, path, config)
    result = await recovery.run(root, config, live=True)
    assert result.get("reason_type") not in {
        "NameError",
        "TypeError",
        "KeyError",
        "AttributeError",
    }
    assert (
        requests
        and sum(p["model"] == config.llm.model_for("generation") for p in requests) == 1
    )
    assert result["status"] in {"blocked", "failed_no_replay"}
    assert result["budget"]["requests_sent"] == 3 + len(requests)
    assert result["budget"]["used_tokens"] == 160 + 20 * len(requests)
    state = json.loads((root / "targeted-repair-state.json").read_text())["state"]
    assert len(state["attempts"]) == 2
    if codec_pass:
        assert any(p["model"] == config.llm.model_for("evaluation") for p in requests)
        artifact = json.loads((tmp_path / "runs/live-child.json").read_text())
        assert not artifact["report"]["delivery"]["publishable"]
        assert any(
            e["event"] == "llm.start" and e["data"]["task_id"].startswith("evaluation")
            for e in artifact["events"]
        )
        assert list((root / "evaluations").glob("*/progress.json"))
    count = len(requests)
    assert (
        await recovery.run(root, config, live=True) == result and len(requests) == count
    )
