"""Run the real runner/evaluators on a frozen report with a transport-only mock."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_body_analysis import analysis_document, render_analysis
from test_body_quality_comparison import prepared
from test_evaluation_wire_contract import AnalysisTransport
from test_inventory_diagnostic import ProbeClient
from test_strict_automatic import frozen

from litagent.eval.scoped import build_source_fragments
from litagent.observability.recorder import ArchiveRepository
from litagent.report_audit import atomic_json
from litagent.report_render import build_report_source_units
from litagent.safety.budget import CostBudget
from scripts import run_frozen_evaluation as evaluation
from scripts import run_inventory_diagnostic as diagnostic


def transport_factory(monkeypatch, wire, clients, fault=None, *, actual_model=None):
    """Mock only the provider transport and preserve real event accounting."""
    transport = AnalysisTransport(fault)

    def factory(budget, hook):
        client = evaluation.EvaluationClient(
            "https://offline.invalid/v1",
            diagnostic.MODEL,
            api_key="offline",
            cost_budget=budget,
            trace_hook=hook,
        )
        clients.append(client)

        async def sdk_create(**kwargs):
            wire.append(copy.deepcopy(kwargs))
            response = await transport.chat(
                kwargs["messages"],
                **{k: v for k, v in kwargs.items() if k != "messages"},
            )
            return SimpleNamespace(
                model=actual_model or diagnostic.MODEL,
                _request_id=f"request-{len(wire)}",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=response.content, tool_calls=None
                        ),
                    )
                ],
            )

        monkeypatch.setattr(
            client._client.chat.completions, "create", AsyncMock(side_effect=sdk_create)
        )
        return client

    return factory


async def evaluation_fixture(tmp_path):
    """Construct a terminal parent whose events describe the actual frozen prose."""
    source, config, manifest = prepared(tmp_path, formal_enabled=False)
    base = frozen()
    report = copy.deepcopy(base["report"])
    cards = report["metadata"]["fact_cards"]
    report["survey"], bindings = render_analysis(analysis_document(cards), cards)
    report["metadata"].update(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        review_structure="academic_v2",
        report_language="en",
        review_methodology={},
        writing_bindings=bindings,
        coverage_requirements=[
            {"requirement_id": "question", "kind": "question", "text": "EViT speed"}
        ],
        approval={"status": "approved", "actor": "old-human"},
    )
    report_sha = diagnostic.file_sha_text(report["survey"])
    slot = source / "vit-baseline"
    atomic_json(
        slot / "checkpoint.json", {"report": report, "report_sha256": report_sha}
    )
    budgets = {
        s: {"charged": 0, "requests": 0, "usage_complete": True}
        for s in ("survey", "judge", "jev")
    }
    atomic_json(
        source / "comparison.json",
        {
            "binding": manifest["binding"],
            "status": "stopped",
            "pending": None,
            "budgets": budgets,
        },
    )
    atomic_json(
        source / "terminal-summary.json",
        {
            "binding": manifest["binding"],
            "budgets": budgets,
            "report_sha256": report_sha,
            "cumulative_after": manifest["historical"]["services"],
        },
    )
    units = [
        {
            "source_unit_id": u.unit_id,
            "scope_context": "",
            "citation_aliases": ["e1"],
            "source_fragments": [
                {
                    "source_fragment_id": f["source_fragment_id"],
                    "text": f["source_quote"].replace("[E:a]", "[E:e1]"),
                    "citation_aliases": ["e1"],
                }
                for f in build_source_fragments(u)
            ],
        }
        for u in build_report_source_units(report["survey"])
    ]
    payload = {"source_units": units, "schema": {}, "allowed_citation_aliases": ["e1"]}
    (slot / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "llm.start",
                "data": {
                    "task_id": "evaluation.initial",
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                },
            }
        )
        + "\n"
    )
    diagnostic_root = tmp_path / "probes"
    diagnostic.prepare(diagnostic_root, source, config)
    await diagnostic.run(
        diagnostic_root,
        config,
        live=True,
        client_factory=lambda b, h: ProbeClient(b, h, None),
    )
    archive_root = tmp_path / "source-archives"
    archive_path = archive_root / f"{source.name}-vit-baseline.json"
    results = {
        "extract": [
            {
                "paper_id": "evit",
                "claims": ["EViT speed"],
                "evidence_items": list(base["ledger"].values()),
            }
        ],
        "adversarial_review": {
            **report["metadata"],
            "evidence_selection": {
                "valid": True,
                "section_evidence_ids": {"methods": ["a"]},
            },
        },
    }
    atomic_json(
        archive_path,
        {
            "run_id": f"{source.name}-vit-baseline",
            "report": report,
            "nodes": {
                k: {"kind": "worker", "task_id": k, "output": v}
                for k, v in results.items()
            },
        },
    )
    root = tmp_path / "evaluation-only"
    plan = evaluation.prepare(root, diagnostic_root, config, archive_root=archive_root)
    return root, config, plan, archive_path


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "incomplete_inventory"])
async def test_only_evaluation_runs_and_old_report_and_approval_stay_unchanged(
    tmp_path, monkeypatch, fault
):
    root, config, plan, old_path = await evaluation_fixture(tmp_path)
    original = old_path.read_bytes()
    wire, clients = [], []
    factory = transport_factory(monkeypatch, wire, clients, fault)

    assert (await evaluation.run(root, config))["paid_calls"] == 0
    monkeypatch.chdir(tmp_path)
    repo = ArchiveRepository(tmp_path / "new-archives")
    result = await evaluation.run(
        root, config, live=True, client_factory=factory, repository=repo
    )
    assert result["status"] == "completed"
    report = result["report"]
    assert diagnostic.file_sha_text(report["survey"]) == plan["source_report_sha256"]
    assert report["metadata"]["evaluation_only"]["approval_inherited"] is False
    assert "approval" not in report["metadata"]
    assert not report["delivery"]["publishable"]
    assert result["budget"]["requests_sent"] == 3 + len(wire)
    assert result["budget"]["used_tokens"] == 60 + 20 * len(wire)
    assert old_path.read_bytes() == original
    assert repo.get(plan["run_id"])["report"]["survey"] == report["survey"]
    assert len(wire) == (1 if fault else 3)
    if fault:
        assert report["evaluation"]["faithfulness"]["skipped"]
    else:
        assert report["evaluation"]["faithfulness"]["details"]["coverage"]["complete"]
        assert report["evaluation"]["faithfulness"]["score"] == 1.0
    again = await evaluation.run(
        root, config, live=True, client_factory=factory, repository=repo
    )
    assert again == result and len(clients) == 1


@pytest.mark.asyncio
async def test_started_attempt_never_reissues_requests_and_retains_allowance(tmp_path):
    root, config, plan, _ = await evaluation_fixture(tmp_path)
    atomic_json(
        root / "checkpoint.json",
        {
            "binding": plan["binding"],
            "status": "started_no_replay",
            "budget": plan["source_budget"],
        },
    )
    factory = AsyncMock()
    result = await evaluation.run(root, config, live=True, client_factory=factory)
    assert result["status"] == "interrupted_no_replay"
    assert (
        result["conservative_tokens"] == 200000
        and result["conservative_requests"] == 13
    )
    assert result["usage_complete"] is False
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_shared_request_cap_stops_before_sdk_transport(monkeypatch):
    """The smaller experiment cap includes prior diagnostic requests."""
    budget = CostBudget(max_tokens=200000)
    budget.configure_strict(halt_on_unknown_usage=True)
    snapshot = budget.snapshot()
    snapshot["requests_sent"] = 13
    budget.restore(snapshot)
    client = evaluation.EvaluationClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    sdk = AsyncMock()
    monkeypatch.setattr(client._client.chat.completions, "create", sdk)
    try:
        with pytest.raises(ValueError, match="evaluation_diagnostic_request_limit"):
            await client.chat([{"role": "user", "content": "bounded"}])
    finally:
        await client.close()
    sdk.assert_not_called()
    assert budget.snapshot() == snapshot


@pytest.mark.asyncio
async def test_continuation_spends_only_the_reconciled_parent_remainder(
    tmp_path, monkeypatch
):
    root, config, plan, old_path = await evaluation_fixture(tmp_path)
    wire, clients = [], []
    factory = transport_factory(monkeypatch, wire, clients)
    repo = ArchiveRepository(tmp_path / "new-archives")
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        parent = await evaluation.run(
            root, config, live=True, client_factory=factory, repository=repo
        )
    parent_bytes = (root / "checkpoint.json").read_bytes()
    successor = tmp_path / "continuation"
    new = evaluation.prepare(
        successor,
        tmp_path / "probes",
        config,
        archive_root=old_path.parent,
        prior_evaluation=root,
    )
    assert new["source_budget"] == parent["budget"]
    assert new["source_budget"]["used_tokens"] == 120
    assert new["source_budget"]["requests_sent"] == 6
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        result = await evaluation.run(
            successor, config, live=True, client_factory=factory, repository=repo
        )
    assert result["budget"]["used_tokens"] == 180
    assert result["budget"]["requests_sent"] == 9
    assert (
        result["report"]["metadata"]["evaluation_only"]["prior_experiment_requests"]
        == 6
    )
    assert len(wire) == 6 and len(clients) == 2
    assert (root / "checkpoint.json").read_bytes() == parent_bytes
    assert (
        diagnostic.file_sha_text(result["report"]["survey"])
        == plan["source_report_sha256"]
    )
    with pytest.raises(ValueError, match="successor_already_reserved"):
        evaluation.prepare(
            tmp_path / "other-successor",
            tmp_path / "probes",
            config,
            archive_root=old_path.parent,
            prior_evaluation=root,
        )
    receipt = json.loads(Path(new["continuation"]["receipt"]).read_text())
    receipt["root"] = "another-root"
    atomic_json(Path(new["continuation"]["receipt"]), receipt)
    with pytest.raises(ValueError, match="successor_receipt_invalid"):
        await evaluation.run(
            successor, config, live=True, client_factory=factory, repository=repo
        )
    assert len(wire) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "unknown_usage",
        "refund",
        "request_count",
        "wrong_prose",
        "source_changed",
        "event_model",
    ],
)
async def test_continuation_rejects_unknown_or_inconsistent_prior_accounting(
    tmp_path, monkeypatch, fault
):
    root, config, plan, old_path = await evaluation_fixture(tmp_path)
    factory = transport_factory(monkeypatch, [], [])
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        state = await evaluation.run(
            root,
            config,
            live=True,
            client_factory=factory,
            repository=ArchiveRepository(tmp_path / "new-archives"),
        )
    if fault == "unknown_usage":
        state["usage_complete"] = False
    elif fault == "refund":
        state["budget"]["used_tokens"] -= 1
        state["budget"]["phase_usage"][2] -= 1
    elif fault == "request_count":
        state["budget"]["requests_sent"] += 1
    elif fault == "wrong_prose":
        state["report"]["survey"] += "Changed prose."
    elif fault == "source_changed":
        payload = json.loads((root / "input.json").read_text())
        payload["query"] = "Another question"
        atomic_json(root / "input.json", payload)
    else:
        events = [
            json.loads(x) for x in (root / "events.jsonl").read_text().splitlines()
        ]
        next(e["data"] for e in events if e["event"] == "llm.complete")[
            "actual_model"
        ] = "other"
        (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    atomic_json(root / "checkpoint.json", state)
    with pytest.raises(ValueError, match="frozen_evaluation_"):
        evaluation.prepare(
            tmp_path / "next",
            tmp_path / "probes",
            config,
            archive_root=old_path.parent,
            prior_evaluation=root,
        )


@pytest.mark.asyncio
async def test_actual_model_mismatch_stops_after_one_charged_request(
    tmp_path, monkeypatch
):
    root, config, _, _ = await evaluation_fixture(tmp_path)
    wire = []
    factory = transport_factory(monkeypatch, wire, [], actual_model="unexpected-model")
    monkeypatch.chdir(tmp_path)
    result = await evaluation.run(
        root,
        config,
        live=True,
        client_factory=factory,
        repository=ArchiveRepository(tmp_path / "new-archives"),
    )
    assert len(wire) == 1
    assert result["budget"]["requests_sent"] == 4
    assert result["budget"]["used_tokens"] == 80
    assert result["budget"]["stop_reason"] == "strict_worker_failed"
    assert not result["report"]["delivery"]["publishable"]
