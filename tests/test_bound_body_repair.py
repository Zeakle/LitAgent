"""Replay real repair permissions and the SDK-only mocked end-to-end consumer."""

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from model_config_helpers import load_legacy_mini_config as load_config
from test_evaluation_wire_contract import AnalysisTransport

from litagent.body_quality import analysis_repair_diagnostics, bound_repair_diagnostics
from litagent.contracts import build_config_summary
from litagent.observability.recorder import ArchiveRepository
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget
from scripts import run_bound_body_repair as repair


def real_case():
    """Load automatic observations, preserving full selected original sources."""
    return json.loads(
        (Path(__file__).parent / "fixtures/repair-cause-real.json").read_text()
    )


def parent_fixture(tmp_path):
    """Create a complete parent ledger with three earlier and three newer requests."""
    source = real_case()
    source.pop("provenance")
    source["report"]["metadata"]["approval"] = {
        "actor": "old-human",
        "status": "approved",
    }
    parent = tmp_path / "evaluation-parent"
    parent.mkdir()
    atomic_json(parent / "input.json", source)
    budget = CostBudget(max_tokens=200000)
    budget.configure_strict(halt_on_unknown_usage=True)
    before = budget.snapshot()
    before.update(
        used_tokens=60,
        call_count=3,
        strict_phase=2,
        requests_sent=3,
        phase_usage=[0, 0, 60],
    )
    after = {
        **before,
        "used_tokens": 120,
        "call_count": 6,
        "requests_sent": 6,
        "phase_usage": [0, 0, 120],
    }
    events = []
    for index in range(3):
        data = {
            "operation_id": f"old-{index}",
            "task_id": "evaluation.initial",
            "actual_model": repair.MODEL,
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
            "usage_available": True,
        }
        events += [
            {"event": "llm.start", "data": data},
            {"event": "llm.complete", "data": data},
        ]
    (parent / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    config = load_config()
    plan = {
        "max_tokens": 200000,
        "max_requests": 13,
        "source_budget": before,
        "config": build_config_summary(config, "strict_auto_v2"),
        "hashes": {
            str((parent / "input.json").resolve()): repair.file_sha(
                parent / "input.json"
            )
        },
        "historical_before_diagnostic": {
            s: {
                "charged": 500 if s == "survey" else 0,
                "requests": 30 if s == "survey" else 0,
                "usage_complete": s != "jev",
            }
            for s in ["survey", "judge", "jev"]
        },
        "run_id": parent.name,
        **{k: False for k in ["generation", "search", "repair", "judge", "jev"]},
    }
    plan["binding"] = digest(plan)
    atomic_json(parent / "manifest.json", plan)
    atomic_json(
        parent / "checkpoint.json",
        {
            "binding": plan["binding"],
            "status": "completed",
            "usage_complete": True,
            "budget": after,
            "report": source["report"],
        },
    )
    return parent, config, source


def test_real_host_science_rejection_cannot_expand_supported_edit_permissions():
    source = real_case()
    report = source["report"]
    details = report["evaluation"]["faithfulness"]["details"]
    before = copy.deepcopy(details)
    assert analysis_repair_diagnostics(details) == []
    assert [d["claim_id"] for d in bound_repair_diagnostics(details)] == [
        "claim_8",
        "claim_12",
    ]
    targets = LitAgent._strict_repair_targets(
        report["evaluation"], report["survey"], "academic_v2", "zh-CN"
    )
    assert len(targets) == 2
    assert all(t["source_fragment_id"].endswith(":fragment:1") for t in targets)
    assert details == before


@pytest.mark.parametrize("mode", ["actual_semantic_failure", "legacy"])
def test_true_analysis_failure_and_legacy_rows_still_authorize_supported_depth(mode):
    details = real_case()["report"]["evaluation"]["faithfulness"]["details"]
    for row in details["analysis_check"]["sections"]:
        if mode == "legacy":
            row.pop("model_adequate", None)
        else:
            row["model_adequate"] = False
    assert {d["claim_id"] for d in analysis_repair_diagnostics(details)} == {
        "claim_7",
        "claim_11",
    }


class RepairTransport(AnalysisTransport):
    """Mock provider decisions, not host repair compilation or full evaluation."""

    def __init__(self, fault=None):
        super().__init__("unsupported" if fault == "unsupported" else None)
        self.repair_fault = fault

    async def chat(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        if "response_schema_contract" in payload:
            payload["schema"] = kwargs["response_format"]["json_schema"]["schema"]
        if "repair_targets" in payload:
            result = {"patches": {}}
            for fid, t in payload["repair_targets"].items():
                scientific = (
                    "心脏MRI" in t["source_quote"] or t["section_key"] == "methods"
                )
                source = real_case()
                eid = (
                    "arxiv:2610.08109:abstract:claim:593317f2de12"
                    if scientific
                    else "arxiv:2506.08612:abstract:claim:95800934074d"
                )
                card = next(
                    cid
                    for cid, c in source["report"]["metadata"]["fact_cards"].items()
                    if c["evidence_id"] == eid
                )
                text = (
                    "在心脏MRI重建中，冻结视觉基础模型表示可在低数据条件下改善重建性能。"
                    if scientific
                    else "VIPriors挑战要求模型在少量样本下从头训练，不允许迁移学习，因此其结果只覆盖该受限训练设置。"
                )
                if (
                    self.repair_fault in {"duplicate", "semantic_duplicate"}
                    and not scientific
                ):
                    c = next(
                        c
                        for c in source["report"]["evaluation"]["faithfulness"][
                            "details"
                        ]["assertions"]
                        if c["claim_id"] == "claim_11"
                    )
                    text = c["claim_text"]
                    if self.repair_fault == "semantic_duplicate":
                        text = text.removeprefix("现有").replace("（如医学图像）", "")
                    card = next(
                        cid
                        for cid, c2 in source["report"]["metadata"][
                            "fact_cards"
                        ].items()
                        if c2["evidence_id"] == c["evidence_ids"][0]
                    )
                result["patches"][fid] = {
                    "first_sentence": {"text": text, "fact_card_ids": [card]},
                    "additional_sentences": [],
                }
                if self.repair_fault == "inline":
                    result["patches"][fid]["first_sentence"]["text"] += f" [F:{card}]"
            if self.repair_fault == "scope":
                result["patches"]["unknown-fragment"] = next(
                    iter(result["patches"].values())
                )
            return SimpleNamespace(content=json.dumps(result))
        response = await super().chat(messages, **kwargs)
        if "allowed_citation_aliases" in payload:
            result = json.loads(response.content)

            def prose(text):
                return re.sub(r"\[E:[^\]]+\]", "", text).strip(" .。\n")

            # Reuse the real provider's prior atomic extraction for unchanged
            # source fragments; the transport fixture is not a second extractor.
            observed = {
                prose(c["source_quote"]): c["claim_text"]
                for c in real_case()["report"]["evaluation"]["faithfulness"]["details"][
                    "assertions"
                ]
            }
            for unit in payload["source_units"]:
                for fragment in unit["source_fragments"]:
                    text = prose(fragment["text"])
                    result["fragments"][fragment["source_fragment_id"]]["assertions"][
                        0
                    ]["claim_text"] = observed.get(text, text)
                    if self.repair_fault == "semantic_duplicate" and text.startswith(
                        "针对Transformer的核心集"
                    ):
                        result["fragments"][fragment["source_fragment_id"]][
                            "assertions"
                        ][0]["claim_text"] = next(
                            c["claim_text"]
                            for c in real_case()["report"]["evaluation"][
                                "faithfulness"
                            ]["details"]["assertions"]
                            if c["claim_id"] == "claim_11"
                        )
            response.content = json.dumps(result)
        elif "analyses" in payload.get("schema", {}).get("properties", {}):
            result = json.loads(response.content)
            result["analyses"] = [
                {
                    "section_key": k,
                    "adequate": True,
                    "claim_ids": ids,
                    "explanation": "Controlled complete analytical protocol fixture.",
                }
                for k, ids in payload["analysis_claim_ids_by_section"].items()
            ]
            result["limitations"] = {
                "adequate": True,
                "claim_ids": payload["analysis_claim_ids_by_section"]["open_problems"],
                "explanation": "Controlled source-bound protocol fixture.",
            }
            response.content = json.dumps(result)
        return response


def client_factory(monkeypatch, wire, *, fault=None):
    """Retain the production client reservations and actual lifecycle events."""
    transport = RepairTransport(fault)

    def make(budget, hook):
        client = repair.RepairClient(
            "https://offline.invalid/v1",
            repair.MODEL,
            api_key="offline",
            cost_budget=budget,
            trace_hook=hook,
        )

        async def create(**kwargs):
            wire.append(copy.deepcopy(kwargs))
            response = await transport.chat(
                kwargs["messages"],
                **{k: v for k, v in kwargs.items() if k != "messages"},
            )
            return SimpleNamespace(
                model="wrong-model" if fault == "model" else repair.MODEL,
                _request_id=f"offline-{len(wire)}",
                usage=(
                    None
                    if fault == "unknown_usage"
                    else SimpleNamespace(prompt_tokens=10, completion_tokens=10)
                ),
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
            client._client.chat.completions, "create", AsyncMock(side_effect=create)
        )
        return client

    return make


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "unsupported",
        "duplicate",
        "semantic_duplicate",
        "scope",
        "unknown_usage",
        "model",
        "inline",
    ],
)
async def test_real_repair_and_complete_evaluation_keep_failed_candidates_uncommitted(
    tmp_path, monkeypatch, fault
):
    parent, config, source = parent_fixture(tmp_path)
    original = (parent / "checkpoint.json").read_bytes()
    root = tmp_path / "repair-once"
    plan = repair.prepare(root, parent, config)
    assert plan["historical_before"]["survey"] == {
        "charged": 620,
        "requests": 36,
        "usage_complete": True,
    }
    wire = []
    factory = client_factory(monkeypatch, wire, fault=fault)
    assert (await repair.run(root, config))["paid_calls"] == 0
    monkeypatch.chdir(tmp_path)
    repo = ArchiveRepository(tmp_path / "runs")
    result = await repair.run(
        root, config, live=True, client_factory=factory, repository=repo
    )
    assert result["status"] == "completed"
    assert result["budget"]["requests_sent"] == len(wire)
    candidate = json.loads((root / "candidate.json").read_text())
    assert candidate["committed"] is (fault is None)
    assert result["report"]["delivery"]["publishable"] is (fault is None)
    assert (parent / "checkpoint.json").read_bytes() == original
    assert len(result["report"]["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert result["report"]["metadata"]["verification"] == {
        "version": "optional-verification-v1",
        "judge": False,
        "jev": False,
    }
    assert "approval" not in result["report"]["metadata"]
    if fault in {None, "unsupported", "semantic_duplicate"}:
        assert len(wire) == 4
        assert candidate["candidate_evaluation"]["faithfulness"]["details"]["coverage"][
            "complete"
        ]
        targets = json.loads((root / "targets.json").read_text())
        body = source["report"]["survey"]
        text = candidate["candidate"]
        # Every protected interval, including both neighboring supported facts, survives.
        cursor = 0
        for t in targets:
            assert body[cursor : t["source_start"]] in text
            cursor = t["source_end"]
        assert text.endswith(body[cursor:])
        if fault == "semantic_duplicate":
            assert candidate["reason_code"] == "strict_repair_duplicate_content"
            assert candidate["candidate_quality"]["failed_metrics"] == [
                "body_repetition"
            ]
    else:
        assert len(wire) == 1
    if fault is not None:
        assert result["report"]["survey"] == source["report"]["survey"]
    if fault == "unknown_usage":
        assert result["usage_complete"] is False
    again = await repair.run(
        root, config, live=True, client_factory=factory, repository=repo
    )
    assert again == result


@pytest.mark.asyncio
async def test_last_shared_repair_retains_first_cost_and_counter(tmp_path, monkeypatch):
    parent, config, _ = parent_fixture(tmp_path)
    first = tmp_path / "first"
    repair.prepare(first, parent, config)
    repo = ArchiveRepository(tmp_path / "runs")
    wire = []
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        previous = await repair.run(
            first,
            config,
            live=True,
            client_factory=client_factory(monkeypatch, wire, fault="inline"),
            repository=repo,
        )
    assert previous["reason"] == "strict_repair_host_content_invalid"
    second = tmp_path / "second"
    plan = repair.prepare(second, parent, config, prior_repair=first)
    assert plan["source_budget"]["requests_sent"] == 1
    assert plan["source_budget"]["used_tokens"] == 20
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        result = await repair.run(
            second,
            config,
            live=True,
            client_factory=client_factory(monkeypatch, wire),
            repository=repo,
        )
    assert result["report"]["delivery"]["publishable"]
    assert result["budget"]["requests_sent"] == 5
    assert result["budget"]["used_tokens"] == 100
    assert len(result["report"]["metadata"]["targeted_repair_state"]["attempts"]) == 2
    assert len(result["report"]["metadata"]["evidence_rewrite_attempts"]) == 2
    assert len(wire) == 5
    request = json.loads(wire[1]["messages"][1]["content"])
    assert "[F:" not in request["current_draft"]
    assert "[F:" not in request["located_feedback"]
    assert (
        request["sources"] == json.loads(wire[0]["messages"][1]["content"])["sources"]
    )
    assert all(
        "[F:" not in t["source_quote"] and t["original_fact_card_ids"]
        for t in request["repair_targets"].values()
    )
    with pytest.raises(ValueError, match="prior_invalid"):
        repair.prepare(tmp_path / "third", parent, config, prior_repair=second)


@pytest.mark.parametrize(
    "fault", ["unknown_usage", "refund", "source_sha", "report", "requests"]
)
def test_changed_or_uncertain_parent_never_receives_new_allowance(tmp_path, fault):
    parent, config, _ = parent_fixture(tmp_path)
    state = json.loads((parent / "checkpoint.json").read_text())
    if fault == "unknown_usage":
        state["usage_complete"] = False
    elif fault == "refund":
        state["budget"]["used_tokens"] -= 1
        state["budget"]["phase_usage"][2] -= 1
    elif fault == "source_sha":
        payload = json.loads((parent / "input.json").read_text())
        payload["query"] += " changed"
        atomic_json(parent / "input.json", payload)
    elif fault == "report":
        state["report"]["survey"] += " changed"
    else:
        state["budget"]["requests_sent"] += 1
    atomic_json(parent / "checkpoint.json", state)
    with pytest.raises(ValueError):
        repair.prepare(tmp_path / "new", parent, config)


@pytest.mark.asyncio
async def test_bound_identity_receipt_and_interruption_prevent_reissue(
    tmp_path, monkeypatch
):
    parent, config, _ = parent_fixture(tmp_path)
    root = tmp_path / "one"
    plan = repair.prepare(root, parent, config)
    with pytest.raises(FileExistsError):
        repair.prepare(tmp_path / "two", parent, config)
    atomic_json(
        root / "checkpoint.json",
        {"binding": plan["binding"], "status": "started_no_replay"},
    )
    factory = AsyncMock()
    state = await repair.run(root, config, live=True, client_factory=factory)
    assert state["status"] == "interrupted_no_replay" and not state["usage_complete"]
    assert (
        state["conservative_tokens"] == 500000 and state["conservative_requests"] == 11
    )
    factory.assert_not_called()
    atomic_json(Path(plan["receipt"]), {"binding": "wrong"})
    with pytest.raises(ValueError, match="identity_changed"):
        await repair.run(root, config, live=True, client_factory=factory)


@pytest.mark.asyncio
async def test_transport_request_cap_prevents_sdk_call(monkeypatch):
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    snap = budget.snapshot()
    snap["requests_sent"] = 11
    budget.restore(snap)
    client = repair.RepairClient(
        "https://offline.invalid/v1",
        repair.MODEL,
        api_key="offline",
        cost_budget=budget,
    )
    sdk = AsyncMock()
    monkeypatch.setattr(client._client.chat.completions, "create", sdk)
    try:
        with pytest.raises(ValueError, match="bound_repair_request_limit"):
            await client.chat([{"role": "user", "content": "bounded"}])
    finally:
        await client.close()
    sdk.assert_not_called()
