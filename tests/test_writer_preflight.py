"""Check sending boundaries using real writer consumers and SDK accounting."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.agents.synthesis import SynthesisWorker
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions
from litagent.writing_v2 import (
    BODY,
    CONTRACT,
    WriterError,
    WriterSession,
    schema_for,
    synthesize,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["markup", "empty", "capacity"])
async def test_invalid_body_cannot_purchase_summary(case, tmp_path, fault):
    """Local chapter defects stop synthesis before dependent chapters or summaries."""
    if fault == "markup":
        case.body["methods"][0]["sentences"][0]["text"] = "[E:borrowed] scientific text"
    elif fault == "empty":
        case.body["methods"][0]["sentences"][0]["text"] = "   "
    else:
        unit = case.body["methods"][0]["sentences"][0]
        case.body["methods"] = [{"sentences": [copy.deepcopy(unit)] * 25}] * 5
    provider = Provider(case)
    worker = SynthesisWorker(
        provider,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=case.selection)
        ),
        run_id_provider=lambda: "preflight",
        repair_state_path_provider=lambda: tmp_path / "r.json",
    )
    task = SubTask(
        "synthesis",
        "test",
        "synthesis",
        input_data={"query": case.packet["query"], "report_language": "en"},
    )
    with pytest.raises(WriterError, match="writer_(text_invalid|assertion_capacity)"):
        await synthesize(worker, task, case.cards, case.selection, {}, [])
    expected = BODY[: BODY.index("methods") + 1]
    assert len(provider.calls) == len(expected)
    journal = json.loads((tmp_path / "r.json.writer-v2.json").read_text())
    assert set(journal["calls"]) == {"body:" + section for section in expected}
    assert journal["failure"]["body_section"] == "methods"
    assert journal["failure"]["code"] == (
        "writer_assertion_capacity_exceeded"
        if fault == "capacity"
        else "writer_text_invalid"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["negative", "bool", "missing", "old", "duplicate", "section"]
)
async def test_target_identity_rejected_before_count_or_http(case, tmp_path, fault):
    """Malformed or stale edit locations cannot spend an attempt or target another unit."""
    p = Provider(case)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    targets = {
        "x": {
            "path": ["methods", 0, 0],
            "original": copy.deepcopy(case.body["methods"][0]["sentences"][0]),
        }
    }
    if fault == "negative":
        targets["x"]["path"][2] = -1
    elif fault == "bool":
        targets["x"]["path"][1] = False
    elif fault == "missing":
        targets["x"]["path"][2] = 100
    elif fault == "old":
        targets["x"]["original"]["text"] = "A previous revision."
    elif fault == "duplicate":
        targets["y"] = copy.deepcopy(targets["x"])
    else:
        targets["x"]["path"][0] = "abstract"
    with pytest.raises(WriterError, match="patch_identity_invalid"):
        await s.patch(case.body, targets, cause="evaluation", claim_count=5)
    assert not p.calls and not s.state["attempts"]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", [None, "length", "content_filter", "tool_calls"])
async def test_nonterminal_or_filtered_response_is_not_accepted(case, tmp_path, finish):
    """Even valid JSON is unusable when the provider did not finish normally."""
    p = Provider(case)
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(case.body),
            model="offline",
            finish_reason=finish,
            request_id="fixture-request",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )
    )
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(
        WriterError, match="writer_(output_truncated|finish_reason_invalid)"
    ):
        await s.call("body", "system", case.packet, schema_for("body", case.aliases))
    entry = s.journal["calls"]["body"]
    assert entry["request_id"] == "fixture-request"
    assert entry["raw_sha256"] == hashlib.sha256(entry["raw"].encode()).hexdigest()
    with pytest.raises(WriterError, match="no_replay"):
        await s.call("body", "system", case.packet, schema_for("body", case.aliases))
    assert p.chat.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", ["valid", "invalid_json", "unknown_usage", "cancelled"]
)
async def test_sdk_repair_owns_reservation_and_preserves_accounting(
    case, tmp_path, result
):
    """The real SDK sees a reservation throughout transport; release is not a refund."""
    import asyncio

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    sent, held = [], []

    def transport(request):
        sent.append(json.loads(request.content))
        held.append(budget._reserved_tokens)
        assert held[-1] > 20000
        with pytest.raises(ValueError, match="budget_exceeded"):
            budget.reserve(budget.remaining() + 1)
        if result == "cancelled":
            raise asyncio.CancelledError()
        raw = json.dumps(case.body)
        if result == "invalid_json":
            raw = raw[:-1]
        return httpx.Response(
            200,
            headers={"x-request-id": "sdk-request"},
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ],
                "usage": (
                    None
                    if result == "unknown_usage"
                    else {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    }
                ),
            },
        )

    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    await owner._client.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    session = WriterSession(owner, tmp_path / "r.json", "r", case.packet, case.aliases)
    token = set_task_id("evidence_rewrite")
    try:
        if result == "valid":
            assert (
                await session.call(
                    "repair", "system", case.packet, schema_for("body", case.aliases)
                )
                == case.body
            )
        else:
            with pytest.raises((ValueError, asyncio.CancelledError)):
                await session.call(
                    "repair", "system", case.packet, schema_for("body", case.aliases)
                )
        assert len(sent) == budget.snapshot()["requests_sent"] == 1
        assert budget._reserved_tokens == 0
        assert budget.used == (
            held[0] if result in {"cancelled", "unknown_usage"} else 30
        )
        if result == "invalid_json":
            diagnostic = session.journal["calls"]["repair"]["failure"]["diagnostics"][0]
            raw = session.journal["calls"]["repair"]["raw"]
            assert diagnostic["offset"] == len(raw)
            assert diagnostic["column"] == len(raw) + 1
        restored = WriterSession(
            owner, tmp_path / "r.json", "r", case.packet, case.aliases
        )
        with pytest.raises(WriterError, match="no_replay"):
            await restored.call(
                "repair", "system", case.packet, schema_for("body", case.aliases)
            )
        assert len(sent) == 1
    finally:
        reset_task_id(token)
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 10, "completion_tokens": 1},
        {"usage_available": True, "prompt_tokens": 10},
        {"usage_available": True, "prompt_tokens": True, "completion_tokens": 1},
        {"usage_available": True, "prompt_tokens": -1, "completion_tokens": 1},
    ],
)
async def test_incomplete_usage_cannot_be_accepted_as_known(case, tmp_path, usage):
    """A nonempty usage object is not evidence of complete token accounting."""
    p = Provider(case)
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(case.body), finish_reason="stop", usage=usage
        )
    )
    session = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="usage_unknown"):
        await session.call(
            "body", "system", case.packet, schema_for("body", case.aliases)
        )
    assert p.chat.await_count == 1
    assert session.journal["calls"]["body"]["failure"]["family"] == "interrupted"


def test_execution_guard_changes_only_v2_identity_and_rejects_old_journal(
    case, tmp_path, monkeypatch
):
    """History remains readable, while changed execution rules cannot resume old calls."""
    from litagent import writing_v2

    opts = {
        s: RunOptions(
            quality_policy="strict_auto_v2",
            body_quality="analysis_v1",
            generation_strategy=s,
        )
        for s in ("single_v1", "body_summary_v1", "body_summary_v2")
    }
    before = {s: o.fingerprint() for s, o in opts.items()}
    monkeypatch.setattr(writing_v2, "EXECUTION_CONTRACT", "changed-guard")
    after = {s: o.fingerprint() for s, o in opts.items()}
    assert before["single_v1"] == after["single_v1"]
    assert before["body_summary_v1"] == after["body_summary_v1"]
    assert before["body_summary_v2"] != after["body_summary_v2"]
    path = tmp_path / "r.json"
    old = {
        "contract": CONTRACT,
        "identity": digest([CONTRACT, case.packet, case.aliases]),
        "calls": {},
        "candidates": [],
    }
    journal = path.with_name(path.name + ".writer-v2.json")
    journal.write_text(json.dumps(old))
    with pytest.raises(WriterError, match="checkpoint_identity_conflict"):
        WriterSession(Provider(case), path, "r", case.packet, case.aliases)
    assert json.loads(journal.read_text()) == old


@pytest.mark.parametrize("count", [0, True, 129, -1])
def test_over_capacity_reassessment_is_not_clamped_to_eight_batches(
    case, tmp_path, count
):
    """A capacity failure must precede repair even with an otherwise empty budget."""
    session = WriterSession(
        Provider(case), tmp_path / "r.json", "r", case.packet, case.aliases
    )
    with pytest.raises(WriterError, match="capacity_exceeded"):
        session.reserve_followup([], {}, 1, count)
