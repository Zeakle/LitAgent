"""Exercise fixed low reasoning through native workers, SDK and durable accounting."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import AdversarialConfig
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.writing_v2 import WriterError, WriterSession, schema_for


@pytest.mark.asyncio
async def test_client_real_sdk_does_not_drop_effort_or_reasoning_details():
    """Reach the old real client directly, independently of new Worker signatures."""
    sent = []

    def transport(request):
        body = json.loads(request.content)
        sent.append(body)
        assert body.get("reasoning_effort") == "low"
        return httpx.Response(
            200,
            json={
                "id": "direct-low",
                "object": "chat.completion",
                "created": 0,
                "model": "pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                    "completion_tokens_details": {"reasoning_tokens": 70},
                },
            },
        )

    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1", "pro", api_key="offline"
    )
    await owner.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    try:
        response = await owner.chat(
            [{"role": "user", "content": "offline"}],
            max_completion_tokens=20000,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "enabled"}},
            transport_max_attempts=1,
        )
        assert response.usage["reasoning_tokens"] == 70 and len(sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("review_effort", [None, "low"])
async def test_real_workers_share_frozen_effort_but_reviewer_can_omit(
    case, tmp_path, review_effort
):
    """The native path consumes parameters; Reviewer does not inherit Pro's effort."""
    provider = Provider(case)
    synthesis = SynthesisWorker(
        provider,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=case.selection)
        ),
        thinking_mode="enabled",
        native_reasoning_effort="low",
        run_id_provider=lambda: "r",
        repair_state_path_provider=lambda: tmp_path / "r.json",
    )
    mode = "enabled" if review_effort else "disabled"
    reviewer = ReviewerWorker(
        provider,
        config=AdversarialConfig(thinking_mode=mode),
        native_reasoning_effort=review_effort,
    )
    worker = AdversarialReviewWorker(
        provider, synthesis, reviewer, thinking_mode="enabled"
    )
    task = SubTask(
        "adversarial_review",
        "Offline native reasoning",
        "adversarial_review",
        input_data={
            "query": case.packet["query"],
            "report_language": "en",
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "generation_strategy": "body_summary_v2",
            "verification": {"judge": False, "jev": False},
            "upstream_results": {},
        },
    )
    result = await worker.execute(task)
    assert len(provider.calls) == 7
    assert all(kwargs["reasoning_effort"] == "low" for _, kwargs in provider.calls[:6])
    last = provider.calls[-1][1]
    assert last.get("reasoning_effort") == review_effort
    if review_effort is None:
        assert "reasoning_effort" not in last
    assert result["writer_document"]["generation_reasoning_effort"] == "low"
    journal = json.loads((tmp_path / "r.json.writer-v2.json").read_text())
    assert journal["generation_reasoning_effort"] == "low"
    assert journal["calls"]["reviewer"]["reasoning_effort"] == review_effort


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["missing", None, 0, 70, True, -1, 101])
async def test_actual_sdk_preserves_effort_and_optional_reasoning_usage_once(
    case, tmp_path, detail
):
    """Mock HTTP only; the real SDK, native contract and CostBudget stay in the path."""
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    sent, events = [], []
    thought = "NOT_SCIENTIFIC_EVIDENCE"

    def transport(request):
        body = json.loads(request.content)
        sent.append(body)
        assert body["model"] == "pro" and body["reasoning_effort"] == "low"
        assert body["thinking"] == {"type": "enabled"}
        assert body["max_completion_tokens"] == 20000 and "max_tokens" not in body
        assert budget._reserved_tokens == strict_request_reservation(
            body["messages"], 20000, body["response_format"]
        )
        usage = {"prompt_tokens": 20, "completion_tokens": 100, "total_tokens": 120}
        if detail != "missing":
            usage["completion_tokens_details"] = {"reasoning_tokens": detail}
        return httpx.Response(
            200,
            json={
                "id": "low",
                "object": "chat.completion",
                "created": 0,
                "model": "pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(case.body),
                            "reasoning_content": thought,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )

    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "lite",
        api_key="offline",
        cost_budget=budget,
        trace_hook=lambda e, d: events.append((e, d)),
    )
    await owner.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="enabled",
        reasoning_effort="low",
    )
    try:
        assert (
            await session.call(
                "body",
                "Preserve complete evidence.",
                case.packet,
                schema_for("body", case.aliases),
            )
            == case.body
        )
        entry = session.journal["calls"]["body"]
        # The SDK normalizes raw JSON booleans before exposing its typed object.
        # The consumer guard cannot reconstruct that original wire type.
        parsed_detail = 1 if detail is True else detail
        valid = type(parsed_detail) is int and 0 <= parsed_detail <= 100
        assert entry["reasoning_effort"] == "low"
        assert entry["usage"].get("reasoning_tokens") == (
            parsed_detail if valid else None
        )
        if valid:
            assert entry["usage"]["reasoning_usage_available"] is True
        elif detail in ("missing", None):
            assert (
                "reasoning_tokens" not in entry["usage"]
                and "reasoning_usage_available" not in entry["usage"]
            )
        else:
            assert entry["usage"]["reasoning_usage_available"] is False
        assert entry["input_sha256"] == digest(
            [
                sent[0]["messages"],
                sent[0]["response_format"],
                {"max_completion_tokens": 20000},
                {"thinking": {"type": "enabled"}},
                {"reasoning_effort": "low"},
            ]
        )
        complete = next(data for event, data in events if event == "llm.complete")
        assert complete["reasoning_effort"] == "low"
        assert complete.get("reasoning_tokens") == (parsed_detail if valid else None)
        assert complete["total_tokens"] == budget.used == 120
        assert (
            budget.snapshot()["requests_sent"] == len(sent) == 1
            and budget._reserved_tokens == 0
        )
        assert thought not in json.dumps(session.journal) and thought not in json.dumps(
            events
        )
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "mode,effort",
    [
        ("disabled", "low"),
        ("provider_default", "low"),
        ("enabled", True),
        ("enabled", "high"),
        ("enabled", {}),
    ],
)
def test_invalid_native_effort_does_not_open_or_send(case, tmp_path, mode, effort):
    """Only the fixed explicit low configuration belongs to this native contract."""
    p = Provider(case)
    with pytest.raises(WriterError, match="reasoning"):
        WriterSession(
            p,
            tmp_path / "r.json",
            "r",
            case.packet,
            case.aliases,
            thinking_mode=mode,
            reasoning_effort=effort,
        )
    assert not p.calls and not (tmp_path / "r.json.writer-v2.json").exists()


def test_effort_changes_checkpoint_identity_without_rewriting_history(case, tmp_path):
    """An enabled journal with implicit effort cannot be reinterpreted as explicit low."""
    p = Provider(case)
    s = WriterSession(
        p, tmp_path / "r.json", "r", case.packet, case.aliases, thinking_mode="enabled"
    )
    s.save()
    before = s.journal_path.read_bytes()
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(
            p,
            tmp_path / "r.json",
            "r",
            case.packet,
            case.aliases,
            thinking_mode="enabled",
            reasoning_effort="low",
        )
    assert s.journal_path.read_bytes() == before


@pytest.mark.asyncio
async def test_summary_and_two_shared_declined_repairs_inherit_low(case, tmp_path):
    """Low changes a provider parameter, never the shared repair allowance."""
    p = Provider(case, decline=True)
    s = WriterSession(
        p,
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="enabled",
        reasoning_effort="low",
    )
    await s.summaries(case.body, "summary_initial")
    targets = {
        "t": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
        }
    }
    for _ in range(2):
        with pytest.raises(WriterError, match="repair_declined"):
            await s.patch(case.body, targets, cause="evaluation", claim_count=5)
    with pytest.raises(WriterError):
        await s.patch(case.body, targets, cause="evaluation", claim_count=5)
    assert len(s.state["attempts"]) == 2 and len(p.calls) == 3
    assert all(kwargs["reasoning_effort"] == "low" for _, kwargs in p.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["synthesis", "reviewer"])
async def test_native_parameter_cannot_be_silently_ignored_in_old_strategy(case, kind):
    """Prevent an apparently configured experiment from falling through an old path."""
    p = Provider(case)
    worker = (
        SynthesisWorker(p, thinking_mode="enabled", native_reasoning_effort="low")
        if kind == "synthesis"
        else ReviewerWorker(
            p,
            config=AdversarialConfig(thinking_mode="enabled"),
            native_reasoning_effort="low",
        )
    )
    task = SubTask(
        kind,
        "Legacy request",
        kind,
        input_data={"query": "q", "generation_strategy": "single_v1"},
    )
    with pytest.raises(ValueError, match="native_reasoning"):
        await worker.execute(task)
    assert not p.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", [True, -1, 101, 0, 70])
async def test_consumer_checks_actual_sdk_object_value_without_inventing_usage(detail):
    """Exercise validation separately from the SDK's earlier wire-type coercion."""
    budget = CostBudget()
    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1", "pro", api_key="offline", cost_budget=budget
    )
    await owner.close()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=100,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=detail),
        ),
        model="pro",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="{}", tool_calls=[], reasoning_content=""
                ),
                finish_reason="stop",
            )
        ],
    )
    owner._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        ),
        close=AsyncMock(),
    )
    try:
        result = await owner.chat([{"role": "user", "content": "offline"}])
        valid = type(detail) is int and 0 <= detail <= 100
        assert result.usage["reasoning_usage_available"] is valid
        assert result.usage.get("reasoning_tokens") == (detail if valid else None)
        assert budget.used == 120
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["high", False, {}, []])
async def test_bad_call_override_has_no_request_or_repair_charge(
    case, tmp_path, effort
):
    """Invalid Reviewer overrides are journaled as not sent, rather than charged unknown."""
    p = Provider(case)
    p._cost_budget = CostBudget()
    p._cost_budget.configure_strict(halt_on_unknown_usage=True)
    session = WriterSession(
        p,
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="enabled",
        reasoning_effort="low",
    )
    before = p._cost_budget.snapshot()
    with pytest.raises(WriterError, match="reasoning_configuration_invalid"):
        await session.call(
            "override",
            "Only evidence.",
            case.packet,
            schema_for("body", case.aliases),
            reasoning_effort=effort,
        )
    assert session.journal["calls"]["override"]["status"] == "rejected_not_sent"
    assert (
        p._cost_budget.snapshot() == before
        and not p.calls
        and not session.state["attempts"]
    )


def test_reasoning_metadata_is_redacted_without_double_billing_in_langfuse():
    """The optional subset belongs to metadata, never another billed usage category."""
    from unittest.mock import MagicMock

    from litagent.observability.recorder import RedactingTraceHook
    from litagent.observability.tracing import LangFuseTracer

    tracer = LangFuseTracer(host="", public_key="", secret_key="")
    tracer._client = object()
    tracer._root = MagicMock()
    hook = RedactingTraceHook(tracer, payload_mode="metadata_only")
    data = {
        "operation_id": "low",
        "model": "pro",
        "reasoning_effort": "low",
        "reasoning_tokens": 70,
        "reasoning_usage_available": True,
        "prompt_tokens": 20,
        "completion_tokens": 100,
        "total_tokens": 120,
        "usage_available": True,
        "content": "PRIVATE",
        "reasoning_content": "PRIVATE",
    }
    hook("llm.start", data)
    hook("llm.complete", data)
    kwargs = tracer._root.start_observation.return_value.update.call_args.kwargs
    assert (
        kwargs["metadata"]["reasoning_tokens"] == 70
        and kwargs["metadata"]["reasoning_effort"] == "low"
    )
    assert kwargs["usage_details"] == {
        "prompt_tokens": 20,
        "completion_tokens": 100,
        "total_tokens": 120,
    }
    assert "PRIVATE" not in str(tracer._root.mock_calls)


@pytest.mark.asyncio
async def test_actual_008_empty_output_stays_rejected_and_not_backfilled(
    case, tmp_path
):
    """Low cannot reinterpret historical length termination as a usable body."""
    from pathlib import Path

    from litagent.writing_v2 import body_section_schema

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/native-thinking-cap-real-008.json"
        ).read_text()
    )
    p = Provider(case)
    original = p.chat

    async def transport(messages, **kwargs):
        result = await original(messages, **kwargs)
        result.content = fixture["content"]
        result.usage = dict(fixture["usage"])
        result.finish_reason = fixture["finish_reason"]
        return result

    p.chat = transport
    session = WriterSession(
        p,
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="enabled",
        reasoning_effort="low",
    )
    with pytest.raises(WriterError, match="writer_output_truncated"):
        await session.call(
            "body:taxonomy",
            "Preserve facts.",
            case.packet,
            body_section_schema("taxonomy", case.aliases),
        )
    entry = session.journal["calls"]["body:taxonomy"]
    assert (
        entry["raw"] == "" and entry["raw_sha256"] == fixture["source_response_sha256"]
    )
    assert (
        entry["usage"]["completion_tokens"] == 20000
        and "reasoning_tokens" not in entry["usage"]
    )
    assert len(p.calls) == 1 and not session.state["attempts"]
