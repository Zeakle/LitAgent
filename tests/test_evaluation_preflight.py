"""Preserve complete rejected evaluation inputs without paying or leaking them."""

import hashlib
import json
from unittest.mock import Mock

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.batched import BATCH_PROMPT, GLOBAL_PROMPT, request_size
from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.safety.budget import CostBudget, strict_request_reservation


def capture_hooks(llm):
    """Capture local full inputs and the metadata-only external projection."""
    local, public = [], []
    safe = RedactingTraceHook(lambda e, d: public.append((e, d)), "metadata_only")

    def hook(event, data):
        local.append((event, data))
        safe(event, data)

    llm._trace_hook = hook
    return local, public


def assert_reconstructable(detail):
    """Recompute request identity and complete output reservation from saved input."""
    assert (
        detail["input_sha256"]
        == hashlib.sha256(
            json.dumps(detail["messages"], ensure_ascii=False).encode()
        ).hexdigest()
    )
    assert detail["budget"]["reservation"] == strict_request_reservation(
        detail["messages"], detail["max_tokens"], detail["response_format"]
    )
    assert detail["status"] == "rejected_not_sent"


@pytest.mark.asyncio
async def test_inventory_rejection_saves_full_input_without_issuing_or_advancing_budget():
    survey, context = materials(42)
    llm = BatchLLM()
    budget = CostBudget()
    budget.configure_strict()
    llm._cost_budget = budget
    local, public = capture_hooks(llm)
    before = budget.snapshot()
    token = set_task_id("evaluation.initial")
    try:
        with pytest.raises(ScopedEvaluationError) as error:
            await evaluate_scoped(
                llm=llm,
                survey=survey,
                context=context,
                config=EvalConfig(context_tokens=32768),
            )
    finally:
        reset_task_id(token)
    assert str(error.value) == "evaluation_context_budget_exceeded"
    assert llm.calls == [] and budget.snapshot() == before
    assert len(local) == 1 and local[0][0] == "evaluation.request.rejected"
    detail = local[0][1]
    assert detail["phase"] == "inventory"
    assert_reconstructable(detail)
    payload = json.loads(detail["messages"][1]["content"])
    assert len(payload["source_units"]) == 42
    saved = error.value.details["stages"]["rejected_requests"][0]
    assert "messages" not in saved and saved["input_sha256"] == detail["input_sha256"]
    assert "messages" not in public[0][1] and "ViT uses" not in json.dumps(public)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["claim_capacity", "batch_capacity", "source_size"])
async def test_batch_planning_rejection_retains_complete_material_and_exact_prompt(
    failure,
):
    count = (
        129 if failure == "claim_capacity" else 17 if failure == "batch_capacity" else 1
    )
    survey, context = materials(count)
    config = EvalConfig(faithfulness_max_evidence_items=150)
    if failure == "claim_capacity":
        # Exercise batch capacity, independently of the inventory context ceiling.
        config.context_tokens = 262144
    if failure == "batch_capacity":
        config.strict_max_batches = 1
    if failure == "source_size":
        from litagent.quality import build_fact_cards, digest

        source = "ViT uses image patches. " * 8000
        context["evidence"]["a0"].update(text=source, supporting_text=source)
        context["fact_cards"] = build_fact_cards(context["evidence"])
        context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    llm = BatchLLM()
    local, _ = capture_hooks(llm)
    with pytest.raises(ScopedEvaluationError):
        await evaluate_scoped(llm=llm, survey=survey, context=context, config=config)
    assert len(llm.calls) == 1
    detail = local[0][1]
    assert detail["phase"] == "batch_planning"
    assert detail["messages"][0]["content"] == BATCH_PROMPT
    assert_reconstructable(detail)
    payload = json.loads(detail["messages"][1]["content"])
    assert len(payload["assertions"]) == count
    if failure == "source_size":
        assert next(iter(payload["original_supports"].values()))["text"] == source


@pytest.mark.asyncio
async def test_global_context_rejection_preserves_completed_batches_without_a_score():
    survey, context = materials(42)
    baseline = BatchLLM()
    await evaluate_scoped(
        llm=baseline, survey=survey, context=context, config=EvalConfig()
    )
    global_size = request_size(GLOBAL_PROMPT, baseline.calls[-1], 16384)
    earlier_sizes = [request_size(BATCH_PROMPT, p, 16384) for p in baseline.calls[1:-1]]
    limit = global_size - 1
    assert max(earlier_sizes) < limit
    config = EvalConfig(context_tokens=int(limit / 0.9))
    llm = BatchLLM()
    local, _ = capture_hooks(llm)
    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(llm=llm, survey=survey, context=context, config=config)
    assert len(llm.calls) == 4 and llm.batches == 3
    assert error.value.details["stages"]["processed_assertions"] == 42
    assert not error.value.details["coverage"]["complete"]
    assert "score" not in error.value.details
    assert local[-1][1]["phase"] == "global"
    assert_reconstructable(local[-1][1])


@pytest.mark.asyncio
async def test_run_budget_and_broken_observer_cannot_turn_refusal_into_a_call():
    survey, context = materials(1)
    llm = BatchLLM()
    budget = CostBudget(max_tokens=100)
    budget.configure_strict()
    llm._cost_budget = budget

    def broken(*args):
        raise OSError("unavailable local observer")

    llm._trace_hook = broken
    context["evaluation_checkpoint"] = broken
    before = budget.snapshot()
    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert str(error.value) == "evaluation_run_budget_exceeded"
    assert not llm.calls and budget.snapshot() == before
    assert error.value.details["stages"]["rejected_requests"][0]["phase"] == "inventory"


@pytest.mark.parametrize("mode", ["full_redacted", "metadata_only"])
def test_external_tracer_never_receives_rejected_scientific_payload(mode):
    tracer = LangFuseTracer(host="", public_key="", secret_key="")
    tracer._client = object()
    tracer._root = Mock()
    hook = RedactingTraceHook(tracer, mode)
    hook(
        "evaluation.request.rejected",
        {
            "task_id": "evaluation.initial",
            "phase": "global",
            "contract_version": "evaluation-request-diagnostics-v1",
            "input_sha256": "sha",
            "reason_code": "evaluation_context_budget_exceeded",
            "budget": {"reservation": 100},
            "max_tokens": 16384,
            "status": "rejected_not_sent",
            "messages": [{"content": "private_scientific_marker"}],
            "response_format": {"schema": "private_scientific_marker"},
        },
    )
    metadata = tracer._root.update.call_args.kwargs["metadata"]
    assert metadata["evaluation_request"]["input_sha256"] == "sha"
    assert metadata["evaluation_request"]["max_tokens"] == 16384
    assert "private_scientific_marker" not in json.dumps(metadata)
