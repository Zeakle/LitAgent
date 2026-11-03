"""Keep bounded failure evidence local across real evaluator consumers."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from test_batched_evaluation import BatchLLM, materials
from test_native_responses import owner_for
from test_step_timeouts import response

from litagent.config import load_config
from litagent.eval import ragas_eval
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.scoped import ScopedEvaluationError
from litagent.llm.routing import ModelBoundClient
from litagent.observability.recorder import RedactingTraceHook
from litagent.runner import LitAgent

CONTRACT = "evaluation-local-failure-v1"
LOCAL_KEY = "local_failure_diagnostic"
PRIVATE = "sk-PRIVATE_KEY_EXCEPTION_SENTINEL"
RAW_SOURCE = "PRIVATE_UNTRUSTED_PAPER_EXCERPT_SENTINEL"
MESSAGE = PRIVATE + RAW_SOURCE + "very_long_private_message" * 10000


def sha(value):
    """Hash exact report text with the production encoding."""
    return hashlib.sha256(value.encode()).hexdigest()


def failure(progress, *, cause=None):
    """Provide the actual scoped failure container without changing its details."""
    error = ScopedEvaluationError(
        "evaluation_contract_invalid",
        {
            "source_report_sha256": "0" * 64,
            "stages": progress,
            "coverage": {"complete": False},
        },
    )
    error.__cause__ = cause
    return error


def checkpoint_diagnostic(progress, *, survey="exact report", cause=None):
    """Call the helper while preserving object identities for mutation checks."""
    checkpoint = Mock()
    error = failure(progress, cause=cause)
    before = copy.deepcopy(error.details)
    before_bytes = json.dumps(error.details, ensure_ascii=False).encode()
    ragas_eval._checkpoint_failure(
        error,
        survey,
        {"quality_policy": "strict_auto_v2", "evaluation_checkpoint": checkpoint},
    )
    checkpoint.assert_called_once()
    saved = checkpoint.call_args.args[0]
    assert saved is not progress
    assert saved["batches"] is progress["batches"]
    assert LOCAL_KEY not in progress and error.details == before
    assert json.dumps(error.details, ensure_ascii=False).encode() == before_bytes
    assert {key: value for key, value in saved.items() if key != LOCAL_KEY} == progress
    diagnostic = saved[LOCAL_KEY]
    assert diagnostic["contract_version"] == CONTRACT
    assert diagnostic["source_report_sha256"] == sha(survey)
    return diagnostic


def assert_private_absent(value):
    """Reject private exception payloads at both disk and public boundaries."""
    serialized = json.dumps(value, ensure_ascii=False)
    assert PRIVATE not in serialized
    assert RAW_SOURCE not in serialized
    assert "very_long_private_message" not in serialized


def response_for_model(content, model):
    """Preserve the real stage model in the controlled provider envelope."""
    envelope = response(content).json()
    envelope["model"] = model
    return httpx.Response(200, json=envelope)


class FailureTransport:
    """Replace only HTTP, retaining SDK, scope, parser and accounting behavior."""

    def __init__(self, fault, *, fail_call=2):
        self.fault = fault
        self.fail_call = fail_call
        self.sent = []
        self.offline = BatchLLM()

    async def __call__(self, request):
        """Return real wire responses or fail at the selected transport boundary."""
        body = json.loads(request.content)
        self.sent.append(body)
        if len(self.sent) == self.fail_call:
            if self.fault == "exception":
                raise RuntimeError(MESSAGE)
            if self.fault == "cancel":
                raise asyncio.CancelledError(MESSAGE)
            if self.fault == "json":
                return response_for_model("{" + PRIVATE, body["model"])
            raise AssertionError("unexpected offline fault")
        generated = await self.offline.chat(
            body["messages"], response_format=body.get("response_format")
        )
        return response_for_model(generated.content, body["model"])


@pytest.mark.asyncio
async def test_real_scoped_failure_is_checkpointed_without_public_or_progress_mutation():
    survey, context = materials(1)
    transport = FailureTransport("exception")
    owner, budget, events = await owner_for(transport)
    snapshots, original_progress = [], []

    def checkpoint(progress):
        if LOCAL_KEY not in progress:
            original_progress.append(progress)
        snapshots.append(copy.deepcopy(progress))

    context["evaluation_checkpoint"] = checkpoint
    try:
        result = await RagasFaithfulnessEvaluator(
            load_config(), llm=ModelBoundClient(owner, "lite", "evaluation")
        ).evaluate(survey, context)
        assert result.skipped and result.score == 0
        assert result.details["skipped_reason"] == "evaluation_contract_invalid"
        assert not result.details["coverage"]["complete"]
        assert len(transport.sent) == 2
        assert all(request["model"] == "lite" for request in transport.sent)
        assert budget.snapshot()["requests_sent"] == 2
        assert budget.snapshot()["usage_uncertain"]
        assert budget.stop_reason == "strict_usage_unknown"
        assert budget.snapshot()["used_tokens"] > 20
        assert all(LOCAL_KEY not in progress for progress in original_progress)
        assert LOCAL_KEY not in result.details
        assert LOCAL_KEY not in result.details["stages"]
        saved = snapshots[-1]
        diagnostic = saved.pop(LOCAL_KEY)
        assert saved == result.details["stages"]
        assert diagnostic["source_report_sha256"] == sha(survey)
        assert diagnostic["contract_version"] == CONTRACT
        assert diagnostic["stage"] == "assessment"
        assert diagnostic["batch_index"] == 0
        assert diagnostic["batch_payload_sha256"] == saved["batches"][0]["input_sha256"]
        assert diagnostic["request_input_sha256"] is None
        types = [item["type"] for item in diagnostic["exception_chain"]]
        assert types[0].endswith("ScopedEvaluationError")
        assert any(item.endswith("RuntimeError") for item in types)
        assert any(
            frame["function"] == "__call__"
            for item in diagnostic["exception_chain"]
            for frame in item["frames"]
        )
        assert saved["batches"][-1]["status"] == "interrupted_no_replay"
        assert_private_absent(diagnostic)
        assert_private_absent(result.details)
        assert_private_absent(events)
        assert LOCAL_KEY not in json.dumps(events)
        assert LOCAL_KEY not in json.dumps(transport.sent)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_runner_real_save_progress_keeps_diagnostic_on_disk_and_out_of_events(
    tmp_path,
):
    survey, source_context = materials(1)
    transport = FailureTransport("exception")
    owner, budget, llm_events = await owner_for(transport)
    local_events, external_events = [], []
    redacted = RedactingTraceHook(
        lambda event, data: external_events.append((event, data)), "metadata_only"
    )

    def trace(event, data):
        local_events.append((event, copy.deepcopy(data)))
        redacted(event, data)

    config = load_config()
    agent = LitAgent(config, trace_hook=trace)
    agent._cost_budget = budget
    agent._evaluation_progress_root = tmp_path / "evaluation"
    agent._evaluators = [
        RagasFaithfulnessEvaluator(
            config, llm=ModelBoundClient(owner, "lite", "evaluation")
        )
    ]
    results = {
        "extract_0": [
            {
                "paper_id": item["paper_id"],
                "claims": [item["text"]],
                "evidence_items": [item],
            }
            for item in source_context["evidence"].values()
        ],
        "adversarial_review": {
            key: source_context[key]
            for key in (
                "quality_policy",
                "fact_cards",
                "evidence_snapshot_sha256",
                "coverage_requirements",
            )
        },
    }
    try:
        output = await agent._evaluate(
            query=source_context["query"],
            survey=survey,
            results=results,
            phase="initial",
        )
        paths = list((tmp_path / "evaluation").rglob("progress.json"))
        assert len(paths) == 1
        saved = json.loads(paths[0].read_text(encoding="utf-8"))
        diagnostic = saved["progress"][LOCAL_KEY]
        assert diagnostic["contract_version"] == CONTRACT
        assert (
            diagnostic["source_report_sha256"] == saved["report_sha256"] == sha(survey)
        )
        assert saved["budget"] == budget.snapshot()
        assert output["faithfulness"]["skipped"]
        assert (
            output["faithfulness"]["details"]["skipped_reason"]
            == "evaluation_contract_invalid"
        )
        assert len(transport.sent) == saved["budget"]["requests_sent"] == 2
        assert saved["budget"]["usage_uncertain"]
        assert any(event == "evaluation.batch" for event, _ in local_events)
        assert any(event == "subspan.end" for event, _ in local_events)
        for projected in (
            output,
            local_events,
            external_events,
            llm_events,
            transport.sent,
        ):
            assert LOCAL_KEY not in json.dumps(projected)
            assert_private_absent(projected)
        assert_private_absent(diagnostic)
    finally:
        await owner.close()


def deep_failure(depth, error_type=RuntimeError):
    """Produce traceback depth using real frames without reading their locals."""
    if depth:
        return deep_failure(depth - 1, error_type)
    raise error_type(MESSAGE)


def captured_failure(depth=40, error_type=RuntimeError):
    """Retain a private exception solely for bounded traceback projection."""
    try:
        deep_failure(depth, error_type)
    except RuntimeError as error:
        return error


def test_helper_bounds_cause_chain_frames_and_strings_without_rendering_exception():
    class OpaqueFailure(RuntimeError):
        """Detect any accidental string conversion of private exception data."""

        def __str__(self):
            raise AssertionError("diagnostic must not render exception messages")

    long_type = type("PrivateFailure" * 100, (OpaqueFailure,), {})
    errors = [captured_failure(error_type=long_type) for _ in range(7)]
    for current, cause in zip(errors, errors[1:]):
        current.__cause__ = cause
    errors[-1].__cause__ = errors[0]
    diagnostic = checkpoint_diagnostic(
        {"batches": [], "processed_assertions": 0}, cause=errors[0]
    )
    assert len(diagnostic["exception_chain"]) == 4
    assert diagnostic["chain_truncated"] is True
    assert any(entry["frames_truncated"] for entry in diagnostic["exception_chain"])
    for entry in diagnostic["exception_chain"]:
        assert set(entry) == {"type", "frames", "frames_truncated"}
        assert len(entry["type"]) <= 128
        assert len(entry["frames"]) <= 16
        for frame in entry["frames"]:
            assert set(frame) == {"file", "function", "line"}
            assert not Path(frame["file"]).is_absolute()
            assert len(frame["file"]) <= 128 and len(frame["function"]) <= 128
            assert type(frame["line"]) is int and frame["line"] > 0
    assert_private_absent(diagnostic)


def test_helper_stops_a_short_cause_cycle_without_duplicating_exceptions():
    error = failure({"batches": []})
    cause = RuntimeError(MESSAGE)
    error.__cause__ = cause
    cause.__cause__ = error
    checkpoint = Mock()
    ragas_eval._checkpoint_failure(
        error,
        "report",
        {"quality_policy": "strict_v1", "evaluation_checkpoint": checkpoint},
    )
    chain = checkpoint.call_args.args[0][LOCAL_KEY]["exception_chain"]
    assert len(chain) == 2
    assert chain[0]["type"].endswith("ScopedEvaluationError")
    assert chain[1]["type"].endswith("RuntimeError")
    assert_private_absent(chain)


@pytest.mark.parametrize(
    "fault", ["legacy", "no_callback", "not_callable", "no_stages", "bad_stages"]
)
def test_helper_only_checkpoints_strict_failures_with_local_callback_and_valid_stages(
    fault,
):
    checkpoint = Mock()
    context = {"quality_policy": "strict_auto_v2", "evaluation_checkpoint": checkpoint}
    error = failure({"batches": []}, cause=RuntimeError(MESSAGE))
    if fault == "legacy":
        context["quality_policy"] = "legacy"
    elif fault == "no_callback":
        context.pop("evaluation_checkpoint")
    elif fault == "not_callable":
        context["evaluation_checkpoint"] = "untrusted callback name"
    elif fault == "no_stages":
        error.details.pop("stages")
    else:
        error.details["stages"] = ["invalid stage container"]
    before = copy.deepcopy(error.details)
    ragas_eval._checkpoint_failure(error, "report", context)
    checkpoint.assert_not_called()
    assert error.details == before


def settlement_request(batch_sha, request_sha, survey="exact report"):
    """Represent an already settled local association without invoking an owner."""
    receipt = {
        "contract": "host-settled-assessment-receipt-v1",
        "stage": "assessment",
        "input_sha256": request_sha,
        "round_number": 3,
    }
    return {
        "phase": "assessment",
        "status": "response_received",
        "input_sha256": request_sha,
        "response": "{}",
        "semantic_conflict_settlement": {
            "receipt": receipt,
            "report_sha256": sha(survey),
            "response_sha256": sha("{}"),
            "input_sha256": request_sha,
            "batch_payload_sha256": batch_sha,
        },
    }


@pytest.mark.parametrize(
    "fault",
    [
        "inventory",
        "previous_batch",
        "incomplete",
        "duplicate",
        "request_mismatch",
        "report_mismatch",
        "response_mismatch",
        "round_mismatch",
        "no_response",
    ],
)
def test_http_identity_never_borrows_inventory_or_another_batch_settlement(fault):
    current, previous, request_hash = "a" * 64, "b" * 64, "c" * 64
    request = settlement_request(current, request_hash)
    if fault == "inventory":
        request["phase"] = "inventory"
    elif fault == "previous_batch":
        request["semantic_conflict_settlement"]["batch_payload_sha256"] = previous
    elif fault == "incomplete":
        request["semantic_conflict_settlement"].pop("receipt")
    elif fault == "request_mismatch":
        request["semantic_conflict_settlement"]["input_sha256"] = "e" * 64
    elif fault == "report_mismatch":
        request["semantic_conflict_settlement"]["report_sha256"] = "e" * 64
    elif fault == "response_mismatch":
        request["semantic_conflict_settlement"]["response_sha256"] = "e" * 64
    elif fault == "round_mismatch":
        request["semantic_conflict_settlement"]["receipt"]["round_number"] = 4
    elif fault == "no_response":
        request.pop("response")
    progress = {
        "batches": [
            {"index": 0, "input_sha256": previous, "status": "completed"},
            {"index": 1, "input_sha256": current, "status": "interrupted_no_replay"},
        ],
        "requests": (
            [request, copy.deepcopy(request)] if fault == "duplicate" else [request]
        ),
        "capacity": {"round": {"number": 3}},
    }
    diagnostic = checkpoint_diagnostic(progress)
    assert diagnostic["stage"] == "assessment"
    assert diagnostic["batch_index"] == 1
    assert diagnostic["batch_payload_sha256"] == current
    assert diagnostic["request_input_sha256"] is None


def test_matching_batch_settlement_retains_http_identity_and_separate_round_indexes():
    batch_hash, request_hash = "a" * 64, "c" * 64
    progress = {
        "batches": [
            {"index": 2, "input_sha256": batch_hash, "status": "invalid_response"}
        ],
        "requests": [settlement_request(batch_hash, request_hash)],
        "capacity": {"round": {"number": 3}},
        "inventory_recovery": {"attempt_index": 1},
    }
    diagnostic = checkpoint_diagnostic(progress)
    assert diagnostic["stage"] == "assessment"
    assert diagnostic["batch_index"] == 2
    assert diagnostic["batch_payload_sha256"] == batch_hash
    assert diagnostic["request_input_sha256"] == request_hash
    assert diagnostic["capacity_round_number"] == 3
    assert diagnostic["recovery_attempt_index"] == 1


@pytest.mark.parametrize(
    "index",
    [None, True, -1, PRIVATE * 1000, {"secret": MESSAGE}, 2.5],
    ids=["missing", "boolean", "negative", "long_string", "mapping", "float"],
)
def test_invalid_batch_index_cannot_add_unbounded_or_private_diagnostic_fields(index):
    diagnostic = checkpoint_diagnostic(
        {
            "batches": [
                {
                    "index": index,
                    "input_sha256": "a" * 64,
                    "status": "interrupted_no_replay",
                }
            ]
        }
    )
    assert diagnostic["batch_index"] is None
    assert_private_absent(diagnostic)


@pytest.mark.parametrize("status", ["started_no_replay", "interrupted_no_replay"])
def test_global_stage_never_reuses_last_completed_batch_identity(status):
    diagnostic = checkpoint_diagnostic(
        {
            "batches": [{"index": 0, "input_sha256": "a" * 64, "status": "completed"}],
            "global_check": status,
            "requests": [settlement_request("a" * 64, "b" * 64)],
        }
    )
    assert diagnostic["stage"] == "global"
    assert diagnostic["batch_index"] is None
    assert diagnostic["batch_payload_sha256"] is None
    assert diagnostic["request_input_sha256"] is None


def test_global_preflight_diagnostic_cannot_inherit_inventory_http_identity():
    diagnostic = checkpoint_diagnostic(
        {
            "batches": [],
            "global_preflight": {
                "status": "rejected_not_sent",
                "input_sha256": "a" * 64,
            },
            "requests": [
                {
                    "phase": "inventory",
                    "status": "response_received",
                    "input_sha256": "b" * 64,
                }
            ],
        }
    )
    assert diagnostic["stage"] == "global_preflight"
    assert diagnostic["batch_index"] is None
    assert diagnostic["batch_payload_sha256"] is None
    assert diagnostic["request_input_sha256"] is None


@pytest.mark.asyncio
async def test_failed_local_checkpoint_never_masks_original_safe_failure():
    survey, context = materials(1)
    transport = FailureTransport("exception")
    owner, budget, _ = await owner_for(transport)
    attempts = []

    def checkpoint(progress):
        if LOCAL_KEY in progress:
            attempts.append(copy.deepcopy(progress))
            raise OSError(MESSAGE)

    context["evaluation_checkpoint"] = checkpoint
    try:
        result = await RagasFaithfulnessEvaluator(
            load_config(), llm=ModelBoundClient(owner, "lite", "evaluation")
        ).evaluate(survey, context)
        assert len(attempts) == 1
        assert result.skipped
        assert result.details["skipped_reason"] == "evaluation_contract_invalid"
        assert LOCAL_KEY not in json.dumps(result.details)
        assert_private_absent(result.details)
        assert len(transport.sent) == budget.snapshot()["requests_sent"] == 2
        assert budget.snapshot()["usage_uncertain"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_malformed_json_keeps_existing_reason_without_replaying_completed_transport():
    survey, context = materials(1)
    transport = FailureTransport("json")
    owner, budget, _ = await owner_for(transport)
    snapshots = []
    context["evaluation_checkpoint"] = lambda progress: snapshots.append(
        copy.deepcopy(progress)
    )
    try:
        result = await RagasFaithfulnessEvaluator(
            load_config(), llm=ModelBoundClient(owner, "lite", "evaluation")
        ).evaluate(survey, context)
        assert result.skipped
        assert result.details["skipped_reason"] == "evaluation_contract_invalid"
        assert len(transport.sent) == budget.snapshot()["requests_sent"] == 2
        assert not budget.snapshot()["usage_uncertain"]
        assert result.details["stages"]["batches"][-1]["status"] == "invalid_response"
        assert LOCAL_KEY in snapshots[-1]
        assert_private_absent(snapshots[-1][LOCAL_KEY])
        assert LOCAL_KEY not in json.dumps(result.details)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_cancel_propagates_without_failure_diagnostic_or_unknown_request_replay():
    survey, context = materials(1)
    transport = FailureTransport("cancel")
    owner, budget, _ = await owner_for(transport)
    snapshots = []
    context["evaluation_checkpoint"] = lambda progress: snapshots.append(
        copy.deepcopy(progress)
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await RagasFaithfulnessEvaluator(
                load_config(), llm=ModelBoundClient(owner, "lite", "evaluation")
            ).evaluate(survey, context)
        assert len(transport.sent) == budget.snapshot()["requests_sent"] == 2
        assert budget.snapshot()["usage_uncertain"]
        assert budget.stop_reason == "strict_usage_unknown"
        assert all(LOCAL_KEY not in progress for progress in snapshots)
        assert snapshots[-1]["batches"][-1]["status"] == "interrupted_no_replay"
    finally:
        await owner.close()
