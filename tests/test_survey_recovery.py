"""Exercise frozen Judge inputs and independent recovery without network access."""

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from test_phase14_5 import _artifact, _case

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.survey_judge import SurveyBenchmarkJudge
from litagent.benchmark.survey_models import SurveyJudgeConfig
from litagent.benchmark.survey_recovery import SurveyRecoveryStore, verified_product
from litagent.llm.client import LLMResponse
from litagent.report_render import build_report_source_units


def response(payload):
    """Return a provider response with known nonzero usage."""
    return LLMResponse(
        content=json.dumps(payload),
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def inventory(
    claim="Claim.",
    source_unit_id="unit:0:0",
    evidence_aliases=("e1",),
):
    """Build one source-unit claim extraction."""
    return response(
        {
            "claims": [
                {
                    "claim_text": claim,
                    "source_unit_id": source_unit_id,
                    "evidence_aliases": list(evidence_aliases),
                }
            ]
        }
    )


def audit():
    """Build a support decision using a deterministic evidence alias."""
    return response(
        {
            "topic_assessments": [{"topic_id": "metric-space", "covered": True}],
            "claim_assessments": [
                {
                    "claim_id": "claim_0",
                    "source_faithful": True,
                    "supported": True,
                    "evidence_ids": ["e1"],
                }
            ],
            "contradictions": [],
        }
    )


@pytest.mark.asyncio
async def test_audit_retry_keeps_inventory_and_restores_evidence_ids(monkeypatch):
    monkeypatch.setattr("litagent.benchmark.survey_judge.asyncio.sleep", AsyncMock())
    llm = AsyncMock()
    llm.chat.side_effect = [inventory(), response({"claim_11": "supported"}), audit()]
    judge = SurveyBenchmarkJudge(
        llm,
        SurveyJudgeConfig(
            base_url="https://judge.example", model="judge", max_retries=1
        ),
    )
    metrics, usage = await judge.evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {"text": "Claim."}},
    )
    assert metrics.status == "completed"
    assert llm.chat.await_count == 3
    assert metrics.claim_assessments[0].evidence_ids == ["long-evidence-id"]
    assert usage.total_tokens == 45
    assert llm.chat.await_args_list[1].args == llm.chat.await_args_list[2].args


@pytest.mark.asyncio
async def test_inventory_cannot_introduce_a_citation():
    llm = AsyncMock()
    llm.chat.return_value = inventory(claim="Claim [E:invented].")
    metrics, _ = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {"text": "Claim."}},
    )
    assert metrics.reason_codes == ["judge_inventory_schema_invalid"]
    assert llm.chat.await_count == 1


def test_judge_retries_are_opt_in():
    assert (
        SurveyJudgeConfig(base_url="https://judge.example", model="judge").max_retries
        == 0
    )


def test_real_failure_shapes_remain_rejected():
    from litagent.benchmark.survey_judge import (
        _anchor_inventory,
        _parse_audit,
        _parse_inventory,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "survey_judge_failures.json").read_text()
    )
    with pytest.raises(ValueError, match="judge_inventory_schema_invalid"):
        _parse_inventory(
            json.dumps(
                {
                    "claims": [
                        {
                            "claim_text": fixture["invented_claim"],
                            "source_unit_id": "unit:0:0",
                            "evidence_aliases": ["e1"],
                        }
                    ]
                }
            )
        )
    with pytest.raises(ValueError, match="judge_audit_schema_invalid"):
        _parse_audit(
            json.dumps(
                {
                    "topic_assessments": [],
                    "contradictions": [],
                    "claim_assessments": [fixture["malformed_assessment"]],
                }
            )
        )


def test_strict_schema_requires_all_nested_fields():
    from litagent.benchmark.survey_judge import _JudgeResponse, _strict_schema

    schema = _strict_schema(_JudgeResponse)
    for value in [schema, *schema["$defs"].values()]:
        if value.get("type") == "object":
            assert set(value["required"]) == set(value["properties"])
            assert value["additionalProperties"] is False


def test_alias_substitution_never_rewrites_a_generated_alias():
    from litagent.benchmark.survey_judge import _alias_citations

    assert (
        _alias_citations("First [E:e2], second [E:e1].", {"e2": "e1", "e1": "e2"})
        == "First [E:e1], second [E:e2]."
    )


@pytest.mark.asyncio
async def test_judge_trace_labels_stages_without_extra_calls():
    from litagent.observability.context import get_task_id

    events = []
    task_ids = []
    replies = iter([inventory(), audit()])

    async def chat(*args, **kwargs):
        task_ids.append(get_task_id())
        return next(replies)

    llm = AsyncMock()
    llm.chat.side_effect = chat
    _, usage = await SurveyBenchmarkJudge(
        llm,
        SurveyJudgeConfig(base_url="https://judge.example", model="judge"),
        trace_hook=lambda event, data: events.append((event, data)),
    ).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {}},
    )
    starts = [data for event, data in events if event == "subspan.start"]
    assert [item["name"] for item in starts] == ["judge.inventory", "judge.audit"]
    assert task_ids == [item["task_id"] for item in starts]
    assert usage.call_count == 2
    assert usage.total_tokens == 30
    assert get_task_id() == ""
    assert all(item["input"]["messages"] for item in starts)


def test_langfuse_stage_input_and_failure_output_stay_redacted():
    from test_observability import _tracer_with_mock

    from litagent.observability.recorder import RedactingTraceHook

    tracer, events = _tracer_with_mock()
    hook = RedactingTraceHook(tracer._handle, payload_mode="full_redacted")
    hook(
        "subspan.start",
        {
            "task_id": "judge:audit",
            "name": "judge.audit",
            "input": {"claim_id": "claim_0", "api_key": "test-secret"},
        },
    )
    hook(
        "subspan.end",
        {
            "task_id": "judge:audit",
            "error": "schema_invalid",
            "output": {"reason": "missing_claim_id", "password": "another-secret"},
        },
    )
    start = next(data for event, data in events if event == "start")
    assert start["input"]["claim_id"] == "claim_0"
    assert "test-secret" not in repr(events)
    assert "another-secret" not in repr(events)
    assert any(
        data.get("output", {}).get("reason") == "missing_claim_id"
        for event, data in events
        if event == "update"
    )


@pytest.mark.asyncio
async def test_provider_missing_usage_flag_is_not_known_zero_cost():
    llm = AsyncMock()
    reply = inventory()
    reply.usage = {"prompt_tokens": 0, "completion_tokens": 0, "usage_available": False}
    llm.chat.side_effect = [reply, audit()]
    _, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {}},
    )
    assert usage.usage_complete is False
    assert usage.missing_usage_calls == 1
    assert usage.total_tokens == 15


@pytest.mark.asyncio
async def test_non_cited_ledger_entry_cannot_be_used_by_audit():
    llm = AsyncMock()
    bad = json.loads(audit().content)
    bad["claim_assessments"][0]["evidence_ids"] = ["e2"]
    llm.chat.side_effect = [inventory(), response(bad)]
    metrics, _ = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {}, "not-cited": {}},
    )
    assert metrics.reason_codes == ["judge_unknown_evidence_ids"]


def test_duplicate_source_text_is_disambiguated_by_host_unit_id():
    from litagent.benchmark.survey_judge import _anchor_inventory, _parse_inventory

    quote = "Claim [E:long-evidence-id]."
    survey = quote + "\n\n" + quote
    units = build_report_source_units(survey)
    result = _anchor_inventory(
        _parse_inventory(inventory(source_unit_id=units[1].unit_id).content),
        survey,
        source_units=units,
        alias_to_id={"e1": "long-evidence-id"},
        ledger_ids={"long-evidence-id"},
    )
    assert result.claims[0].source_start == units[1].start
    assert result.claims[0].source_quote == units[1].text


@pytest.mark.asyncio
async def test_restart_reuses_frozen_inventory_and_counts_failed_attempt(tmp_path):
    config = SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    repository = BenchmarkArtifactRepository(tmp_path)
    key = "a" * 64

    def save(state):
        repository.write_checkpoint("judge", key, state)

    first = AsyncMock()
    first.chat.side_effect = [inventory(), response({"claim_11": "supported"})]
    kwargs = dict(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {"text": "Claim."}},
    )
    failed, _ = await SurveyBenchmarkJudge(first, config).evaluate(
        **kwargs, checkpoint={}, save_checkpoint=save
    )
    assert failed.status == "failed"
    second = AsyncMock()
    second.chat.return_value = audit()
    state = repository.read_checkpoint("judge", key, strict=True)
    result, usage = await SurveyBenchmarkJudge(second, config).evaluate(
        **kwargs, checkpoint=state, save_checkpoint=save
    )
    assert result.status == "completed"
    assert second.chat.await_count == 1
    assert usage.total_tokens == 45
    assert usage.call_count == 3
    assert len(result.failure_details) == 1
    assert len(state["calls"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.CancelledError(), asyncio.TimeoutError()])
async def test_interrupted_call_persists_unknown_cost(failure):
    llm = AsyncMock()
    llm.chat.side_effect = failure
    state = {}
    coroutine = SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(case=_case(), survey="Claim.", ledger={}, checkpoint=state)
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await coroutine
        assert state["calls"][0]["status"] == "cancelled"
    else:
        metrics, usage = await coroutine
        assert metrics.status == "failed"
        assert usage.usage_complete is False
        assert usage.missing_usage_calls == 1
    assert state["calls"][0]["usage_complete"] is False


@pytest.mark.asyncio
async def test_checkpoint_binding_rejects_modified_report_before_call():
    state = {}
    config = SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    first = AsyncMock()
    first.chat.side_effect = [inventory(), audit()]
    await SurveyBenchmarkJudge(first, config).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {}},
        checkpoint=state,
    )
    second = AsyncMock()
    with pytest.raises(ValueError, match="input mismatch"):
        await SurveyBenchmarkJudge(second, config).evaluate(
            case=_case(),
            survey="Changed [E:long-evidence-id].",
            ledger={"long-evidence-id": {}},
            checkpoint=state,
        )
    second.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_budget_stops_between_inventory_and_audit():
    llm = AsyncMock()
    llm.chat.return_value = inventory()
    gate = Mock(side_effect=[True, False])
    metrics, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {}},
        before_call=gate,
    )
    assert metrics.reason_codes == ["judge_budget_exhausted"]
    assert llm.chat.await_count == 1
    assert usage.total_tokens == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("source_unit_id", ["unit:999:0", "unknown"])
async def test_source_unit_must_exist_in_report(source_unit_id):
    llm = AsyncMock()
    llm.chat.return_value = inventory(source_unit_id=source_unit_id)
    metrics, _ = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(
        case=_case(),
        survey="Claim [E:long-evidence-id].",
        ledger={"long-evidence-id": {}},
    )
    assert metrics.reason_codes == ["judge_inventory_source_unit_unknown"]
    assert llm.chat.await_count == 1


def make_product(tmp_path):
    """Save a valid immutable product fixture with known usage."""
    from litagent.benchmark.survey_metrics import evaluate_survey_artifact
    from litagent.benchmark.survey_models import SurveyCaseObservation

    artifact = _artifact("Claim [E:e1].")
    artifact["query"] = _case().query
    artifact["events"][0]["data"] = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 20}
    }
    path = tmp_path / "product.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    metrics = evaluate_survey_artifact(case=_case(), artifact=artifact)
    metrics.cost.usage_complete = True
    product = SurveyCaseObservation(
        run_id="run-1",
        case_id=_case().case_id,
        domain=_case().domain,
        profile_id="external-only",
        repetition=1,
        status="completed",
        metrics=metrics,
        artifact_path=str(path),
        artifact_sha256=f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
    )
    return product, artifact


def test_product_checkpoint_refuses_changed_identity_and_damaged_bytes(tmp_path):
    product, _ = make_product(tmp_path)
    store = SurveyRecoveryStore(BenchmarkArtifactRepository(tmp_path / "checkpoints"))
    store.save_product("a" * 64, "identity", product)
    assert store.product("a" * 64, "identity") == product
    with pytest.raises(ValueError, match="incompatible"):
        store.product("a" * 64, "changed")
    Path(product.artifact_path).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sha_mismatch"):
        store.product("a" * 64, "identity")


def test_inflight_and_corrupt_checkpoints_are_not_cache_misses(tmp_path):
    repository = BenchmarkArtifactRepository(tmp_path)
    store = SurveyRecoveryStore(repository)
    store.save_product("a" * 64, "identity", None)
    with pytest.raises(ValueError, match="interrupted"):
        store.product("a" * 64, "identity")
    path = repository.write_checkpoint("survey-products", "b" * 64, {})
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        store.accounting()


def test_valid_json_with_invalid_paid_state_is_not_a_cache_miss(tmp_path):
    repository = BenchmarkArtifactRepository(tmp_path)
    store = SurveyRecoveryStore(repository)
    repository.write_checkpoint("survey-products", "a" * 64, {})
    with pytest.raises(ValueError, match="schema_invalid"):
        store.product("a" * 64, "identity")
    repository.write_checkpoint("survey-judges", "b" * 64, {})
    with pytest.raises(ValueError, match="calls_invalid"):
        store.judge("b" * 64)


@pytest.mark.asyncio
async def test_preflight_failure_is_a_valid_zero_call_checkpoint(tmp_path):
    repository = BenchmarkArtifactRepository(tmp_path)
    store = SurveyRecoveryStore(repository)
    state = {}
    llm = AsyncMock()
    metrics, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://judge.example", model="judge")
    ).evaluate(case=_case(), survey="Claim [E:a, b].", ledger={}, checkpoint=state)
    assert metrics.status == "failed"
    assert usage.usage_complete is True
    assert usage.total_tokens == 0
    store.save_judge("a" * 64, state)
    assert store.judge("a" * 64)["calls"] == []
    assert store.accounting()["usage_complete"] is True
    llm.chat.assert_not_awaited()


@pytest.mark.parametrize("tokens", [-1, "unknown", True])
def test_corrupt_accounting_cannot_reduce_spending(tmp_path, tokens):
    repository = BenchmarkArtifactRepository(tmp_path)
    store = SurveyRecoveryStore(repository)
    store.save_judge(
        "a" * 64,
        {
            "calls": [
                {
                    "usage_complete": True,
                    "prompt_tokens": tokens,
                    "completion_tokens": 0,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="tokens_invalid"):
        store.accounting()


def test_root_accounting_survives_restarts_without_counting_product_twice(tmp_path):
    product, _ = make_product(tmp_path)
    repository = BenchmarkArtifactRepository(tmp_path / "benchmark")
    store = SurveyRecoveryStore(repository)
    store.save_product("a" * 64, "one", product)
    store.save_product("b" * 64, "two", product)
    store.save_judge(
        "c" * 64,
        {
            "calls": [
                {"usage_complete": True, "prompt_tokens": 10, "completion_tokens": 5},
                {"usage_complete": False, "status": "failed"},
            ]
        },
    )
    counts = SurveyRecoveryStore(repository).accounting()
    assert counts["product_tokens"] == product.metrics.cost.total_tokens
    assert counts["judge_tokens"] == 15
    assert counts["usage_complete"] is False
    assert counts["products"] == 1


@pytest.mark.asyncio
async def test_judge_only_dry_run_reads_artifacts_without_services_or_writes(
    tmp_path, monkeypatch
):
    from litagent.benchmark.datasets import load_survey_dataset, load_survey_profiles
    from litagent.benchmark.survey_runner import SurveyBenchmarkRunner
    from litagent.cli import _cmd_benchmark, build_parser
    from litagent.config import load_config
    from litagent.observability.recorder import ArchiveRepository

    dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
    profiles = load_survey_profiles(Path("benchmarks/survey/profiles.yaml"))
    case = dataset.cases[0]
    product, artifact = make_product(tmp_path)
    artifact["query"] = case.query
    path = Path(product.artifact_path)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    product = product.model_copy(
        update={
            "case_id": case.case_id,
            "domain": case.domain,
            "artifact_sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
        }
    )
    output = tmp_path / "output"
    runner = SurveyBenchmarkRunner(
        base_config=load_config(),
        judge_config=SurveyJudgeConfig(base_url="https://judge.example", model="judge"),
        artifacts=BenchmarkArtifactRepository(output),
        run_archive=ArchiveRepository(tmp_path / "runs"),
    )
    original = runner._build_result(
        stage="pilot",
        dataset=dataset,
        profiles=profiles,
        observations=[product],
        case_aggregates=[],
        summaries=[],
        recommended_profile_id=None,
        git_sha="historical",
        git_dirty=True,
        reason_codes=[],
    )
    source = tmp_path / "source.json"
    source.write_text(original.model_dump_json(), encoding="utf-8")
    before = source.read_bytes(), path.read_bytes()
    forbidden = Mock(side_effect=AssertionError("paid service must not be constructed"))
    monkeypatch.setattr("litagent.benchmark.survey_runner.LitAgent", forbidden)
    monkeypatch.setattr(
        "litagent.benchmark.survey_runner.OpenAICompatibleClient", forbidden
    )
    result = await runner.rejudge(source=source, dataset=dataset, dry_run=True)
    assert result["status"] == "planned"
    assert result["actions"][0]["product"] == "reuse"
    assert result["actions"][0]["judge"] == "inventory_and_audit"
    assert not output.exists()
    args = build_parser().parse_args(
        [
            "benchmark",
            "survey",
            "--stage",
            "pilot",
            "--dataset",
            "benchmarks/survey/dataset.yaml",
            "--profiles",
            "benchmarks/survey/profiles.yaml",
            "--judge-config",
            "benchmarks/survey/judge.example.yaml",
            "--judge-only",
            "--reuse-surveys",
            str(source),
            "--dry-run",
            "--output-root",
            str(output),
        ]
    )
    await _cmd_benchmark(args)
    assert before == (source.read_bytes(), path.read_bytes())
    assert not output.exists()


def test_budget_counts_history_and_does_not_block_judge_on_product_count(tmp_path):
    from litagent.benchmark.survey_models import SurveyBenchmarkBudget
    from litagent.benchmark.survey_runner import SurveyBenchmarkRunner
    from litagent.config import load_config
    from litagent.observability.recorder import ArchiveRepository

    product, _ = make_product(tmp_path)
    repo = BenchmarkArtifactRepository(tmp_path / "bench")
    store = SurveyRecoveryStore(repo)
    store.save_product("a" * 64, "identity", product)
    runner = SurveyBenchmarkRunner(
        base_config=load_config(),
        judge_config=SurveyJudgeConfig(base_url="https://judge.example", model="judge"),
        artifacts=repo,
        run_archive=ArchiveRepository(tmp_path / "runs"),
        budget=SurveyBenchmarkBudget(
            max_observations=1, max_product_tokens=10000, max_judge_tokens=30
        ),
    )
    assert not runner._budget_allows_new_observation()
    assert runner._budget_allows_new_observation(judge_only=True)
    store.save_judge(
        "b" * 64,
        {
            "calls": [
                {"usage_complete": True, "prompt_tokens": 25, "completion_tokens": 5}
            ]
        },
    )
    runner._reset_budget_state()
    assert not runner._budget_allows_new_observation(judge_only=True)


@pytest.mark.asyncio
async def test_partial_product_does_not_spend_on_judge(tmp_path, monkeypatch):
    from litagent.benchmark.survey_runner import SurveyBenchmarkRunner
    from litagent.config import load_config
    from litagent.observability.recorder import ArchiveRepository

    product, artifact = make_product(tmp_path)
    product.metrics.partial = True
    forbidden = Mock(
        side_effect=AssertionError("partial product must not trigger Judge")
    )
    monkeypatch.setattr(
        "litagent.benchmark.survey_runner.OpenAICompatibleClient", forbidden
    )
    runner = SurveyBenchmarkRunner(
        base_config=load_config(),
        judge_config=SurveyJudgeConfig(base_url="https://judge.example", model="judge"),
        artifacts=BenchmarkArtifactRepository(tmp_path / "bench"),
        run_archive=ArchiveRepository(tmp_path / "runs"),
    )
    result = await runner._judge_product(_case(), load_config(), product, artifact)
    assert result.judge.status == "skipped"
    assert result.judge.reason_codes == ["product_incomplete"]
    assert result.judge_usage.total_tokens == 0


@pytest.mark.asyncio
async def test_authentication_failure_does_not_trigger_contract_retry():
    import httpx
    from openai import AuthenticationError

    llm = AsyncMock()
    llm.chat.side_effect = AuthenticationError(
        "invalid key",
        body={},
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://judge.example")
        ),
    )
    metrics, usage = await SurveyBenchmarkJudge(
        llm,
        SurveyJudgeConfig(
            base_url="https://judge.example", model="judge", max_retries=1
        ),
    ).evaluate(case=_case(), survey="Claim.", ledger={})
    assert metrics.status == "failed"
    assert llm.chat.await_count == 1
    assert usage.usage_complete is False


@pytest.mark.asyncio
async def test_product_is_saved_before_judge_and_restart_never_generates_again(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from litagent.benchmark.datasets import load_survey_dataset, load_survey_profiles
    from litagent.benchmark.survey_runner import SurveyBenchmarkRunner
    from litagent.config import load_config
    from litagent.observability.recorder import ArchiveRepository

    dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
    case = dataset.cases[0]
    profile = load_survey_profiles(Path("benchmarks/survey/profiles.yaml"))[0]
    repo = BenchmarkArtifactRepository(tmp_path / "bench")
    archive = ArchiveRepository(tmp_path / "runs")
    generations = []
    artifact = _artifact("Claim [E:e1].")
    artifact["query"] = case.query

    class Agent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, query):
            generations.append(query)
            return SimpleNamespace()

    monkeypatch.setattr(
        "litagent.benchmark.survey_runner.LitAgent", lambda *a, **kw: Agent()
    )

    def finalize(self, recorder, tracer, **kwargs):
        artifact["run_id"] = recorder.run_id
        archive.save(artifact)
        return artifact

    monkeypatch.setattr(SurveyBenchmarkRunner, "_finalize_survey_artifact", finalize)
    monkeypatch.setattr(
        "litagent.benchmark.survey_runner.LangFuseTracer",
        lambda **kw: Mock(trace_id="trace", trace_url="url"),
    )
    clients = []
    replies = [
        response(
            {
                "claims": [
                    {
                        "claim_text": "Claim.",
                        "source_unit_id": "unit:0:0",
                        "evidence_aliases": ["e1"],
                    }
                ]
            }
        ),
        response({"claim_11": "supported"}),
    ]
    good = json.loads(audit().content)
    good["topic_assessments"] = [
        {"topic_id": topic.topic_id, "covered": True} for topic in case.expected_topics
    ]

    def client(**kwargs):
        instance = AsyncMock()
        instance.chat.side_effect = replies if not clients else [response(good)]
        clients.append(instance)
        return instance

    monkeypatch.setattr(
        "litagent.benchmark.survey_runner.OpenAICompatibleClient", client
    )

    def runner():
        return SurveyBenchmarkRunner(
            base_config=load_config(),
            judge_config=SurveyJudgeConfig(
                base_url="https://judge.example", model="judge"
            ),
            artifacts=repo,
            run_archive=archive,
        )

    kwargs = dict(
        dataset=dataset,
        cases=[case],
        profiles=[profile],
        repetition=1,
        git_sha="old",
        git_dirty=True,
    )
    first_runner = runner()
    first_runner._judged_paper_ids = set(dataset.corpus_paper_ids)
    slot, identity = first_runner._product_identity(dataset, case, profile, 1)
    first = [
        await first_runner._run_live_case(
            case,
            profile,
            first_runner._config_for(profile, dataset.dataset_version),
            1,
            product_slot=slot,
            product_identity=identity,
        )
    ]
    assert first[0].judge is not None, first[0].reason_codes
    assert first[0].judge.status == "failed"
    second = await runner()._run_matrix(**{**kwargs, "git_sha": "new"})
    assert second[0].judge.status == "completed"
    assert len(generations) == 1
    assert clients[1].chat.await_count == 1
    assert first[0].artifact_sha256 == second[0].artifact_sha256
    assert second[0].judge_usage.total_tokens == 45
