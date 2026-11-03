"""Exercise stage routing at the SDK boundary and production wiring consumers."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from model_config_helpers import load_legacy_mini_config
from test_bound_body_repair import RepairTransport, real_case

from litagent.config import LLMConfig, load_config
from litagent.contracts import build_config_summary
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.runner import Infra, LitAgent
from litagent.safety.budget import CostBudget

LITE = "doubao-seed-2-1-lite-260915"
PRO = "doubao-seed-2-1-pro-260915"


def test_default_profile_routes_both_lite_roles_and_pro_writing():
    cfg = load_config()
    assert [
        cfg.llm.model_for(s) for s in ("extraction", "generation", "evaluation")
    ] == [LITE, PRO, LITE]
    assert (
        cfg.extractor.thinking_mode
        == cfg.eval.thinking_mode
        == cfg.adversarial.thinking_mode
        == "disabled"
    )
    assert cfg.safety.max_cost_tokens == 500000
    assert not cfg.llm.uses_single_model(cfg.llm.model)


@pytest.mark.parametrize(
    "policy", ["legacy", "strict_v1", "strict_auto_v1", "strict_auto_v2"]
)
def test_omitted_stage_fields_preserve_the_four_historical_config_identities(policy):
    before = json.loads(
        (
            Path(__file__).parent / "fixtures/legacy-model-config-fingerprints.json"
        ).read_text()
    )
    summary = build_config_summary(load_legacy_mini_config(), policy)
    assert summary["fingerprint"] == before[policy]
    assert "model_routing" not in summary["effective"]["llm"]


@pytest.mark.parametrize("stage", ["extraction", "generation", "evaluation"])
def test_every_stage_model_changes_resume_identity(stage):
    cfg = load_config()
    before = build_config_summary(cfg, "strict_auto_v2")
    setattr(cfg.llm, f"{stage}_model", "different-fixed-model")
    after = build_config_summary(cfg, "strict_auto_v2")
    assert before["fingerprint"] != after["fingerprint"]
    assert after["effective"]["llm"]["model_routing"][stage] == "different-fixed-model"


@pytest.mark.parametrize("value", ["", " ", " padded "])
def test_invalid_model_overrides_are_rejected(value):
    with pytest.raises(ValueError):
        LLMConfig(generation_model=value)
    with pytest.raises(ValueError, match="unknown"):
        LLMConfig().model_for("typo")


def sdk_response(model, *, usage=True):
    """Return provider metadata after real request preparation and reservation."""
    return SimpleNamespace(
        model=model,
        _request_id="offline",
        usage=(
            SimpleNamespace(prompt_tokens=10, completion_tokens=10) if usage else None
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="{}", tool_calls=None),
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "model", "unknown", "cancel"])
async def test_stage_views_share_budget_and_stop_without_retries(monkeypatch, fault):
    cfg = load_config()
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    wire, events = [], []
    base = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        LITE,
        api_key="offline",
        cost_budget=budget,
        trace_hook=lambda e, d: events.append((e, d)),
    )

    async def create(**kwargs):
        wire.append(copy.deepcopy(kwargs))
        if fault == "cancel":
            raise asyncio.CancelledError()
        return sdk_response(
            "wrong" if fault == "model" else kwargs["model"], usage=fault != "unknown"
        )

    monkeypatch.setattr(
        base._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    views = {
        s: ModelBoundClient(base, cfg.llm.model_for(s), s)
        for s in ("extraction", "generation", "evaluation")
    }
    try:
        for stage, task in [
            ("extraction", "extract"),
            ("generation", "synthesis"),
            ("evaluation", "evaluation.initial"),
        ]:
            token = set_task_id(task)
            try:
                if fault:
                    with pytest.raises((ValueError, asyncio.CancelledError)):
                        await views[stage].chat(
                            [{"role": "user", "content": "offline"}], max_tokens=100
                        )
                    break
                await views[stage].chat(
                    [{"role": "user", "content": "offline"}], max_tokens=100
                )
            finally:
                reset_task_id(token)
        if fault is None:
            assert [w["model"] for w in wire] == [LITE, PRO, LITE]
            assert budget.used == 60
            assert budget.snapshot()["phase_usage"] == [20, 20, 20]
        else:
            assert len(wire) == 1
            assert budget.stop_reason
            before = len(wire)
            with pytest.raises(ValueError):
                await views["generation"].chat([{"role": "user", "content": "stop"}])
            assert len(wire) == before
            if fault == "model":
                assert budget.used == 20 and not budget.snapshot()["usage_uncertain"]
        assert all(v._cost_budget is budget for v in views.values())
        with pytest.raises(ValueError, match="override_conflict"):
            await views["generation"].chat([], model=LITE)
        assert all(
            w.get("extra_body", {}).get("thinking", {}).get("type") != "enabled"
            for w in wire
        )
    finally:
        await base.close()


@pytest.mark.asyncio
async def test_real_wiring_routes_all_workers_and_owns_only_one_transport(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    cfg.observability.enabled = False
    cfg.mcp_servers = {}
    agent = LitAgent(cfg)
    monkeypatch.setattr(agent, "_connect_infra", AsyncMock(return_value=Infra()))
    await agent._wire()
    owner = agent._owned_llm
    closed = AsyncMock()
    monkeypatch.setattr(owner, "close", closed)
    assert agent._extraction_strategy._llm._llm._model == LITE
    assert agent._extractor._strict_paper_batch_size == 2
    assert agent._extractor._strict_field_spans is False
    assert agent._extractor._strict_grounded_claims is True
    assert agent._synthesis._llm._model == PRO
    assert agent._reviewer._llm._model == LITE
    assert agent._adversarial._llm._model == PRO
    assert all(e._llm._model == LITE for e in agent._evaluators)
    assert all(v._cost_budget is agent._cost_budget for v in agent._stage_llms.values())
    await agent._stage_llm("generation").close()
    closed.assert_not_awaited()
    await agent.cleanup()
    await agent.cleanup()
    closed.assert_awaited_once()
    # The real underlying client was borrowed by the close spy above.
    await owner._client.close()


@pytest.mark.asyncio
async def test_partial_wire_failure_closes_its_published_owner(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    cfg.observability.enabled = False
    agent = LitAgent(cfg)
    owned = []
    original = agent._wire_llms

    def publish():
        original()
        owned.append(agent._owned_llm)

    monkeypatch.setattr(agent, "_wire_llms", publish)
    monkeypatch.setattr(
        agent, "_connect_infra", AsyncMock(side_effect=RuntimeError("offline failure"))
    )
    with pytest.raises(RuntimeError):
        await agent._wire()
    assert agent._owned_llm is None and owned[0]._client.is_closed()


@pytest.mark.asyncio
async def test_real_targeted_repair_uses_pro_and_all_three_reassessment_calls_use_lite(
    tmp_path, monkeypatch
):
    cfg = load_config()
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    cfg.observability.enabled = False
    cfg.mcp_servers = {}
    agent = LitAgent(cfg)
    agent._session_id = "mixed-model-repair"
    monkeypatch.setattr(agent, "_connect_infra", AsyncMock(return_value=Infra()))
    await agent._wire()
    agent._cost_budget.configure_strict(halt_on_unknown_usage=True)
    wire = []
    transport = RepairTransport()

    async def create(**kwargs):
        wire.append(copy.deepcopy(kwargs))
        response = await transport.chat(
            kwargs["messages"], **{k: v for k, v in kwargs.items() if k != "messages"}
        )
        result = sdk_response(kwargs["model"])
        result.choices[0].message.content = response.content
        return result

    monkeypatch.setattr(
        agent._llm._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    source = real_case()
    report = source["report"]
    from litagent.quality import targeted_repair_state

    report["metadata"]["targeted_repair_state"] = targeted_repair_state(
        None, agent._session_id
    )
    agent._targeted_repair_state_path = tmp_path / "repairs.json"
    monkeypatch.chdir(tmp_path)
    try:
        result = await agent._attempt_evidence_rewrite(
            query=source["query"],
            report_data=report,
            results=source["results"],
            initial_evaluation=report["evaluation"],
            initial_quality=report["quality"],
            report_language="zh-CN",
        )
        assert result.committed
        assert [r["model"] for r in wire] == [PRO, LITE, LITE, LITE]
        assert result.candidate_evaluation["faithfulness"]["details"]["coverage"][
            "complete"
        ]
        assert agent._cost_budget.used == 80
    finally:
        await agent.cleanup()


def test_historical_paid_entry_rejects_a_hidden_stage_override(tmp_path):
    from scripts.run_body_quality_comparison import prepare

    cfg = load_legacy_mini_config()
    cfg.llm.generation_model = PRO
    with pytest.raises(ValueError, match="body_comparison_model_invalid"):
        prepare(tmp_path / "new", cfg, tmp_path / "unused")


def test_audit_uses_bound_generation_model_instead_of_today_or_base_model():
    from litagent import api
    from litagent.report_audit import ReportAuditService
    from litagent.workbench import ReportVersion

    cfg = load_config()
    report = {
        "survey": "Public claim.",
        "metadata": {"config_summary": build_config_summary(cfg)},
    }
    version = ReportVersion(
        run_id="fixed",
        revision=1,
        report_language="en",
        survey=report["survey"],
        human_status="pending",
    )
    snapshot = api._audit_snapshot_from_source(version, report, {}, {})
    assert snapshot["generation_provider"]["model"] == PRO
    service = ReportAuditService(cfg)
    assert service.judge_config().model == PRO
    assert service.model_comparison(snapshot)["self_judged"]
    cfg.report_audit.judge_model = LITE
    assert not service.model_comparison(snapshot)["self_judged"]


def test_identical_prose_from_another_revision_cannot_supply_generation_identity():
    from litagent import api
    from litagent.workbench import ReportVersion

    cfg = load_config()
    version = ReportVersion(
        run_id="same-run",
        revision=1,
        report_language="en",
        survey="Same prose.",
        human_status="pending",
    )
    report = {"survey": version.survey, "metadata": {}}
    artifact = {
        "run_id": "same-run",
        "config": build_config_summary(cfg),
        "report": {"survey": version.survey, "metadata": {"report_revision": 2}},
    }
    snapshot = api._audit_snapshot_from_source(version, report, {}, artifact)
    assert snapshot["generation_provider"]["model"] is None
    artifact["report"]["metadata"]["report_revision"] = 1
    snapshot = api._audit_snapshot_from_source(version, report, {}, artifact)
    assert snapshot["generation_provider"]["model"] == PRO


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "provider_default", "disabled", "enabled"])
async def test_planner_thinking_mode_reaches_real_sdk_body(monkeypatch, mode):
    import httpx
    from openai import AsyncOpenAI

    from litagent.agents.planner import SurveyPlanner
    from litagent.config import PlannerConfig

    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "planner-probe",
                "created": 1,
                "object": "chat.completion",
                "model": LITE,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"sub_queries": ["metric learning few shot"]}
                            ),
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
            **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        ),
    )
    budget = CostBudget(max_tokens=50000)
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", LITE, api_key="offline", cost_budget=budget
    )
    planner = SurveyPlanner(client, PlannerConfig(thinking_mode=mode))
    try:
        assert await planner._decompose(
            "Compare metric learning for few-shot classification."
        ) == ["metric learning few shot"]
        assert len(requests) == 1
        assert requests[0]["model"] == LITE and requests[0]["max_tokens"] == 8192
        if mode in (None, "provider_default"):
            assert "thinking" not in requests[0]
        else:
            assert requests[0]["thinking"] == {"type": mode}
        assert budget.snapshot()["used_tokens"] == 20
    finally:
        await client.close()


def test_planner_mode_changes_new_identity_and_omission_keeps_legacy():
    config = load_config()
    assert config.planner.thinking_mode == "disabled"
    before = build_config_summary(config, "strict_auto_v2")
    config.planner.thinking_mode = None
    after = build_config_summary(config, "strict_auto_v2")
    assert before["fingerprint"] != after["fingerprint"]
    assert "thinking_mode" not in after["effective"]["planner"]
