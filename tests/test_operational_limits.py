"""Check complete resource expansion without weakening historical scientific gates."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine, native_writer_budget_plan
from litagent.operational_limits import increase, uplift_configuration
from litagent.output_capacity import configured_output_tokens, envelope_context_limit
from litagent.run_limits import (
    CONTRACT,
    UPLIFT_CONTRACT,
    configured_run_limits,
    validate_run_limits,
)
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import (
    EnvelopeExceeded,
    PlannedBudget,
    check_writer_envelope,
    nested_json_size,
    request_scope,
)


def test_every_declared_resource_ceiling_is_raised_without_growing_writing_targets():
    """Audit every declared field against the baseline instead of spot-checking bytes."""
    baseline = yaml.safe_load(Path("config/expanded-capacity.yaml").read_text())
    expected, inventory = uplift_configuration(baseline, CapacityConfig().model_dump())
    actual = yaml.safe_load(Path("config/operational-plus20.yaml").read_text())
    assert actual == expected
    assert len(inventory) > 60
    for row in inventory.values():
        assert row["new"] == increase(row["old"])
    assert actual["capacity"]["target_body_units"] == 48
    assert actual["capacity"]["max_body_units"] == 80
    assert actual["capacity"]["body_json_bytes"] == 38400
    assert (
        actual["adversarial"]["pass_threshold"]
        == baseline["adversarial"]["pass_threshold"]
    )
    assert actual["eval"]["strict_context_margin"] == 0.1


def test_new_and_legacy_limit_identities_cannot_be_interchanged():
    """Preserve old explicit profiles and require a complete new server descriptor."""
    old = configured_run_limits(load_config("config/expanded-capacity.yaml"))
    assert old == {"contract": CONTRACT, "max_tokens": 10_000_000}
    new = configured_run_limits(load_config("config/operational-plus20.yaml"))
    assert new["contract"] == UPLIFT_CONTRACT
    assert (new["max_tokens"], new["max_requests"], new["max_repairs"]) == (
        12_000_000,
        77,
        3,
    )
    assert validate_run_limits(new) == new
    assert "revision" not in CapacityConfig().model_dump()
    for bad in (
        {**new, "contract": CONTRACT},
        {**new, "max_tokens": 12_000_001},
        {**new, "max_requests": 78},
    ):
        with pytest.raises(ValueError):
            validate_run_limits(bad)


@pytest.fixture
def uplift_plan(case):
    """Exercise the actual planner, complete envelope and new RAGAS batch descriptor."""
    cfg = load_config("config/operational-plus20.yaml")
    result = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        semantic_engine=configured_engine(cfg),
        capacity=cfg.capacity,
        methodology={},
        output_tokens=configured_output_tokens(cfg),
        run_limits=configured_run_limits(cfg),
    )
    assert result["status"] == "admitted", result
    return result


def test_real_072_body_is_inside_expanded_capacity_without_changing_text(
    uplift_plan, case
):
    """Replay the exact failing body; old capacity still rejects its complete bytes."""
    document = json.loads(
        Path("tests/fixtures/body-byte-capacity-072.json").read_text()
    )
    before = copy.deepcopy(document)
    assert nested_json_size(document) == 37159
    check_writer_envelope(document, uplift_plan["envelope"], stage="body")
    assert document == before
    old_cfg = load_config("config/expanded-capacity.yaml")
    old = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        semantic_engine=configured_engine(old_cfg),
        capacity=old_cfg.capacity,
        methodology={},
        output_tokens=configured_output_tokens(old_cfg),
        run_limits=configured_run_limits(old_cfg),
    )
    with pytest.raises(EnvelopeExceeded):
        check_writer_envelope(document, old["envelope"], stage="body")
    assert (
        len(
            [
                unit
                for key in (
                    "introduction",
                    "taxonomy",
                    "methods",
                    "experiments",
                    "open_problems",
                )
                for p in document[key]
                for unit in p["sentences"]
            ]
        )
        == 48
    )


def test_every_maximum_stage_settles_and_restores_with_all_locked_repairs(uplift_plan):
    """Spend the full legal execution path without taking tokens from later stages."""
    plan = uplift_plan
    assert plan["repair_limit"] in (2, 3)
    assert plan["maximum_requests"] == 77
    assert plan["envelope"]["semantic_engine"]["batch_size"] == 20
    assert envelope_context_limit(plan["envelope"]) == 212337
    owner = PlannedBudget(plan)
    for stage, spec in plan["stages"].items():
        for n in range(spec["count"]):
            with request_scope(stage, f"{stage}:{n}"):
                owner.reserve(spec["request_cap"])
                owner.bind_wire({"stage": stage, "n": n})
                owner.begin_request()
                owner.record(
                    {
                        "prompt_tokens": spec["request_cap"] - spec["output_tokens"],
                        "completion_tokens": spec["output_tokens"],
                    }
                )
    snapshot = owner.snapshot()
    assert snapshot["used_tokens"] == plan["required_tokens"]
    assert snapshot["requests_sent"] == plan["planned_requests"] <= 77
    validate_accounting_snapshot(snapshot, expected_plan=plan)


def test_three_repair_attempts_are_versioned_and_persisted(tmp_path):
    """The raised ceiling does not silently upgrade an old two-attempt ledger."""
    from litagent.quality import (
        begin_targeted_repair,
        load_targeted_repairs,
        save_targeted_repairs,
    )

    path = tmp_path / "repairs.json"
    state = load_targeted_repairs(path, "new-profile", limit=3)
    assert state["version"] == "shared-targeted-repair-limit-v2"
    for index in range(3):
        entry = begin_targeted_repair(state, "evaluation", str(index) * 64)
        entry["status"] = "completed"
        save_targeted_repairs(path, state)
        assert load_targeted_repairs(path, "new-profile") == state
    with pytest.raises(ValueError, match="limit_exceeded"):
        begin_targeted_repair(state, "evaluation", "f" * 64)
    with pytest.raises(ValueError):
        load_targeted_repairs(path, "new-profile", limit=2)
    old = load_targeted_repairs(tmp_path / "old.json", "old")
    assert old["version"] == "shared-targeted-repair-limit-v1" and old["limit"] == 2


def test_new_default_selects_expanded_profile_without_overwriting_explicit_legacy():
    """Default non-workbench callers keep their existing resource profile."""
    assert load_config().capacity.body_json_bytes == 38400
    assert configured_run_limits(load_config())["max_requests"] == 77
    assert load_config("config/default.yaml").capacity is None


def test_live_workbench_selects_largest_profile_and_honors_explicit_paths(monkeypatch):
    """Select the full live profile while keeping replay and explicit config behavior."""
    monkeypatch.delenv("LITAGENT_CONFIG", raising=False)
    from litagent import api

    monkeypatch.setenv("LITAGENT_DEMO_MODE", "live")
    live = api._load_runtime_config()
    assert configured_run_limits(live) == configured_run_limits(
        load_config("config/agent-format-repair.yaml")
    )
    assert live.capacity.max_body_units == 80
    assert live.capacity.max_assertions is None
    assert live.capacity.max_assessment_batches is None
    assert (
        configured_run_limits(load_config("config/default.yaml", live_workbench=True))[
            "max_requests"
        ]
        == 200
    )
    monkeypatch.setenv("LITAGENT_CONFIG", "config/default.yaml")
    assert configured_run_limits(api._load_runtime_config())["max_requests"] == 200
    monkeypatch.delenv("LITAGENT_CONFIG")
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "offline")
    assert configured_run_limits(api._load_runtime_config())["max_requests"] == 77


def test_jev_and_independent_audit_resources_follow_the_same_profile(tmp_path):
    """Capabilities describe the resources that actual senders will use."""
    from litagent.extraction_verification import admission_timeout_ms, extraction_config
    from litagent.report_audit import ReportAuditService
    from litagent.workbench import default_verification_options

    cfg = load_config("config/operational-plus20.yaml")
    options = default_verification_options(cfg)
    assert options.extraction_resource_policy == "operational-plus20-v1"
    front = extraction_config(0.6, resource_policy=options.extraction_resource_policy)
    assert (
        front.max_input_tokens,
        front.max_requests,
        front.max_questions_per_batch,
        front.concurrency,
    ) == (4_800_000, 240, 39, 5)
    assert (front.max_request_bytes, front.timeout_seconds, front.max_retries) == (
        28800,
        36,
        0,
    )
    assert admission_timeout_ms(options.extraction_resource_policy) == 1812000
    service = ReportAuditService(cfg, root=tmp_path / "audits")
    judge, jev = service.judge_config(), service.jev_config()
    assert (judge.max_tokens, judge.timeout_seconds, judge.max_retries) == (
        19661,
        216,
        0,
    )
    assert (jev.max_requests, jev.max_input_tokens, jev.concurrency) == (24, 240000, 3)
    caps = service.capabilities()["limits"]
    assert caps["judge_max_output_tokens_per_call"] == judge.max_tokens
    assert caps["jev"]["max_requests"] == jev.max_requests


def test_preparation_expansion_is_versioned_and_settles_without_replaying(tmp_path):
    """Preserve old 500k semantics while exercising the new preparation ledger."""
    from test_preparation_budget import requests, settle

    from litagent.operational_limits import PROFILE
    from litagent.quality import digest
    from litagent.safety.budget import CostBudget
    from litagent.safety.planned_budget import (
        UPLIFT_PREPARATION_CONTRACT,
        preparation_budget_plan,
        settled_preparation_budget,
    )

    with pytest.raises(ValueError, match="strict_total_budget_limit"):
        CostBudget(max_tokens=600000).configure_strict()
    owner = CostBudget(max_tokens=600000, resource_profile=PROFILE)
    owner.configure_strict(halt_on_unknown_usage=True)
    before = owner.snapshot()
    assert CostBudget.from_snapshot(before).snapshot() == before
    rows = requests([50000] * 10)
    for row in rows:
        row["output_tokens"] = 9831
    plan = preparation_budget_plan(
        rows, identity=digest(rows), resource_profile=PROFILE
    )
    assert plan["contract"] == UPLIFT_PREPARATION_CONTRACT
    assert plan["status"] == "admitted"
    assert plan["planned_tokens"] == 550000
    assert plan["maximum_tokens"] == 600000
    assert plan["maximum_requests"] == 77
    assert plan["context_tokens"] == 157287
    assert plan["context_reservation_limit"] == 141558
    oversized = requests([141559])
    oversized[0]["output_tokens"] = 9831
    rejected = preparation_budget_plan(
        oversized, identity=digest(oversized), resource_profile=PROFILE
    )
    assert rejected["status"] == "preflight_rejected"
    assert rejected["reason"] == "budget_plan_context_exceeded"
    owner.activate_plan(plan, path=tmp_path / "preparation.json")
    for row in rows:
        settle(owner, row)
    restored = CostBudget.from_snapshot(owner.snapshot())
    assert settled_preparation_budget(restored).used == 500000
    with pytest.raises(
        ValueError, match="no_replay|count_exceeded|request_envelope_exceeded"
    ):
        settle(owner, rows[0])
    with pytest.raises(ValueError, match="preparation_budget_plan_invalid"):
        preparation_budget_plan(rows, identity=digest(rows))


def test_fulltext_uses_effective_profile_and_preserves_explicit_old_configuration():
    """Check actual consumer bounds, including the former hidden hard caps."""
    from litagent.rag.targeted_fulltext import TargetedFulltext

    old = TargetedFulltext(load_config("config/expanded-capacity.yaml").rag)
    new = TargetedFulltext(load_config("config/operational-plus20.yaml").rag)
    for field in (
        "max_papers",
        "concurrency",
        "max_pdf_bytes",
        "download_timeout",
        "parser_timeout",
    ):
        assert getattr(new, field) == increase(getattr(old, field))
    assert "resource_revision" not in old.config.model_dump()


@pytest.mark.asyncio
async def test_extractor_preflight_measures_the_actual_expanded_wire(tmp_path):
    """Use the real request builder and durable owner, without external transport."""
    from test_extraction_paper_batches import strategy
    from test_question_focused_preparation import attach, paper

    from litagent.llm.client import OpenAICompatibleClient
    from litagent.operational_limits import PROFILE
    from litagent.safety.budget import CostBudget

    owner = CostBudget(max_tokens=600000, resource_profile=PROFILE)
    owner.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=owner
    )
    try:
        extractor = strategy(client)
        groups = [[attach(paper())]]
        bounds = extractor.preflight_batches(
            groups, budget_path=tmp_path / "preparation.json"
        )
        messages, kwargs, _ = extractor.prepare_batch(groups[0])
        assert kwargs["max_tokens"] == 9831
        assert extractor._strict_resources(2)[0] == 19661
        assert (
            owner.snapshot()["plan"]["stages"]["preparation_0"]["output_tokens"]
            == kwargs["max_tokens"]
        )
        assert bounds[0] >= len(json.dumps(messages).encode()) + kwargs["max_tokens"]
        assert owner.snapshot()["requests_sent"] == 0
    finally:
        await client.close()


def test_new_profile_passes_actual_frozen_cli_gates():
    """Keep model-role checks strict while admitting the explicit new allowance."""
    from litagent.writing_v2 import RANGE_WRITING_PLAN
    from scripts.run_frozen_writing import ROUND_PROFILE, check_profile, profile_config

    cfg = profile_config(load_config("config/operational-plus20.yaml"), ROUND_PROFILE)
    check_profile(cfg, ROUND_PROFILE, writing_plan=RANGE_WRITING_PLAN)
    cfg.safety.max_cost_tokens = 600001
    with pytest.raises(ValueError, match="preparation_limit_conflict"):
        check_profile(cfg, ROUND_PROFILE, writing_plan=RANGE_WRITING_PLAN)


def test_effective_config_reports_actual_strict_wire_and_request_limits():
    """Expose the same limits that the actual extractor and budget enforce."""
    from litagent.contracts import build_config_summary

    new = build_config_summary(
        load_config("config/operational-plus20.yaml"), "strict_auto_v2"
    )["effective"]
    old = build_config_summary(load_config("config/default.yaml"), "strict_auto_v2")[
        "effective"
    ]
    assert new["quality_policy"]["request_limit"] == 77
    assert new["extractor"]["strict_max_output_tokens"] == 9831
    assert old["quality_policy"]["request_limit"] == 64
    assert old["extractor"]["strict_max_output_tokens"] == 8192


@pytest.mark.asyncio
async def test_actual_sdk_accepts_expanded_deadline_and_locked_context(uplift_plan):
    """Replace HTTP only while exercising construction, context checks and charging."""
    import httpx
    from openai import AsyncOpenAI

    from litagent.llm.client import OpenAICompatibleClient
    from litagent.observability.context import reset_task_id, set_task_id
    from litagent.safety.budget import strict_request_reservation

    cfg = load_config("config/operational-plus20.yaml")
    owner = PlannedBudget(uplift_plan)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "offline",
        api_key="offline",
        cost_budget=owner,
        request_timeouts=cfg.llm.request_timeouts_seconds,
    )
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    await client._client.close()
    client._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    messages = [{"role": "user", "content": "x" * 130000}]
    output = uplift_plan["stages"]["body"]["output_tokens"]
    assert (
        int(131072 * 0.9)
        < strict_request_reservation(messages, output)
        < uplift_plan["context_reservation_limit"]
    )
    token = set_task_id("synthesis")
    try:
        with request_scope("body", "expanded-context-wire"):
            response = await client.chat(messages, max_tokens=output)
        assert response.content == "{}"
        assert len(sent) == 1 and sent[0].extensions["timeout"]["read"] == 720
        assert owner.used == 20 and owner.snapshot()["requests_sent"] == 1
        assert validate_accounting_snapshot(owner.snapshot(), expected_plan=uplift_plan)
    finally:
        reset_task_id(token)
        await client.close()
    with pytest.raises(ValueError, match="timeout_policy"):
        OpenAICompatibleClient(
            "https://offline.invalid/v1",
            "offline",
            api_key="offline",
            request_timeouts={"synthesis": 720},
        )


@pytest.mark.asyncio
async def test_planner_prompt_and_consumer_share_expanded_query_limit():
    """Allow the fourth query in both the actual request and host clamping."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litagent.agents.planner import SurveyPlanner
    from litagent.llm.client import LLMResponse

    cfg = load_config("config/operational-plus20.yaml")
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                json.dumps(
                    {"sub_queries": ["topic A", "topic B", "topic C", "topic D"]}
                ),
                model="offline",
            )
        ),
        _request_timeouts={},
    )
    result = await SurveyPlanner(llm, config=cfg.planner)._decompose("Explain topic")
    assert len(result) == 4
    assert "2-4 focused" in llm.chat.call_args.args[0][0]["content"]
