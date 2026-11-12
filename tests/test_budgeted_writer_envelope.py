"""Exercise declared whole-report scope through real consumers and HTTP-only mocks."""

import copy
import json

import httpx
import pytest
from test_native_responses import envelope as native_response
from test_native_responses import owner_for
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import EvaluatorTransport, Provider, case  # noqa: F401

from litagent.quality import digest
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    ENVELOPE_BUDGET_CONTRACT,
    EnvelopeExceeded,
    PlannedBudget,
    check_inventory_envelope,
    check_writer_envelope,
    nested_json_size,
    request_scope,
    validate_writer_envelope,
)
from litagent.writing_v2 import writer_budget_plan


@pytest.mark.asyncio
async def test_evaluation_rejects_a_different_capacity_before_transport(plan):
    from litagent.config import EvalConfig
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped

    class NoTransport:
        _cost_budget = PlannedBudget(plan)

        async def chat(self, *args, **kwargs):
            raise AssertionError("An identity conflict must not send a request")

    llm = NoTransport()
    before = llm._cost_budget.snapshot()
    with pytest.raises(ScopedEvaluationError, match="envelope_identity_conflict"):
        await evaluate_scoped(
            llm=llm,
            survey="Scientific body.",
            context={"quality_policy": "strict_auto_v2", "writing_bindings": {}},
            config=EvalConfig(),
        )
    assert llm._cost_budget.snapshot() == before


def test_actual_chapter_builder_respects_nested_whole_body_bound(plan, case):
    from litagent.safety.budget import strict_request_reservation
    from litagent.writing_v2 import (
        BODY,
        NATIVE_WIRE_CONTRACT,
        RANGE_WRITING_PLAN,
        chapter_request_input,
        prepare_planned_input,
        provider_response_format,
        request_messages,
    )

    scope = plan["envelope"]
    for text in ['\\"\t中文😀', "\x00", "x"]:
        prior = copy.deepcopy(case.body)
        prior.pop(BODY[-1])
        prior[BODY[0]][0]["sentences"][0]["text"] = text
        check_writer_envelope(prior, scope, stage="body")
        system, payload, schema = chapter_request_input(
            case.packet,
            case.aliases,
            BODY[-1],
            prior,
            prior,
            RANGE_WRITING_PLAN,
            scope,
        )
        system, payload = prepare_planned_input(
            system,
            payload,
            case.packet,
            case.aliases,
            RANGE_WRITING_PLAN,
            scope,
        )
        size = strict_request_reservation(
            request_messages(
                system, payload, schema, wire_contract=NATIVE_WIRE_CONTRACT
            ),
            20000,
            provider_response_format(
                "synthesis", schema, wire_contract=NATIVE_WIRE_CONTRACT
            ),
        )
        assert size <= scope["request_caps"]["body"]


def test_policy_and_profile_add_capacity_without_reinterpreting_old_defaults():
    from litagent.workbench import RunOptions, VerificationOptions
    from scripts import run_frozen_writing as campaign

    common = dict(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v2",
        verification=VerificationOptions(judge=False, jev=False),
    )
    versions = [
        RunOptions(**common, budget_policy=f"planned-budget-v{v}") for v in (2, 3, 4)
    ]
    assert len({v.fingerprint() for v in versions}) == 3
    assert "budget_policy" not in RunOptions().model_dump()
    v3 = campaign.profile_parameters(
        campaign.ADAPTIVE_PROFILE, writing_plan=campaign.RANGE_WRITING_PLAN
    )
    v4 = campaign.profile_parameters(
        campaign.BUDGETED_PROFILE, writing_plan=campaign.RANGE_WRITING_PLAN
    )
    assert v4 == {**v3, "budget_policy": "planned-budget-v4"}
    assert (
        campaign.profile_options(campaign.BUDGETED_PROFILE).budget_policy
        == ENVELOPE_BUDGET_CONTRACT
    )


def test_frozen_manifest_binds_capacity_and_all_exports(manifest, case, monkeypatch):
    import csv
    from pathlib import Path

    from litagent.report_audit import atomic_json
    from scripts import run_frozen_writing as campaign

    root, config = manifest
    read = campaign.read_bundle

    def bundle(path):
        return {
            **read(path),
            "selection": case.selection.to_dict(),
            "fact_cards_sha256": digest(case.cards),
            "requirements": case.packet["requirements"],
            "methodology": {},
        }

    monkeypatch.setattr(campaign, "read_bundle", bundle)
    config = campaign.profile_config(config, campaign.BUDGETED_PROFILE)
    target = root.with_name("declared-capacity-offline")
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.BUDGETED_PROFILE,
        writing_plan=campaign.RANGE_WRITING_PLAN,
    )
    frozen = campaign.validate(target, config)
    assert frozen["options"]["budget_policy"] == ENVELOPE_BUDGET_CONTRACT
    assert all(p["status"] == "admitted" for p in frozen["budget_plans"].values())
    assert frozen["limits"]["total_tokens"] <= 3000000
    public = campaign.export(
        target,
        {"identity": frozen["identity"], "status": "offline_fixture", "cases": []},
        frozen,
    )
    out = Path("benchmarks/survey/results") / target.name
    html = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            html.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    for row in csv.DictReader((out / "benchmark.csv").open()):
        scope = frozen["budget_plans"][row["case_id"]]["envelope"]
        assert int(row["budget_body_units"]) == scope["body_max_units"]
        assert int(row["budget_model_assertions"]) == scope["model_assertions"]
        assert int(row["budget_assessment_batches"]) == scope["assessment_batches"]
    assert "Declared" in html or "声明" in html
    bad = copy.deepcopy(frozen)
    next(iter(bad["budget_plans"].values()))["envelope"]["model_assertions"] = 128
    bad["identity"] = digest({k: v for k, v in bad.items() if k != "identity"})
    atomic_json(target / "manifest.json", bad)
    with pytest.raises(ValueError):
        campaign.validate(target, config)


def test_declared_batch_wire_limit_splits_before_any_paid_request(plan, monkeypatch):
    from litagent.config import EvalConfig
    from litagent.eval import batched

    scope = copy.deepcopy(plan["envelope"])
    scope.update(model_assertions=32, assessment_batches=2)
    scope["identity"] = digest({k: v for k, v in scope.items() if k != "identity"})
    claims = [{"claim_id": str(i)} for i in range(3)]
    payload = {"assertions": claims, "writer_envelope": scope}
    original = copy.deepcopy(payload)
    monkeypatch.setattr(batched, "batch_payload", lambda p, c: {**p, "assertions": c})
    monkeypatch.setattr(
        batched, "request_size", lambda p, v, o: 40000 * len(v["assertions"])
    )
    result = batched.plan_batches(payload, EvalConfig())
    assert [len(p["assertions"]) for p in result] == [2, 1]
    assert payload == original
    payload["assertions"] = [{"claim_id": str(i)} for i in range(5)]
    with pytest.raises(EnvelopeExceeded, match="evaluation_envelope_exceeded"):
        batched.plan_batches(payload, EvalConfig())


@pytest.fixture
def plan(case):
    return writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy=ENVELOPE_BUDGET_CONTRACT,
    )


def test_full_path_locks_capacity_and_survives_maximum_consumption(plan):
    assert plan["status"] == "admitted" and plan["planned_tokens"] <= 1000000
    scope = plan["envelope"]
    assert scope["body_max_units"] < 66 and scope["model_assertions"] < 128
    assert plan["stages"]["assessment"]["count"] == scope["assessment_batches"] * (
        1 + plan["repair_limit"]
    )
    budget = PlannedBudget(plan)
    for stage, spec in plan["stages"].items():
        for index in range(spec["count"]):
            with request_scope(stage, f"{stage}:{index}"):
                budget.reserve(spec["request_cap"])
                budget.begin_request()
                budget.record(
                    {"prompt_tokens": spec["request_cap"], "completion_tokens": 0}
                )
    assert budget.used == plan["required_tokens"]
    restored = CostBudget.from_snapshot(budget.snapshot())
    assert restored.snapshot() == budget.snapshot()


@pytest.mark.parametrize("repairs", [0, 1, 2])
def test_all_declared_repair_paths_reserve_full_reassessment(plan, repairs):
    """Use explicit small server bounds, without claiming fixed-case admission."""
    from litagent.safety.planned_budget import build_budget_plan

    scope = copy.deepcopy(plan["envelope"])
    scope.update(
        body_max_units=10,
        model_assertions=16,
        assessment_batches=1,
        frozen_component_reservation=20000,
    )
    scope["request_caps"] = {k: 40000 for k in scope["request_caps"]}
    scope["identity"] = digest({k: v for k, v in scope.items() if k != "identity"})
    counts = dict(
        body=5,
        summary=1 + repairs,
        reviewer=1,
        inventory=1 + repairs,
        assessment=1 + repairs,
        **{"global": 1 + repairs},
    )
    if repairs:
        counts["repair"] = repairs
    outputs = dict(
        body=20000,
        summary=20000,
        reviewer=4096,
        inventory=16384,
        assessment=16384,
        repair=20000,
        **{"global": 16384},
    )
    stages = {
        k: dict(
            count=n,
            request_cap=40000,
            output_tokens=outputs[k],
            bound="host_envelope_upper_bound",
        )
        for k, n in counts.items()
    }
    declared = build_budget_plan(
        stages,
        identity=scope["source_identity"],
        contract=ENVELOPE_BUDGET_CONTRACT,
        repairs=repairs,
        envelope=scope,
    )
    assert declared["status"] == "admitted"
    backend = PlannedBudget(declared)
    for stage, spec in declared["stages"].items():
        for i in range(spec["count"]):
            with request_scope(stage, f"{stage}:{i}"):
                backend.reserve(spec["request_cap"])
                backend.begin_request()
                backend.record(
                    {"prompt_tokens": spec["request_cap"], "completion_tokens": 0}
                )
    assert backend.used == declared["required_tokens"] <= 1000000
    assert backend.snapshot()["requests_sent"] == declared["planned_requests"]
    assert CostBudget.from_snapshot(backend.snapshot()).snapshot() == backend.snapshot()


@pytest.mark.parametrize(
    "fault",
    [
        "boolean",
        "missing",
        "unknown",
        "hash",
        "too_few_assertions",
        "too_few_batches",
        "output_reserve",
    ],
)
def test_invalid_envelope_cannot_be_rehashed_into_admission(plan, fault):
    scope = copy.deepcopy(plan["envelope"])
    if fault == "boolean":
        scope["body_max_units"] = True
    elif fault == "missing":
        scope.pop("claim_json_bytes")
    elif fault == "unknown":
        scope["arbitrary"] = 1
    elif fault == "hash":
        scope["source_identity"] = "not-a-hash"
    elif fault == "too_few_assertions":
        scope["model_assertions"] = 1
    elif fault == "too_few_batches":
        scope["assessment_batches"] = 1
    else:
        scope["request_caps"]["assessment"] = 1
    scope["identity"] = digest({k: v for k, v in scope.items() if k != "identity"})
    with pytest.raises(ValueError, match="envelope_invalid"):
        validate_writer_envelope(scope)


@pytest.mark.parametrize("text", ['\\"\t', "中文😀", "\x00" * 30])
def test_nested_encoding_is_measured_without_character_ratios(text):
    value = {"text": text}
    assert (
        nested_json_size(value)
        == len(
            json.dumps(
                json.dumps(value, ensure_ascii=False), ensure_ascii=False
            ).encode()
        )
        - 2
    )


def test_output_and_inventory_limits_do_not_truncate(plan, case):
    scope = plan["envelope"]
    body = copy.deepcopy(case.body)
    body["methods"][0]["sentences"] *= scope["body_max_units"]
    before = copy.deepcopy(body)
    with pytest.raises(EnvelopeExceeded):
        check_writer_envelope(body, scope, stage="body")
    assert body == before
    claims = [{"claim_text": "x"}] * (scope["model_assertions"] + 1)
    with pytest.raises(EnvelopeExceeded):
        check_inventory_envelope(claims, scope)
    assert len(claims) == scope["model_assertions"] + 1
    with pytest.raises(EnvelopeExceeded):
        check_inventory_envelope(
            [{"claim_text": "x" * scope["claim_json_bytes"]}], scope
        )


def test_request_scope_is_not_a_late_budget_relabel(plan):
    budget = PlannedBudget(plan)
    before = budget.snapshot()
    with request_scope("assessment", "too-large"):
        with pytest.raises(EnvelopeExceeded) as failure:
            budget.check_wire([{"role": "user", "content": "x" * 100000}], 16384)
    assert failure.value.diagnostics[0]["field"] == "complete_wire_reservation"
    assert budget.snapshot() == before
    with pytest.raises(EnvelopeExceeded):
        budget.check_stage_requests(
            "assessment", [20000] * (plan["envelope"]["assessment_batches"] + 1)
        )
    assert budget.snapshot() == before


def native_fixture_value(value, schema, root=None):
    """Let the mock provider follow the actual requested container types.

    Deliberate over-capacity fixtures retain every extra value, including values
    beyond the last declared slot. The real consumer must reject that response;
    the mock cannot truncate it or pre-validate away the scientific counterexample.
    """
    root = schema if root is None else root
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return native_fixture_value(value, target, root)
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if isinstance(value, list) and set(properties) == {"head", "tail"}:
            if not value:
                return []
            return {
                "head": native_fixture_value(value[0], properties["head"], root),
                "tail": native_fixture_value(value[1:], properties["tail"], root),
            }
        if isinstance(value, dict):
            return {
                key: native_fixture_value(child, properties.get(key, {}), root)
                for key, child in value.items()
            }
    if schema.get("type") == "array" and isinstance(value, list):
        slots = schema.get("prefixItems", [])
        fallback = schema.get("items", {})
        if not isinstance(fallback, dict):
            fallback = slots[-1] if slots else {}
        return [
            native_fixture_value(
                child, slots[index] if index < len(slots) else fallback, root
            )
            for index, child in enumerate(value)
        ]
    return copy.deepcopy(value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "supported",
        "unsupported",
        "body_scope",
        "inventory_scope",
        "unknown_usage",
        "runtime_plan_drift",
        "caller_mutation",
        "source_drift_before",
        "source_drift_after",
    ],
)
async def test_real_frozen_runtime_keeps_scope_science_and_usage_separate(
    case, tmp_path, monkeypatch, mode
):
    from model_config_helpers import load_previous_default_config as load_config

    from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
    from litagent.review_structure import build_methodology
    from litagent.runner import LitAgent, RunPolicy
    from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

    provider = Provider(case)
    evaluator = EvaluatorTransport(unsupported=mode == "unsupported")
    sent = []

    async def handler(request):
        body = json.loads(request.content)
        native = "input" in body
        messages = copy.deepcopy(body["input"] if native else body["messages"])
        rf = (
            {
                "type": "json_schema",
                "json_schema": {
                    k: v for k, v in body["text"]["format"].items() if k != "type"
                },
            }
            if native
            else body["response_format"]
        )
        payload = json.loads(messages[1]["content"])
        is_eval = any(
            k in payload
            for k in ("allowed_citation_aliases", "cited_materials", "report_structure")
        )
        sent.append((body, copy.deepcopy(payload)))
        if payload.get("assessment_input_contract") and "cited_materials" in payload:
            for claim in payload["assertions"]:
                if "evidence_aliases" in claim:
                    claim["evidence"] = [
                        {"evidence_alias": a, "material_ref": a}
                        for a in claim.pop("evidence_aliases")
                    ]
            messages[1]["content"] = json.dumps(payload)
        answer = await (evaluator if is_eval else provider).chat(
            messages, response_format=rf
        )
        if (
            mode == "body_scope"
            and payload.get("requested_body_section") == "introduction"
        ):
            value = json.loads(answer.content)
            unit = value["introduction"][0]["sentences"][0]
            value["introduction"] = [
                {"sentences": [copy.deepcopy(unit) for _ in range(30)]},
                {"sentences": [copy.deepcopy(unit) for _ in range(30)]},
            ]
            answer.content = json.dumps(value)
        if mode == "inventory_scope" and "allowed_citation_aliases" in payload:
            value = json.loads(answer.content)
            fragment = next(iter(value["fragments"].values()))
            fragment["assertions"].extend(
                {
                    "kind": "scientific",
                    "claim_text": "Controlled assertion "
                    + chr(65 + i // 26)
                    + chr(65 + i % 26),
                }
                for i in range(128)
            )
            answer.content = json.dumps(value)
        if native:
            assert body["thinking"] == {"type": "disabled"} and "reasoning" not in body
            output = native_fixture_value(
                json.loads(answer.content), body["text"]["format"]["schema"]
            )
            response = native_response(json.dumps(output), model=body["model"])
            if mode == "unknown_usage":
                response.pop("usage")
            return httpx.Response(200, json=response)
        return httpx.Response(
            200,
            json={
                "id": "offline-eval",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": answer.content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, _, _ = await owner_for(handler)
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.review_max_tokens = 4096
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**kwargs):
        owner._model, owner._cost_budget, owner._trace_hook = (
            kwargs["model"],
            kwargs["cost_budget"],
            kwargs["trace_hook"],
        )
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    methodology = build_methodology(
        case.packet["query"],
        {
            "__search_queries__": {"search_0": "relation networks"},
            "search_0": [{"paper_id": "relation"}],
            "dedup": [{"paper_id": "relation"}],
            "relevance_gate": [{"paper_id": "relation"}],
        },
        source_outcomes=[
            {
                "task_id": "search_0",
                "query": "relation networks",
                "source": "arxiv",
                "status": "completed",
                "result_count": 1,
            }
        ],
        started_at=1700000000,
    )
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": methodology,
        "fact_cards_sha256": digest(case.cards),
        "requested_methods": ["EViT", "Relation networks"],
    }
    bundle["extractions"][0]["fulltext_acquisition"] = {
        "status": "failed",
        "reason": "offline_unavailable",
    }
    expected_plan = plan_frozen_budget(bundle, budget_policy=ENVELOPE_BUDGET_CONTRACT)
    original_bundle = copy.deepcopy(bundle)
    if mode == "runtime_plan_drift":
        from litagent import review_structure

        prepare = review_structure.prepare_writer_methodology

        def drift(*args, **kwargs):
            prepared = prepare(*args, **kwargs)
            prepared["evidence_counts"]["selected"] += 1
            return prepared

        monkeypatch.setattr(
            "litagent.agents.synthesis.prepare_writer_methodology", drift
        )
    if mode == "caller_mutation":
        from litagent.frozen_writing import FrozenSelector

        select = FrozenSelector.select

        async def mutate(self, *args, **kwargs):
            bundle["methodology"]["evidence_counts"] = {"selected": 999}
            bundle["requirements"].clear()
            return await select(self, *args, **kwargs)

        monkeypatch.setattr(FrozenSelector, "select", mutate)
    try:
        if mode in {"runtime_plan_drift", "source_drift_before", "source_drift_after"}:
            from litagent.writing_v2 import WriterError

            def guard():
                if mode == "source_drift_before" or (
                    mode == "source_drift_after" and sent
                ):
                    raise WriterError("frozen_campaign_source_changed", "identity")

            code = (
                "frozen_budget_plan_drift"
                if mode == "runtime_plan_drift"
                else "frozen_campaign_source_changed"
            )
            with pytest.raises(WriterError, match=code):
                await run_frozen(
                    agent,
                    bundle,
                    "v4-" + mode,
                    native_provider_api="responses",
                    budget_policy=ENVELOPE_BUDGET_CONTRACT,
                    expected_budget_plan=expected_plan,
                    frozen_input_guard=guard,
                )
            if mode == "source_drift_after":
                assert len(sent) == 1 and agent._cost_budget.used == 120
                journal = json.loads(
                    (tmp_path / "repair.json.writer-v2.json").read_text()
                )
                entry = journal["calls"]["body:introduction"]
                assert entry["raw"] and entry["provider_receipt"]
                assert entry["failure"]["code"] == code
            else:
                assert not sent
                if agent._cost_budget is not None:
                    assert agent._cost_budget._planned_backend is None
            assert bundle == original_bundle
            return
        result = await run_frozen(
            agent,
            bundle,
            "v4-" + mode,
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy=ENVELOPE_BUDGET_CONTRACT,
            expected_budget_plan=expected_plan,
        )
        backend = agent._cost_budget._planned_backend
        assert backend.plan["contract"] == ENVELOPE_BUDGET_CONTRACT
        assert backend.plan == expected_plan
        assert backend.snapshot()["requests_sent"] == len(sent)
        if mode == "unknown_usage":
            assert len(sent) == 1 and backend.snapshot()["usage_uncertain"]
        elif mode == "body_scope":
            assert len(sent) == 1 and backend.used == 120
            assert "writer_native_capacity_exceeded" in json.dumps(result)
        elif mode == "inventory_scope":
            assert not any("cited_materials" in p for _, p in sent)
            assert "evaluation_envelope_exceeded" in json.dumps(result)
            assert backend.used == len(sent) * 120
        else:
            details = result["evaluation"]["faithfulness"]["details"]
            assert details["coverage"]["complete"]
            assert (
                details["stages"]["writer_envelope"]["identity"]
                == backend.plan["envelope"]["identity"]
            )
            assert result["evaluation"]["faithfulness"]["score"] == (
                0 if mode == "unsupported" else 1
            )
            assert backend.used == len(sent) * 120
            assert result["quality"]["status"] == (
                "failed" if mode == "unsupported" else "passed"
            )
            assert all(
                p["writing_envelope"]["body_max_units"]
                == backend.plan["envelope"]["body_max_units"]
                for _, p in sent
                if "requested_body_section" in p
            )
        if mode != "caller_mutation":
            assert bundle == original_bundle
    finally:
        await agent.cleanup()
