"""Check the independent native repair allowance and the archived 078 dispatch."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_ragas_budget import native_plan  # noqa: F401
from test_native_ragas_evaluation import SemanticTransport, verdict_pairs
from test_native_responses import envelope, owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.eval.citation import CitationEvaluator
from litagent.eval.ragas_global_wire import unpack_global_input
from litagent.llm.routing import ModelBoundClient
from litagent.review_structure import structure_sections
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_flat import _canonical_task_payload
from scripts.run_native_repair_trial import (
    RepairTrialBudget,
    prepare_candidate,
    read_sealed_source,
    repair_once,
)


def test_one_round_escrow_stops_extra_stages_before_reserving(native_plan, tmp_path):
    historical = PlannedBudget(native_plan).snapshot()
    path = tmp_path / "candidate-ledger.json"
    budget = RepairTrialBudget(historical, native_plan, path=path)
    cap = native_plan["stages"]["repair"]["request_cap"]
    with request_scope("repair", "new-repair"):
        budget.reserve(cap)
        budget.bind_wire({"fixture": "repair"})
        budget.begin_request()
        budget.record({"prompt_tokens": 20, "completion_tokens": 100})
    settled = budget.snapshot()
    with (
        request_scope("repair", "extra"),
        pytest.raises(ValueError, match="native_trial_allowance_exceeded"),
    ):
        budget.reserve(1)
    with (
        request_scope("body", "unapproved"),
        pytest.raises(ValueError, match="native_trial_stage_not_authorized"),
    ):
        budget.reserve(1)
    assert budget.snapshot() == settled == json.loads(path.read_text())
    with pytest.raises(ValueError, match="native_trial_no_replay"):
        budget.restore(settled)
    with pytest.raises(FileExistsError):
        RepairTrialBudget(historical, native_plan, path=path)


@pytest.fixture(scope="module")
def native_078():
    """Rebuild current resources from the protected original input."""
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/ragas-native-single-20261110-078"
    )
    if not (root / "launch-exit.json").exists():
        pytest.skip("requires the local immutable 078 archive")
    spec = importlib.util.spec_from_file_location(
        "native_trial_source", root / "controller.py"
    )
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    base = controller.baseline
    archive = read_sealed_source(
        root,
        root.with_name("native-global-anchor-20261110"),
        base.INPUT,
        workspace=root.parents[2],
    )
    report, trace, historical, writer = (
        json.loads(archive[root / name])
        for name in (
            "report.json",
            "progress-archive.json",
            "final-budget.json",
            "repair.json.writer-v2.json",
        )
    )
    inputs = json.loads(archive[base.INPUT])
    cfg = base.configuration()
    plan = base.plan_frozen_budget(
        inputs["bundle"],
        budget_policy="planned-budget-v7",
        capacity=cfg.capacity,
        output_tokens=base.configured_output_tokens(cfg),
        run_limits=base.configured_run_limits(cfg),
        preparation_budget_snapshot=inputs["preparation_snapshot"],
        semantic_engine=base.configured_engine(cfg),
        **base.FLAGS,
    )
    results = {
        "extractor": inputs["bundle"]["extractions"],
        "adversarial_review": copy.deepcopy(report["metadata"]),
    }
    return (
        archive,
        report,
        trace["events"],
        historical,
        cfg,
        plan,
        results,
        writer,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["declined", "unsupported", "supported"])
async def test_078_native_trial_reaches_exact_sdk_repair_and_never_replays(
    native_078, tmp_path, result
):
    archive, report, events, historical, config, plan, results, writer = native_078
    requests = []
    transport_errors = []
    semantic = SemanticTransport()
    revised_suffix = "（此处只讨论所引研究的具体设置。）"
    decline = result == "declined"
    supported = result == "supported"

    async def reply(request):
        wire = json.loads(request.content)
        requests.append(wire)
        messages = wire["input"]
        payload = json.loads(messages[1]["content"])
        schema = wire["text"]["format"]["schema"]
        if "targets" in payload:
            raw = {
                "outcome": "declined" if decline else "patched",
                "explanation": "Offline repair path fixture.",
                "patches": [],
            }
            if not decline:
                canonical = _canonical_task_payload(payload)
                fields = schema["properties"]["patches"]["items"]["properties"]
                for index, (target_id, target) in enumerate(
                    canonical["targets"].items()
                ):
                    section, pi, si = target["path"]
                    assert section not in {"abstract", "conclusion"}
                    original = canonical["current_body"][section][pi]["sentences"][si]
                    patch = dict(original)
                    if not index:
                        patch["text"] += revised_suffix
                    if "target_id" in fields:
                        patch["target_id"] = target_id
                    if "source_unit_ids" in fields:
                        patch["source_unit_ids"] = target["original_unit_ids"]
                    raw["patches"].append(patch)
                if "deleted_sources" in schema["properties"]:
                    raw["deleted_sources"] = []
        elif "body_units" in payload:
            raw = {}
            for name in ("abstract", "conclusion"):
                original = writer["summaries"][name][0]["sentences"][0]
                raw[name] = [{**original, "paragraph_start": True}]
        else:
            raw = json.loads((await semantic.chat(messages)).content)
            if "context_pool" in payload and not supported:
                for key, job in raw.items():
                    for statement, verdict in verdict_pairs(payload["jobs"][key], job):
                        if revised_suffix in statement:
                            verdict.update(
                                verdict=0, reason="Offline unsupported candidate."
                            )
            if "analyses" in raw:
                canonical = (
                    unpack_global_input(payload) if "owners" in payload else payload
                )
                titles = dict(
                    structure_sections(
                        "academic_v2", report["metadata"]["report_language"]
                    )
                )
                for analysis in raw["analyses"]:
                    analysis["claim_ids"] = [
                        claim["claim_id"]
                        for claim in canonical["assertions"]
                        if claim["section"] == titles[analysis["section_key"]]
                    ]
                raw["limitations"]["claim_ids"] = next(
                    analysis["claim_ids"]
                    for analysis in raw["analyses"]
                    if analysis["section_key"] == "open_problems"
                )
        Draft202012Validator(schema).validate(raw)
        response = envelope(json.dumps(raw, ensure_ascii=False), model=wire["model"])
        response["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
        return httpx.Response(200, json=response)

    async def handler(request):
        try:
            return await reply(request)
        except Exception as error:
            transport_errors.append(error)
            raise

    owner, _, _ = await owner_for(handler)
    try:
        runner = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
        runner._session_id = "078-repair-offline-fixture"
        runner._targeted_repair_state_path = tmp_path / "repair.json"
        runner._evaluation_progress_root = tmp_path / "evaluation"
        budget = runner._cost_budget = RepairTrialBudget(
            historical, plan, path=tmp_path / "ledger.json"
        )
        owner._cost_budget = budget
        runner._stage_llms = {
            stage: ModelBoundClient(owner, config.llm.model_for(stage), stage)
            for stage in ("generation", "evaluation")
        }
        runner._evaluators = [CitationEvaluator(runner._stage_llm("evaluation"))]
        prepared, trial_results, decision = prepare_candidate(
            runner, report, events, results
        )
        assert prepared["survey"] == report["survey"]
        assert len(decision["targets"]) == 14
        assert len(decision["inventory_findings"]["conflicting_unit_ids"]) == 12
        assert sum(row["requests"] for row in budget.trial_limits.values()) == 13
        assert sum(row["tokens"] for row in budget.trial_limits.values()) == 2760381
        lock = tmp_path / "attempt.lock"
        with pytest.raises(ValueError, match="native_trial_storage_not_isolated"):
            await repair_once(
                runner,
                prepared,
                trial_results,
                decision,
                lock=tmp_path / "other" / "attempt.lock",
                historical_writer=writer,
            )
        assert not requests
        outcome = await repair_once(
            runner,
            prepared,
            trial_results,
            decision,
            lock=lock,
            historical_writer=writer,
        )
        assert not transport_errors, transport_errors
        assert outcome.attempted
        acceptance = prepared["metadata"]["automatic_acceptance"]
        assert json.loads(lock.with_suffix(".delivery.json").read_bytes()) == acceptance
        assert (
            acceptance["source_run_id"]
            == report["metadata"]["automatic_acceptance"]["source_run_id"]
        )
        if decline:
            assert (
                not outcome.committed
                and outcome.reason_code == "writer_repair_declined"
            )
            assert len(requests) == 1
        else:
            assert len(semantic.calls) >= 3, (
                outcome.reason_code,
                prepared["metadata"].get("writer_failure"),
            )
            (progress,) = (tmp_path / "evaluation").glob("*/progress.json")
            evaluation = json.loads(progress.read_bytes())
            details = outcome.candidate_evaluation["faithfulness"]["details"]
            assert evaluation["phase"] == "post_rewrite"
            candidate_sha = hashlib.sha256(outcome.candidate.encode()).hexdigest()
            assert (
                evaluation["report_sha256"]
                == details["source_report_sha256"]
                == candidate_sha
            )
            assert details["stages"]["global_check"] == "completed"
            assert details["unsupported_count"] == (0 if supported else 1)
            assert details["unverified_count"] == 0
            if supported:
                assert outcome.committed, outcome.candidate_quality
                assert acceptance["publishable"], acceptance["failure_codes"]
                assert acceptance["report_sha256"] == candidate_sha
            else:
                assert not outcome.committed
                assert prepared["survey"] == report["survey"]
                assert not acceptance["publishable"]
                assert (
                    acceptance["report_sha256"]
                    == hashlib.sha256(report["survey"].encode()).hexdigest()
                )
        assert prepared["delivery"]["publishable"] is supported
        sent_count = len(requests)
        sent = json.loads(requests[0]["input"][1]["content"])
        assert set(sent["targets"]) == set(decision["targets"])
        assert (
            budget.snapshot()["requests_sent"]
            == historical["requests_sent"] + sent_count
        )
        with pytest.raises(
            ValueError,
            match="native_trial_no_replay|native_trial_requires_new_identity",
        ):
            await repair_once(
                runner,
                prepared,
                trial_results,
                decision,
                lock=lock,
                historical_writer=writer,
            )
        assert len(requests) == sent_count
        assert all(path.read_bytes() == expected for path, expected in archive.items())
    finally:
        await owner.close()
