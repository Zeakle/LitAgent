"""Check frozen import before dispatch and replay the bounded c38 summary retry."""

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from test_native_report_input import source
from test_native_responses import envelope
from test_responses_stream import WireStream, frame
from test_writer_v2 import case  # noqa: F401

from litagent import review_policy, stage_repair
from litagent.frozen_writing import import_retained_draft
from litagent.quality import digest, repair_attempts_remaining
from litagent.writing_v2 import WriterError, compile_document, refresh_summaries

ARCHIVE_SHA = "6f3e06719bfc7de8e3d21e39f91f573b935986839cf864a4a7ecb96cb2854f84"
REAL = (
    Path(__file__).resolve().parents[1]
    / "artifacts/acceptance/vit14-failure-attribution/report-review-no-omissions-once"
)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "archive_sha",
        "missing_method",
        "method_conflict",
        "wrong_section",
        "draft_sha",
        "final_draft",
        "binding",
        "evidence_sha",
    ],
)
def test_import_checks_the_retained_draft(case, tmp_path, fault):
    current = source(case)
    current["writer_document"].update(body=case.body, summaries=case.summaries)
    session = SimpleNamespace(
        packet=case.packet, aliases=case.aliases, envelope=None, writing_plan=None
    )
    original = copy.deepcopy(current["evidence_selection"])
    current["evidence_selection"]["method_by_section"] = {}
    current["editorial_history"] = [{"round": 99}]
    section = "methods"
    if fault == "missing_method":
        del original["method_by_section"][section]
    if fault == "method_conflict":
        current["evidence_selection"]["method_by_section"][section] = "cross_encoder"
    if fault == "wrong_section":
        original["section_evidence_ids"][section] = ["b"]
    if fault == "final_draft":
        current["final_draft"] = current["draft"] + " X"
    if fault == "binding":
        current["writing_bindings"]["units"][0]["source_quote"] += " X"
    if fault == "evidence_sha":
        current["evidence_snapshot_sha256"] = "0" * 64
    archive = tmp_path / "source.json"
    archive.write_text(
        json.dumps(
            {
                "nodes": {
                    "worker:adversarial_review": {
                        "output": {"evidence_selection": original}
                    }
                }
            }
        )
    )
    pinned = {
        "archive_path": archive,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(current["draft"].encode()).hexdigest(),
    }
    if fault == "archive_sha":
        pinned["archive_sha256"] = "0" * 64
    if fault == "draft_sha":
        pinned["report_sha256"] = "0" * 64
    if fault:
        with pytest.raises((ValueError, WriterError)):
            import_retained_draft(session, current, **pinned)
        return
    retained = import_retained_draft(session, current, **pinned)
    assert retained["writing_bindings"] == current["writing_bindings"]
    assert retained["final_draft"] == current["draft"]
    assert retained["evidence_selection"] == original
    assert current["evidence_selection"]["method_by_section"] == {}
    assert retained["editorial_history"] == []


@pytest.mark.asyncio
async def test_real_c38_receipt_retries_only_the_summary(tmp_path, monkeypatch):
    """Reopen real settled receipts and substitute only the next HTTP response."""
    if not REAL.exists():
        pytest.skip("requires the private settled 34-card ViT archive")
    spec = importlib.util.spec_from_file_location(
        "vit_frozen_receipt_fixture", REAL / "run.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    cfg, _, _, source_input, _, plan = fixture.prepare()
    plan_file = json.loads((REAL / "budget-plan.json").read_text())
    assert plan == plan_file
    for name in ("ledger.json", "repairs.json", "repairs.json.writer-v2.json"):
        shutil.copyfile(REAL / name, tmp_path / name)
    report = json.loads((REAL / "evaluated.json").read_text())
    calls = []

    def transport(request):
        wire = json.loads(request.content)
        assert request.url.path.endswith("/responses")
        assert set(wire["text"]["format"]["schema"]["properties"]) == {
            "abstract",
            "conclusion",
        }
        calls.append(request)
        # The replacement is a test response grounded in the original body.
        # Its local validity is independent of any scientific score.
        response = envelope(original_summary, model=wire["model"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=WireStream([frame("response.completed", 0, response=response)]),
        )

    monkeypatch.setattr(
        "litagent.llm.client.AsyncOpenAI",
        lambda **kw: AsyncOpenAI(
            **kw,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(transport), trust_env=False
            ),
        ),
    )
    agent = fixture.LitAgent(cfg)
    agent._cost_budget = fixture.CostBudget(cfg.safety.max_cost_tokens)
    agent._cost_budget.activate_plan(plan, path=tmp_path / "ledger.json")
    agent._wire_llms()
    try:
        doc = source_input["writer_document"]
        session = fixture.writer_session(
            agent._stage_llm("generation"), tmp_path / "repairs.json", fixture.RUN, doc
        )
        retained = import_retained_draft(
            session,
            source_input,
            archive_path=fixture.fx.paths["upstream"],
            archive_sha256=ARCHIVE_SHA,
            report_sha256=fixture.base.REPORT,
        )
        original_summary = session.journal["calls"]["local_summary"]["raw"]
        origin = session.state["attempts"][0]
        provenance = origin["provenance"]
        body = copy.deepcopy(doc["body"])
        for tid, target in sorted(
            provenance["targets"].items(),
            key=lambda item: tuple(item[1]["path"]),
            reverse=True,
        ):
            sec, pi, si = target["path"]
            body[sec][pi]["sentences"][si : target.get("end", si + 1)] = provenance[
                "replacements"
            ][tid]
        assert digest(body) == origin["candidate_sha256"]
        failure = report["metadata"]["writer_failure"]
        assert failure["diagnostics"][0]["reference"] == "c38"
        decision = stage_repair.reject_candidate(
            session,
            WriterError(failure["code"], failure["family"], failure["diagnostics"]),
            retained,
            category="evaluation_revision",
            body=body,
        )
        assert decision["action"] == "repair_summary" and decision["remaining"] == 2
        assert agent._cost_budget.snapshot()["requests_sent"] == 15 and calls == []
        session = fixture.writer_session(
            agent._stage_llm("generation"), tmp_path / "repairs.json", fixture.RUN, doc
        )
        targets = origin["patch_request"]["targets"]
        with (
            review_policy.partition("evaluation_revision"),
            stage_repair.policy(session.envelope),
        ):
            resumed = await session.patch(doc["body"], targets, cause="evaluation")
            assert resumed == body and not calls
            summaries = await refresh_summaries(
                session, doc["body"], resumed, doc["summaries"], targets
            )
        text, bindings = compile_document(
            resumed,
            summaries,
            retained["fact_cards"],
            doc["aliases"],
            retained["report_language"],
            retained["review_methodology"],
            writing_plan=session.writing_plan,
            envelope=session.envelope,
        )
        assert (
            len(calls) == 1
            and text != retained["draft"]
            and hashlib.sha256(text.encode()).hexdigest() == bindings["report_sha256"]
        )
        assert session.state["attempts"][-1]["operation"] == "summary_only"
        assert repair_attempts_remaining(session.state, "evaluation_revision") == 1
        assert agent._cost_budget.snapshot()["requests_sent"] == 16
        assert json.loads((REAL / "ledger.json").read_text())["requests_sent"] == 15
    finally:
        await agent.cleanup()


@pytest.mark.asyncio
async def test_archive_import_uses_checked_source_before_evaluation(
    case, tmp_path, monkeypatch
):
    """Exercise the same imported worker result through preflight and final export."""
    from test_inline_citations import InlineGateway
    from test_llm_stage_repair import stage_config
    from test_separate_review_pipeline import execute

    from litagent.contracts import build_config_summary
    from litagent.eval.citation import CitationEvaluator
    from litagent.eval.consistency import ConsistencyEvaluator
    from litagent.eval.engine_policy import configured_engine, native_writer_budget_plan
    from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
    from litagent.frozen_writing import reevaluate_imported_draft
    from litagent.output_capacity import configured_output_tokens
    from litagent.run_limits import configured_run_limits
    from litagent.runner import LitAgent
    from litagent.safety.budget import CostBudget
    from litagent.workbench import RunOptions, VerificationOptions

    config = stage_config()
    gateway = InlineGateway(case, reviews=0)
    await execute(case, tmp_path, monkeypatch, gateway, config=config)
    (ledger_file,) = list(tmp_path.rglob("*.budget-ledger.json"))
    archive_path = (
        tmp_path / "archive" / ledger_file.name.removesuffix(".budget-ledger.json")
    )
    archive = json.loads(archive_path.read_text())
    selected = archive["nodes"]["worker:adversarial_review"]["output"]
    extracts = archive["nodes"]["worker:extract"]["output"]
    selected["evidence_selection"]["method_by_section"] = {}
    selected["editorial_history"] = None
    options = RunOptions(
        report_language="en",
        review_structure="academic_v2",
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v7",
        capacity_policy="adaptive-capacity-v1",
        verification=VerificationOptions(judge=False, jev=False),
    )
    agent = LitAgent(config)
    agent._session_id = "offline-imported-survey"
    agent._targeted_repair_state_path = tmp_path / "import" / "repairs.json"
    agent._evaluation_progress_root = tmp_path / "import" / "evaluation"
    agent._config_summary = build_config_summary(config, "strict_auto_v2")
    agent._cost_budget = CostBudget(config.safety.max_cost_tokens)
    doc = selected["writer_document"]
    plan = native_writer_budget_plan(
        doc["packet"],
        doc["aliases"],
        budget_policy=options.budget_policy,
        methodology=selected["review_methodology"],
        semantic_engine=configured_engine(config),
        capacity=config.capacity,
        run_limits=configured_run_limits(config),
        output_tokens=configured_output_tokens(config),
    )
    assert plan["envelope"] == doc["budget_envelope"]
    agent._cost_budget.activate_plan(plan, path=tmp_path / "import" / "ledger.json")
    agent._wire_llms()
    ev = agent._stage_llm("evaluation")
    agent._evaluators = [
        CitationEvaluator(ev),
        RagasFaithfulnessEvaluator(config, llm=ev),
        ConsistencyEvaluator(ev),
    ]
    pins = {
        "archive_path": archive_path,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(selected["final_draft"].encode()).hexdigest(),
    }
    try:
        with pytest.raises(ValueError, match="report_sha256_mismatch"):
            await reevaluate_imported_draft(
                agent,
                selected,
                extracts,
                options,
                **{**pins, "report_sha256": "0" * 64},
            )
        assert agent._cost_budget.snapshot()["requests_sent"] == 0
        count = gateway.global_calls
        report = await reevaluate_imported_draft(
            agent, selected, extracts, options, **pins
        )
        assert gateway.global_calls > count
        assessment = report["evaluation"]["faithfulness"]
        assert assessment["details"]["stages"]["global_check"] == "completed"
        assert assessment["score"] is not None
        assert report["metadata"]["editorial_history"] == []
        assert report["metadata"]["review_workflow"]["reviews"] == []
        assert (
            report["metadata"]["automatic_acceptance"]["report_sha256"]
            == hashlib.sha256(report["survey"].encode()).hexdigest()
        )
        entries = agent._cost_budget.snapshot()["entries"]
        assert entries and not {v["stage"] for v in entries.values()}.intersection(
            {"body", "body_initial", "reviewer", "summary"}
        )
        sent = agent._cost_budget.snapshot()["requests_sent"]
        with pytest.raises(ValueError, match="new_planned_run"):
            await reevaluate_imported_draft(agent, selected, extracts, options, **pins)
        assert agent._cost_budget.snapshot()["requests_sent"] == sent
    finally:
        await agent.cleanup()
