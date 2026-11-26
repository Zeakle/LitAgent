"""Check frozen import, settled historical receipts, and current Evaluation."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_report_input import source
from test_writer_v2 import case  # noqa: F401

from litagent.frozen_writing import import_retained_draft
from litagent.llm.responses import decode_private_receipt
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget
from litagent.writing_v2 import WriterError

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


def test_real_c38_receipt_is_readable_but_cannot_resume_on_current_identity():
    """Authenticate the settled invalid summary without reopening its old ledger."""
    if not REAL.exists():
        pytest.skip("requires the private settled 34-card ViT archive")
    originals = {
        name: (REAL / name).read_bytes()
        for name in (
            "evaluated.json",
            "budget-plan.json",
            "ledger.json",
            "repairs.json.writer-v2.json",
        )
    }
    report, plan, ledger, writer = map(json.loads, originals.values())
    snapshot = validate_accounting_snapshot(ledger, expected_plan=plan)
    failure = report["metadata"]["writer_failure"]
    assert failure["code"] == "writer_inline_citation_unknown"
    assert (
        failure["source_report_sha256"]
        == hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    [diagnostic] = failure["diagnostics"]
    assert diagnostic["reference"] == "c38"
    assert diagnostic["path"] == ["conclusion", 0, 0]
    assert "c38" not in report["metadata"]["writer_document"]["aliases"]

    summary = writer["calls"]["summary_repair_1"]
    wire, response = decode_private_receipt(summary["provider_receipt"])
    assert set(wire["text"]["format"]["schema"]["properties"]) == {
        "abstract",
        "conclusion",
    }
    [message] = [row for row in response["output"] if row["type"] == "message"]
    [content] = message["content"]
    assert content["text"] == summary["raw"]
    text = json.loads(content["text"])["conclusion"][0]["text"]
    assert text[diagnostic["offset"] :].startswith("[C:c38]")
    settlement = snapshot["entries"][f"{writer['identity']}:summary_repair_1"]
    assert settlement["stage"] == "evaluation_revision/summary"
    assert settlement["status"] == "settled" and settlement["sent"]
    assert settlement["actual_tokens"] == response["usage"]["total_tokens"] == 12926
    assert snapshot["requests_sent"] == 15 and not snapshot["usage_uncertain"]

    # Current summary-only continuation is tested with a current writer identity.
    with pytest.raises(ValueError, match="stage_repair_identity_invalid"):
        PlannedBudget(plan)
    assert all((REAL / name).read_bytes() == raw for name, raw in originals.items())


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
