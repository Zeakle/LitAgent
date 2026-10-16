"""Reject real repeated repair content before spending a complete reevaluation."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_fragment_repair import case
from test_reference_occurrence_policy import bound

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.eval.base import CTX_EVIDENCE, CTX_SELECTED_EVIDENCE
from litagent.llm.client import LLMResponse
from litagent.quality import (
    load_targeted_repairs,
    targeted_repair_state,
    validate_repair_content,
)
from litagent.runner import LitAgent


def test_actual_paid_duplicate_is_located_without_mutating_archived_text():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/duplicate-repair-real.json").read_text()
    )
    original = fixture["original"]
    assert (
        hashlib.sha256(original.encode()).hexdigest() == fixture["source_report_sha256"]
    )
    targets = [
        {k: d[k] for k in ("source_fragment_id", "source_start", "source_end")}
        for d in fixture["diagnostics"]
    ]
    result = validate_repair_content(original, fixture["candidate"], targets)
    assert result["status"] == "failed"
    assert result["issues"]
    for issue in result["issues"]:
        for row in issue["protected_fragments"]:
            assert (
                original[row["source_start"] : row["source_end"]] == row["source_quote"]
            )
        for row in issue["candidate_fragments"]:
            assert (
                fixture["candidate"][row["source_start"] : row["source_end"]]
                == row["source_quote"]
            )


@pytest.mark.parametrize(
    "changed",
    [
        "Method A improves accuracy by 25% on dataset B.",
        "Method A improves accuracy by 50% on dataset A.",
        "Method B improves accuracy by 25% on dataset A.",
        "Method A does not improve accuracy by 25% on dataset A.",
    ],
)
def test_different_conditions_numbers_entities_and_negation_are_not_duplicates(changed):
    original = "## Methods\n\nMethod A improves accuracy by 25% on dataset A [E:a].\n"
    assert (
        validate_repair_content(original, original + changed, [])["status"] == "passed"
    )
    assert (
        validate_repair_content(
            original,
            original
            + "\n## Conclusion\n\nMethod A improves accuracy by 25% on dataset A [E:a].",
            [],
        )["status"]
        == "passed"
    )
    assert validate_repair_content(original * 2, original * 2, [])["status"] == "passed"


@pytest.mark.asyncio
async def test_real_repair_rejects_duplicate_before_eval_and_uses_only_remaining_slot(
    tmp_path,
):
    body, evaluation, cards, targets, response, good = case()
    cid = next(k for k, c in cards.items() if c["evidence_id"] == "a")
    for patch in response["patches"].values():
        patch["first_sentence"] = {
            "text": "EViT improves speed.",
            "fact_card_ids": [cid],
        }
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                json.dumps(response),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "usage_available": True,
                },
            )
        )
    )
    agent = LitAgent(load_config())
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    ledger = {
        c["evidence_id"]: {
            "paper_id": c["paper_id"],
            "text": c["source_quote"],
            "supporting_text": c["source_quote"],
        }
        for c in cards.values()
    }
    agent._build_eval_context = lambda **kwargs: {
        CTX_EVIDENCE: ledger,
        CTX_SELECTED_EVIDENCE: list(ledger.values()),
    }
    agent._evaluate = AsyncMock()
    report = {
        "survey": body,
        "metadata": {
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
            "fact_cards": cards,
            "targeted_repair_state": targeted_repair_state(None, agent._session_id),
            "writing_bindings": {
                "report_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "locations": [],
                "scope_notice_contract": "scope-notice-v1",
                "scope_notices": {},
            },
        },
    }
    kwargs = dict(
        query="EViT",
        report_data=report,
        results={},
        initial_evaluation=evaluation,
        initial_quality={"status": "failed"},
        report_language="en",
    )
    first = await agent._attempt_evidence_rewrite(**kwargs)
    assert first.reason_code == "strict_repair_duplicate_content"
    assert first.candidate_evaluation is None
    agent._evaluate.assert_not_awaited()
    assert report["survey"] == body and good in body
    assert (
        len(
            load_targeted_repairs(agent._targeted_repair_path(), agent._session_id)[
                "attempts"
            ]
        )
        == 1
    )
    second = await agent._attempt_evidence_rewrite(**kwargs)
    assert second.reason_code == "strict_repair_duplicate_content"
    sent = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert body == report["survey"]
    assert "previous_local_repair_failure" in json.dumps(sent)
    third = await agent._attempt_evidence_rewrite(**kwargs)
    assert third.reason_code == "strict_targeted_repair_limit_exceeded"
    assert llm.chat.await_count == 2
    agent._evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_loop_keeps_original_after_local_failure_and_retries_only_twice(
    tmp_path,
):
    from litagent.orchestrator.task_graph import SubTask, TaskGraph
    from litagent.runner import RewriteOutcome
    from litagent.workbench import RunOptions

    agent = LitAgent(load_config())
    agent._targeted_repair_state_path = tmp_path / "state.json"
    evaluation = {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": False},
    }
    agent._evaluate = AsyncMock(return_value=evaluation)
    seen = []

    async def local_failure(**kwargs):
        working = kwargs["report_data"]
        seen.append(working["survey"])
        working["metadata"]["local_repair_diagnostics"] = {
            "source_report_sha256": hashlib.sha256(b"original").hexdigest()
        }
        return RewriteOutcome(
            True,
            False,
            "duplicate candidate",
            None,
            None,
            None,
            "strict_repair_duplicate_content",
        )

    agent._attempt_evidence_rewrite = AsyncMock(side_effect=local_failure)
    graph = TaskGraph()
    graph.add_task(SubTask("adversarial_review", "write", "adversarial_review"))
    graph.mark_running("adversarial_review")
    graph.mark_done("adversarial_review", {})
    report, _ = await agent._prepare_report(
        query="EViT",
        graph=graph,
        results={"adversarial_review": {"final_draft": "original"}},
        options=RunOptions(quality_policy="strict_auto_v2"),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    assert seen == ["original", "original"]
    assert report["survey"] == "original"
    assert agent._evaluate.await_count == 1
    assert not report["delivery"]["publishable"]
