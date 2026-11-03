"""Exercise bounded scientific repair through SDK, workers and full re-evaluation."""

import json

import pytest
import test_located_evaluation as located
from test_flat_writer_pipeline import FlatGateway
from test_writer_v2 import case  # noqa: F401


class ScientificRepairGateway(FlatGateway):
    """Replace external responses only; keep all application repair consumers real."""

    def __init__(self, case, failing_candidates):
        super().__init__(case)
        self.failing_candidates = failing_candidates
        self.repair_count = 0
        self.inventories = 0
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "allowed_citation_aliases" in payload:
                self.inventories += 1
            if "cited_materials" in payload:
                value = json.loads(response.content)
                by_id = {c["claim_id"]: c for c in payload["assertions"]}
                for d in value["decisions"]:
                    if "FLOPs by" in by_id[d["claim_id"]]["claim_text"]:
                        d.update(
                            status="unsupported",
                            reason_code="insufficient_support",
                            explanation="The actual relation-network source reports no FLOPs reduction.",
                            support_basis=[],
                        )
                        for a in d["aspect_checks"]:
                            a.update(
                                supported=False,
                                support_ids=[],
                                explanation="No actual cited numerical result.",
                            )
                response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if payload.get("requested_body_section") == "methods":
            value["units"][0]["text"] = "Relation networks reduce FLOPs by 44%."
        if "targets" in payload:
            self.repair_count += 1
            for patch in value["patches"]:
                patch["text"] = (
                    f"Relation networks reduce FLOPs by {44 + self.repair_count}%."
                    if self.repair_count <= self.failing_candidates
                    else "In the cited few-shot setting, Relation networks learn a comparison function."
                )
                patch["fact_card_refs"] = [self.case.by_eid["b"]]
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
@pytest.mark.parametrize("classification_review", [False, True])
async def test_first_or_second_candidate_needs_full_assessment_and_never_a_third_call(
    case, tmp_path, monkeypatch, failing_candidates, classification_review
):
    gateway = ScientificRepairGateway(case, failing_candidates)
    monkeypatch.setattr(located, "FindingsGateway", lambda *_: gateway)
    original_methodology = located.build_methodology

    def recorded_methodology(query, upstream, **kwargs):
        # This independent fixture includes host discovery records. The negative
        # findings fixture intentionally lacks them and cannot prove delivery.
        tasks = {
            "__search_queries__": {"search_fixture": "relation networks"},
            "search_fixture": list(case.ledger.values()),
            "dedup": list(case.ledger.values()),
            "relevance_gate": list(case.ledger.values()),
        }
        kwargs["source_outcomes"] = [
            {
                "task_id": "search_fixture",
                "source": "offline-public-fixture",
                "query": "relation networks",
                "status": "completed",
                "result_count": len(case.ledger),
            }
        ]
        return original_methodology(query, tasks, **kwargs)

    monkeypatch.setattr(located, "build_methodology", recorded_methodology)
    report, _ = await located.run_case(
        case,
        tmp_path,
        monkeypatch,
        "scientific-repair",
        classification_review=classification_review,
    )
    history = report["metadata"]["evaluation_history"]["initial_evaluation"]
    initial = history["faithfulness"]["details"]
    assert initial["coverage"]["complete"]
    assert initial["inventory_check"]["status"] == "passed"
    assert initial["unsupported_count"] == 1
    expected = min(2, failing_candidates + 1)
    assert gateway.repair_count == expected
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    assert gateway.inventories == expected + 1
    repairs = [row["payload"] for row in gateway.sent if "targets" in row["payload"]]
    # Failed candidates never become the starting point of the next transaction.
    for payload in repairs:
        current = json.dumps(payload["current_body"])
        assert "FLOPs by 44%" in current
        assert "FLOPs by 45%" not in current
    assert (
        len(
            [
                row
                for row in gateway.sent
                if "body_units" in row["payload"] and "targets" not in row["payload"]
            ]
        )
        == expected + 1
    )
    final = report["evaluation"]["faithfulness"]["details"]
    assert ("classification_review_contract" in final) is classification_review
    if failing_candidates < 2:
        assert "FLOPs by" not in report["survey"]
        assert final["coverage"]["complete"]
        assert final["supported_count"] == final["factual_claim_count"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert final["strict_acceptance"]["status"] == "passed"
        assert report["delivery"]["publishable"]
    else:
        assert "FLOPs by 44%" in report["survey"]
        assert "FLOPs by 45%" not in report["survey"]
        assert not report["delivery"]["publishable"]
