"""Exercise global report repair authority through actual frozen report consumers."""

import copy
import json

import pytest
import test_structured_repair_pipeline as structured
import test_writer_source_view_pipeline as writer_view
from jsonschema import Draft202012Validator
from test_evaluation_dimensions_pipeline import separated_peer
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import PROFILE, profile_flags
from litagent.eval.results import CONTRACT


class ReportIssueGateway(structured.LongScientificGateway):
    """Control provider output only; keep accounting, permissions and transactions real."""

    def __init__(self, case, mode):
        super().__init__(case, 0, reviewer=True)
        previous = self.evaluator.chat
        self.global_outputs = []

        async def chat(messages, **options):
            response = await previous(messages, **options)
            payload = self.evaluator.base.calls[-1]
            if "global_output_contract" not in payload:
                return response
            value = json.loads(response.content)
            if not self.global_outputs or mode == "persistent":
                # The first scientific assertion is supported. Its report placement
                # can still have an independent, source-bound organization defect.
                anchor = value["coverage"][0]["claim_ids"][0]
                if mode == "positive_duplicate":
                    value["issues"] = [
                        dict(
                            code="core_coverage",
                            anchor_ids=[anchor],
                            explanation="The report answers its question.",
                        )
                    ]
                else:
                    value["issues"] = [
                        dict(
                            code="repetition",
                            anchor_ids=[anchor],
                            explanation="This introductory finding is repeated in the same section without advancing the argument.",
                        )
                    ]
            schema = options["response_format"]["json_schema"]["schema"]
            assert Draft202012Validator(schema).is_valid(value) is (
                mode != "positive_duplicate"
            )
            self.global_outputs.append(copy.deepcopy(value))
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["resolved", "persistent", "positive_duplicate"])
async def test_real_report_issues_share_bounded_repairs_and_reassessment(
    source_case, tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(
        writer_view,
        "source_flags",
        lambda *_: {**profile_flags(PROFILE), "evaluation_result_contract": CONTRACT},
    )
    monkeypatch.setattr(writer_view, "conform_evaluation_replies", separated_peer)
    gateway = ReportIssueGateway(source_case, mode)
    monkeypatch.setattr(structured, "LongScientificGateway", lambda *_a, **_k: gateway)
    result = await writer_view.run_scientific_case(
        source_case, tmp_path / "run", monkeypatch
    )
    history = result.report["metadata"]["evaluation_history"]
    initial = history["initial_evaluation"]["faithfulness"]["details"]
    assert gateway.global_outputs
    if mode == "positive_duplicate":
        assert not result.report["delivery"]["publishable"]
        assert not result.journal["candidates"]
        assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == 0
        return
    supported_ids = {
        c["claim_id"] for c in initial["assertions"] if c["status"] == "supported"
    }
    assert any(
        d["claim_id"] in supported_ids and d.get("diagnostic_owner") == "report"
        for d in initial["unsupported_claims"]
    )
    assert result.ledger["plan"]["repair_limit"] == 2
    attempts = result.report["metadata"]["targeted_repair_state"]["attempts"]
    assert 1 <= len(attempts) <= 2
    assert all("purpose" not in a for a in attempts)
    assert result.report["delivery"]["publishable"] is (mode == "resolved")
    if mode == "resolved":
        assert result.journal["candidates"][-1]["committed"]
        final = result.report["evaluation"]["faithfulness"]
        assert final["score"] == 1.0 and final["details"]["coverage"]["complete"]
        assert (
            final["details"]["unsupported_count"]
            == final["details"]["unverified_count"]
            == 0
        )
    else:
        assert not any(c["committed"] for c in result.journal["candidates"])
        assert "FLOPs by 44%" in result.report["survey"]
