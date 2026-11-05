"""Run permitted omissions through recovery, prose repair and full assessment."""

import copy
import json

import pytest
import test_structured_repair_pipeline as structured
import test_writer_source_view_pipeline as writer_view
from jsonschema import Draft202012Validator
from test_evaluation_dimensions_pipeline import separated_peer
from test_evaluation_stability_pipeline import read_inventory_catalog_at_provider
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import PROFILE, profile_flags
from litagent.eval.results import CONTRACT


class OmissionGateway(structured.LongScientificGateway):
    """Control only HTTP results, leaving source and repair consumers unchanged."""

    def __init__(self, case, mode):
        super().__init__(case, 0)
        self.global_replies = []
        previous = self.evaluator.chat

        async def chat(messages, **options):
            response = await previous(messages, **options)
            payload = self.evaluator.base.calls[-1]
            if "global_output_contract" not in payload:
                return response
            value = json.loads(response.content)
            if not self.global_replies or mode == "persistent":
                unit = value["units"][0]
                unit["complete"] = False
                value["issues"] = [
                    {
                        "code": "missing_assertion",
                        "anchor_ids": [unit["source_unit_id"]],
                        "explanation": "The original introductory premise is not fully represented.",
                    }
                ]
                if mode == "mixed":
                    value["issues"][0]["anchor_ids"].append(
                        payload["assertions"][0]["claim_id"]
                    )
            schema = options["response_format"]["json_schema"]["schema"]
            assert Draft202012Validator(schema).is_valid(value) is (mode != "mixed")
            self.global_replies.append(copy.deepcopy(value))
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        read_inventory_catalog_at_provider(self)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["resolved", "persistent", "mixed"])
async def test_actual_omission_keeps_recovery_and_prose_slots(
    source_case, tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(
        writer_view,
        "source_flags",
        lambda *_: {**profile_flags(PROFILE), "evaluation_result_contract": CONTRACT},
    )
    monkeypatch.setattr(writer_view, "conform_evaluation_replies", separated_peer)
    gateway = OmissionGateway(source_case, mode)
    monkeypatch.setattr(structured, "LongScientificGateway", lambda *_a, **_k: gateway)
    result = await writer_view.run_scientific_case(source_case, tmp_path, monkeypatch)
    attempts = result.report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == {"resolved": 2, "persistent": 1, "mixed": 0}[mode]
    assert sum(
        a.get("purpose") == "bounded-inventory-recovery-v2" for a in attempts
    ) == (mode != "mixed")
    assert result.report["delivery"]["publishable"] is (mode == "resolved")
    assert result.ledger["plan"]["repair_limit"] == 2
    assert gateway.global_replies
    actual_global = [
        entry["payload"]
        for entry in gateway.sent
        if "global_output_contract" in entry["payload"]
    ]
    assert actual_global
    assert all(
        p.get("global_input_contract") == "unit-local-global-input-v1"
        for p in actual_global
    )
    assert all(
        "assertions_by_kind" not in p and "source_units" not in p for p in actual_global
    )
    if mode == "resolved":
        assert len(gateway.global_replies) == 3
        assert result.journal["candidates"][-1]["committed"]
        assert result.report["evaluation"]["faithfulness"]["score"] == 1.0
        assert "FLOPs by 44%" not in result.report["survey"]
    else:
        assert not result.journal["candidates"]
        assert result.report["evaluation"]["faithfulness"]["score"] is None
        assert "FLOPs by 44%" in result.report["survey"]
