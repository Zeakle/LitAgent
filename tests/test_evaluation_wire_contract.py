"""Verify native evaluation formats at the SDK boundary and host safety gates."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_batched_evaluation import BatchLLM, materials

from litagent.body_quality import ROLES
from litagent.config import EvalConfig
from litagent.eval.batched import (
    batch_payload,
    batch_prompt,
    plan_batches,
    request_size,
)
from litagent.eval.scoped import (
    ScopedEvaluationError,
    evaluate_scoped,
    evaluation_response_format,
)
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.safety.budget import CostBudget, strict_request_reservation


def wire_identity(messages, response_format):
    """Reconstruct the exact native request binding independently of the host."""
    return hashlib.sha256(
        json.dumps([messages, response_format], ensure_ascii=False).encode()
    ).hexdigest()


class AnalysisTransport(BatchLLM):
    """Replace only provider transport, leaving evaluators and accounting intact."""

    def __init__(self, fault=None):
        """Select a single scientific or protocol fault for a bounded response."""
        super().__init__()
        self.fault = fault

    async def chat(self, messages, **kwargs):
        """Generate complete negative or malformed decisions for actual host IDs."""
        response = await super().chat(messages, **kwargs)
        payload = self.calls[-1]
        if (
            self.fault == "incomplete_inventory"
            and "allowed_citation_aliases" in payload
        ):
            fixture = json.loads(
                (
                    Path(__file__).parent
                    / "fixtures/analysis-inventory-incomplete-real.json"
                ).read_text()
            )
            response.content = fixture["response"]
            response.finish_reason = fixture["finish_reason"]
            return response
        result = json.loads(response.content)
        if "cited_materials" in payload:
            decision = result["decisions"][0]
            if self.fault in {"unsupported", "unverifiable", "uncited"}:
                decision.update(
                    status=(
                        "unverifiable"
                        if self.fault == "unverifiable"
                        else "unsupported"
                    ),
                    reason_code=(
                        "uncited_claim"
                        if self.fault == "uncited"
                        else (
                            "unverifiable"
                            if self.fault == "unverifiable"
                            else "insufficient_support"
                        )
                    ),
                    support_basis=[],
                )
                for aspect in decision["aspect_checks"]:
                    aspect.update(supported=False, support_ids=[])
            elif self.fault == "omit":
                result["decisions"].pop()
            elif self.fault == "duplicate":
                result["decisions"][1] = copy.deepcopy(decision)
            elif self.fault == "unknown":
                decision["claim_id"] = "invented-claim"
            elif self.fault == "foreign_support":
                foreign = result["decisions"][1]
                decision["support_basis"] = copy.deepcopy(foreign["support_basis"])
                for aspect in decision["aspect_checks"]:
                    aspect["support_ids"] = [
                        s["support_id"] for s in foreign["support_basis"]
                    ]
        elif "analyses" in payload.get("schema", {}).get("properties", {}):
            result.update(
                limitations={
                    "adequate": False,
                    "claim_ids": [],
                    "explanation": "No limitations in this plumbing fixture.",
                },
                analyses=[
                    {
                        "section_key": key,
                        "adequate": False,
                        "claim_ids": [],
                        "explanation": "This fixture tests scientific transport only.",
                    }
                    for key in ROLES
                ],
            )
        response.content = json.dumps(result)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "unsupported",
        "unverifiable",
        "uncited",
        "foreign_support",
        "omit",
        "duplicate",
        "unknown",
        "incomplete_inventory",
    ],
)
async def test_actual_sdk_receives_all_native_formats_without_promoting_bad_science(
    monkeypatch, fault
):
    """Exercise evaluate_scoped, the real client and SDK kwargs with zero network."""
    survey, context = materials(2)
    context["body_quality"] = "analysis_v1"
    if fault == "uncited":
        survey = survey.replace("[E:a0]", "")
    snapshots, wire = [], []
    context["evaluation_checkpoint"] = lambda p: snapshots.append(copy.deepcopy(p))
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    llm = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    transport = AnalysisTransport(fault)

    async def sdk_create(**kwargs):
        """Intercept SDK creation after actual budget and format translation."""
        wire.append(copy.deepcopy(kwargs))
        options = {k: v for k, v in kwargs.items() if k != "messages"}
        response = await transport.chat(kwargs["messages"], **options)
        assert kwargs["response_format"] == evaluation_response_format(
            transport.calls[-1]
        )
        validator = Draft202012Validator(
            kwargs["response_format"]["json_schema"]["schema"]
        )
        if fault not in {"omit", "duplicate", "unknown", "incomplete_inventory"}:
            validator.validate(json.loads(response.content))
        return SimpleNamespace(
            model="offline",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response.content, tool_calls=None),
                    finish_reason=response.finish_reason
                    or ("" if fault == "incomplete_inventory" else "stop"),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    monkeypatch.setattr(
        llm._client.chat.completions, "create", AsyncMock(side_effect=sdk_create)
    )
    token = set_task_id("evaluation.initial")
    try:
        if fault == "incomplete_inventory":
            with pytest.raises(ScopedEvaluationError) as error:
                await evaluate_scoped(
                    llm=llm, survey=survey, context=context, config=EvalConfig()
                )
            assert str(error.value) == "evaluation_inventory_json_invalid"
            detail = error.value.details
            assert not detail["coverage"]["complete"] and "score" not in detail
            assert detail["inventory_response"]["trailing_whitespace_chars"] == 510
            assert detail["inventory_response"]["json_error_offset"] == 1751
            assert detail["stages"]["requests"][0]["finish_reason"] == ""
            assert len(wire) == 1
        elif fault in {"omit", "duplicate", "unknown"}:
            with pytest.raises(ScopedEvaluationError) as error:
                await evaluate_scoped(
                    llm=llm, survey=survey, context=context, config=EvalConfig()
                )
            assert not error.value.details["coverage"]["complete"]
            batch = error.value.details["stages"]["batches"][0]
            assert batch["status"] == "invalid_response"
            assert batch["reason_code"] == "evaluation_decision_coverage_invalid"
            assert batch["usage"]["usage_available"] is True
            assert len(wire) == 2
        else:
            result = await evaluate_scoped(
                llm=llm, survey=survey, context=context, config=EvalConfig()
            )
            if fault in {"unverifiable", "foreign_support"}:
                assert not result["coverage"]["complete"]
                assert result["score"] is None and result["unverified_count"] == 1
            else:
                assert result["coverage"]["complete"]
                assert result["score"] == (1.0 if fault is None else 0.5)
            # Complete scientific plumbing does not satisfy absent academic content.
            assert result["analysis_check"]["status"] == "failed"
            assert len(wire) == 3
    finally:
        reset_task_id(token)
        await llm.close()
    assert budget.snapshot()["requests_sent"] == len(wire)
    assert budget.snapshot()["used_tokens"] == len(wire) * 20
    assert not budget.snapshot()["usage_uncertain"]
    requests = snapshots[-1]["requests"]
    assert len(requests) == len(wire)
    for sent, saved in zip(wire, requests, strict=True):
        assert saved["input_sha256"] == wire_identity(
            sent["messages"], sent["response_format"]
        )
        assert saved["reserved_tokens"] == strict_request_reservation(
            sent["messages"], sent["max_tokens"], sent["response_format"]
        )
        assert sent["max_tokens"] == 16384
    assert [c["response_format"]["json_schema"]["name"] for c in wire] == [
        "host_flat_fragment_inventory_v1",
        "host_batch_assessment_v1",
        "host_global_completeness_v1",
    ][: len(wire)]


def planning_payload(count=2, source_chars=12000):
    """Construct independently cited complete originals for real batch planning."""
    return {
        "body_analysis_contract": "body-analysis-v1",
        "assertions": [
            {
                "claim_id": f"c{i}",
                "source_unit_id": f"u{i}",
                "evidence": [{"evidence_alias": f"e{i}"}],
            }
            for i in range(count)
        ],
        "source_units": [
            {"source_unit_id": f"u{i}", "text": "Report claim."} for i in range(count)
        ],
        "report_structure": [],
        "cited_materials": {
            f"e{i}": {"original_support_ref": f"s{i}"} for i in range(count)
        },
        "original_supports": {
            f"s{i}": {"text": "x" * source_chars} for i in range(count)
        },
        "support_passages": {
            f"p{i}": [f"e{i}", f"s{i}", 0, source_chars] for i in range(count)
        },
    }


def test_batch_planner_counts_native_schema_and_rebuilds_each_partition():
    """Split a request that fits messages alone but exceeds the real wire limit."""
    payload = planning_payload()
    combined = batch_payload(payload, payload["assertions"])
    full = request_size(batch_prompt(combined), combined, 16384)
    native_size = len(
        json.dumps(evaluation_response_format(combined), ensure_ascii=False).encode()
    )
    config = EvalConfig(context_tokens=int((full - native_size // 2) / 0.9))
    limit = int(config.context_tokens * 0.9)
    assert full - native_size < limit < full
    batches = plan_batches(payload, config)
    assert len(batches) == 2
    for index, batch in enumerate(batches):
        schema = batch["schema"]
        assert schema["$defs"]["PassageDecision"]["properties"]["claim_id"]["enum"] == [
            f"c{index}"
        ]
        assert schema["$defs"]["SupportSelection"]["properties"]["support_id"][
            "enum"
        ] == [f"p{index}"]
        assert schema["properties"]["decisions"]["minItems"] == 1
        assert len(batch["original_supports"][f"s{index}"]["text"]) == 12000
        assert request_size(batch_prompt(batch), batch, 16384) <= limit
    assert "schema" not in payload


@pytest.mark.parametrize("status", ["unsupported", "unverifiable"])
def test_no_material_schema_allows_explicit_negative_decisions(status):
    """Missing evidence cannot force a fabricated support selection."""
    payload = planning_payload(1)
    payload["assertions"][0]["evidence"] = []
    batch = batch_payload(payload, payload["assertions"])
    result = {
        "units": [
            {
                "source_unit_id": "u0",
                "complete": True,
                "explanation": "Faithful report extraction.",
            }
        ],
        "decisions": [
            {
                "claim_id": "c0",
                "correct_kind": "scientific",
                "atomic": True,
                "classification_valid": True,
                "source_faithful": True,
                "status": status,
                "reason_code": "uncited_claim",
                "explanation": "No cited evidence.",
                "support_basis": [],
                "aspect_checks": [
                    {
                        "aspect": "predicate",
                        "supported": False,
                        "support_ids": [],
                        "explanation": "Not established.",
                    }
                ],
            }
        ],
        "conflicts": [],
    }
    validator = Draft202012Validator(batch["schema"])
    validator.validate(result)
    result["decisions"][0]["support_basis"] = [{"support_id": "invented"}]
    assert list(validator.iter_errors(result))


def test_oversized_native_request_still_stops_without_truncation():
    """A single source that cannot fit is retained for a reconstructable refusal."""
    payload = planning_payload(1, 120000)
    rejected = []
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        plan_batches(
            payload, EvalConfig(), on_rejected=lambda *args: rejected.append(args)
        )
    assert len(rejected) == 1
    assert len(rejected[0][1]["original_supports"]["s0"]["text"]) == 120000
    assert evaluation_response_format(rejected[0][1])["type"] == "json_schema"
