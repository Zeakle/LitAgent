"""Exercise bound response rules through the real frozen workflow and SDK.

The provider replies are controlled transport fixtures, not scientific gold.
Application planning, synthesis, review, evaluation and accounting stay real.
"""

import copy
import json

import pytest
import test_semantic_conflict_pipeline as pipeline
from jsonschema import Draft202012Validator
from test_flat_writer_pipeline import FlatGateway
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.eval.batched import (
    EVALUATION_RESPONSE_CONTRACT,
    response_contract_identity,
    selected_response_contract,
)
from litagent.quality import digest
from litagent.runner import LitAgent
from litagent.safety.planned_budget import planned_backend

RESPONSE_FIELDS = (
    "evaluation_response_contract",
    "evaluation_response_identity",
)


def decision_branches(schema):
    """Resolve actual provider decision definitions without changing the schema."""
    item = schema["properties"]["decisions"]["items"]
    references = item.get("anyOf", [item])
    return [
        (
            schema["$defs"][branch["$ref"].rsplit("/", 1)[-1]]
            if "$ref" in branch
            else branch
        )
        for branch in references
    ]


class BoundResponseGateway(FlatGateway):
    """Change only HTTP replies while recording the unchanged outbound schema."""

    def __init__(self, case, *, fault=None):
        super().__init__(case, declined=fault is not None)
        self.response_fault = fault
        self.replies = []
        self.injected = []
        self.repair_count = 0
        self.assessment_schemas = []
        original = self.evaluator.chat

        async def chat(messages, **options):
            response = await original(messages, **options)
            payload = self.evaluator.base.calls[-1]
            schema = options["response_format"]["json_schema"]["schema"]
            value = json.loads(response.content)
            phase = "inventory"
            if "cited_materials" in payload:
                phase = "assessment"
                self.assessment_schemas.append(copy.deepcopy(schema))
                branches = decision_branches(schema)
                if all(
                    "classification_valid" not in branch["properties"]
                    for branch in branches
                ):
                    # The older provider fixture looks for PassageDecision by
                    # name. Selected schemas now prune that unused definition.
                    for decision in value["decisions"]:
                        decision.pop("classification_valid", None)
                if not self.injected and fault == "ownership":
                    claims = {
                        claim["claim_id"]: claim for claim in payload["assertions"]
                    }
                    for decision in value["decisions"]:
                        claim = claims[decision["claim_id"]]
                        aliases = {
                            evidence["evidence_alias"] for evidence in claim["evidence"]
                        }
                        foreign = next(
                            (
                                support_id
                                for support_id, passage in payload[
                                    "support_passages"
                                ].items()
                                if passage[0] not in aliases
                            ),
                            None,
                        )
                        if foreign is None:
                            continue
                        decision["support_basis"] = [{"support_id": foreign}]
                        for aspect in decision["aspect_checks"]:
                            aspect["support_ids"] = [foreign]
                        self.injected.append(copy.deepcopy(decision))
                        break
                elif not self.injected and fault == "condition":
                    claims = {
                        claim["claim_id"]: claim for claim in payload["assertions"]
                    }
                    for decision in value["decisions"]:
                        if (
                            "On DeiT-B,"
                            not in claims[decision["claim_id"]]["claim_text"]
                        ):
                            continue
                        conditions = [
                            aspect
                            for aspect in decision["aspect_checks"]
                            if aspect["aspect"] == "conditions"
                        ]
                        assert len(conditions) == 1
                        decision.update(
                            status="unsupported",
                            reason_code="insufficient_support",
                            explanation=(
                                "Controlled condition mismatch: the supplied source "
                                "states DeiT-S, while this claim names DeiT-B."
                            ),
                        )
                        conditions[0].update(
                            supported=False,
                            support_ids=[],
                            explanation="The supplied condition is not established.",
                        )
                        self.injected.append(copy.deepcopy(decision))
                        break
            elif "global_output_contract" in payload:
                phase = "global"
                if fault == "global_completeness":
                    unit = value["units"][0]
                    assert unit["complete"] is True
                    issue = {
                        "code": "missing_assertion",
                        "anchor_ids": [unit["source_unit_id"]],
                        "explanation": (
                            "The controlled inventory is complete; no extra "
                            "omissions are present."
                        ),
                    }
                    value["issues"] = [issue]
                    self.injected.append(copy.deepcopy(issue))
            validator = Draft202012Validator(schema)
            hostile_basis = (
                phase == "assessment"
                and fault == "ownership"
                and self.injected
                and self.injected[-1] in value["decisions"]
            )
            if hostile_basis:
                # A hostile provider can ignore native constraints. The host
                # ownership gate must independently keep the response unsafe.
                assert not validator.is_valid(value)
            else:
                validator.validate(value)
            response.content = json.dumps(value)
            self.replies.append(
                {
                    "phase": phase,
                    "raw": response.content,
                    "value": copy.deepcopy(value),
                }
            )
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        """Decline permitted prose repair instead of fabricating a passing edit."""
        if "targets" in payload:
            self.repair_count += 1
        return super().answer(payload, native_schema)


async def run_bound_case(
    case, tmp_path, monkeypatch, *, selected=True, fault=None, forged_context=False
):
    """Reuse complete frozen planning, Synthesis and runner evaluation consumers."""
    flags = dict(pipeline.FLAGS)
    if selected:
        flags["evaluation_response_contract"] = EVALUATION_RESPONSE_CONTRACT
    monkeypatch.setattr(pipeline, "FLAGS", flags)
    gateway = BoundResponseGateway(case, fault=fault)
    monkeypatch.setattr(pipeline, "ConflictGateway", lambda *_args, **_kw: gateway)
    if forged_context:
        original = LitAgent._build_eval_context

        def build_context(agent, **kwargs):
            context = original(agent, **kwargs)
            owner = planned_backend(agent._cost_budget)
            assert not selected_response_contract(owner.effective_envelope)
            forged = copy.deepcopy(owner.effective_envelope)
            forged.update(
                evaluation_response_contract=EVALUATION_RESPONSE_CONTRACT,
                evaluation_response_identity=response_contract_identity(),
            )
            context.update(
                evaluation_response_contract=EVALUATION_RESPONSE_CONTRACT,
                evaluation_response_identity=response_contract_identity(),
                writer_envelope=forged,
            )
            return context

        monkeypatch.setattr(LitAgent, "_build_eval_context", build_context)
    report, _, snapshot = await pipeline.run_conflict_case(case, tmp_path, monkeypatch)
    state, journal, ledger = products(tmp_path)
    assert snapshot == ledger
    assert ledger["requests_sent"] == ledger["call_count"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
    envelope = ledger["plan"]["envelope"]
    assert selected_response_contract(envelope) is selected
    assert envelope["identity"] == digest(
        {key: value for key, value in envelope.items() if key != "identity"}
    )
    assert ledger["plan"]["maximum_tokens"] == 4500000
    assert ledger["plan"]["maximum_requests"] == 64
    assert ledger["plan"]["repair_limit"] == 2
    if selected:
        assert envelope["evaluation_response_contract"] == EVALUATION_RESPONSE_CONTRACT
        assert envelope["evaluation_response_identity"] == response_contract_identity()
    else:
        assert all(field not in envelope for field in RESPONSE_FIELDS)
    return report, gateway, state, journal, ledger


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
async def test_frozen_pipeline_retains_positive_science_and_array_response_shape(
    case, tmp_path, monkeypatch, selected
):
    """Native permission enums add no scientific support to a controlled positive."""
    report, gateway, _, journal, _ = await run_bound_case(
        case, tmp_path, monkeypatch, selected=selected
    )
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"]
    assert details["stages"]["global_check"] == "completed"
    assert details["supported_count"] == details["factual_claim_count"]
    assert details["unsupported_count"] == details["unverified_count"] == 0
    assert report["delivery"]["publishable"] is True
    assert gateway.repair_count == 0 and journal["candidates"] == []
    assert sent_with(gateway, "cited_materials")
    assert len(sent_with(gateway, "global_output_contract")) == 1
    for schema in gateway.assessment_schemas:
        decisions = schema["properties"]["decisions"]
        assert decisions["type"] == "array"
        assert ("anyOf" in decisions["items"]) is selected
    for reply in gateway.replies:
        if reply["phase"] == "assessment":
            assert isinstance(reply["value"]["decisions"], list)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["condition", "ownership"])
async def test_native_bound_schema_never_promotes_wrong_conditions_or_foreign_basis(
    case, tmp_path, monkeypatch, fault
):
    """Keep semantic rejection independent of matching provider containers."""
    if fault == "condition":
        case.body["methods"][0]["sentences"][0].update(
            text="methods: On DeiT-B, EViT reduces computation.",
            fact_card_refs=[case.by_eid["a"]],
        )
    else:
        case.body["taxonomy"][0]["sentences"][0].update(
            text="taxonomy: On DeiT-S, EViT reduces computation.",
            fact_card_refs=[case.by_eid["a"]],
        )
    report, gateway, _, journal, _ = await run_bound_case(
        case, tmp_path, monkeypatch, fault=fault
    )
    assert gateway.injected, "The transport fault must hit a real assigned claim."
    assert report["delivery"]["publishable"] is False
    assert not any(candidate["committed"] for candidate in journal["candidates"])
    details = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    if fault == "condition":
        negative = [
            claim
            for claim in details["assertions"]
            if "On DeiT-B," in claim["claim_text"]
        ]
        assert negative and all(claim["status"] != "supported" for claim in negative)
        assert "On DeiT-B," in report["survey"]
    else:
        assert gateway.repair_count == 0
        assert details.get("coverage", {}).get("complete") is False


@pytest.mark.asyncio
async def test_global_missing_with_complete_true_keeps_raw_and_grants_no_repair(
    case, tmp_path, monkeypatch
):
    """A contradictory global decision is not repaired into scientific acceptance."""
    report, gateway, state, journal, _ = await run_bound_case(
        case, tmp_path, monkeypatch, fault="global_completeness"
    )
    metric = report["evaluation"]["faithfulness"]
    assert metric["skipped"]
    assert metric["details"]["skipped_reason"] == (
        "evaluation_global_completeness_conflict"
    )
    assert report["delivery"]["publishable"] is False
    assert len(sent_with(gateway, "global_output_contract")) == 1
    assert gateway.repair_count == 0 and not sent_with(gateway, "targets")
    assert state["state"]["attempts"] == [] and journal["candidates"] == []
    [original] = [reply for reply in gateway.replies if reply["phase"] == "global"]
    [issue] = original["value"]["issues"]
    assert issue == gateway.injected[0]
    unit = next(
        row
        for row in original["value"]["units"]
        if row["source_unit_id"] == issue["anchor_ids"][0]
    )
    assert unit["complete"] is True
    progress = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "evaluation").glob("*/progress.json")
    ]
    raw_requests = [
        request
        for saved in progress
        for request in saved["progress"]["requests"]
        if request["phase"] == "global"
    ]
    assert len(raw_requests) == 1
    assert raw_requests[0]["response"] == original["raw"]
    assert raw_requests[0]["status"] == "response_received"


@pytest.mark.asyncio
async def test_external_context_cannot_select_or_widen_locked_response_contract(
    case, tmp_path, monkeypatch
):
    """A complete-looking caller envelope cannot opt an old owner into new rules."""
    report, gateway, _, _, ledger = await run_bound_case(
        case, tmp_path, monkeypatch, selected=False, forged_context=True
    )
    assert all(field not in ledger["plan"]["envelope"] for field in RESPONSE_FIELDS)
    assert report["evaluation"]["faithfulness"]["details"]["coverage"]["complete"]
    assert gateway.assessment_schemas
    for schema in gateway.assessment_schemas:
        assert "anyOf" not in schema["properties"]["decisions"]["items"]
        assert not any(
            name.startswith("ClaimPermissionGroup") for name in schema["$defs"]
        )
