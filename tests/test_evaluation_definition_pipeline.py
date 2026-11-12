"""Replay recorded definition conflicts without promoting scientific verdicts."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import test_evaluation_stability_pipeline as stability
import test_guidance_wire_pipeline as guidance_pipeline
import test_mixed_inventory_pipeline as mixed
import test_structured_repair_pipeline as structured
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.eval.batched import (
    _validate_decision_consistency,
    global_review_type,
    normalize_global_findings,
    validate_global_findings,
)
from litagent.eval.scoped import (
    PassageDecision,
    ScopedEvaluationError,
    evaluation_definition_prompt,
    validate_assessment_decisions,
)
from litagent.writing_flat import restore_global_identities

FIXTURE_PATH = Path(__file__).parent / "fixtures/evaluation-045-definitions.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
REVISION = "host-support-receipts-v2"
WIRE_REVISION = "host-unit-ref-guidance-v1"


def original_wire_claim(claim_id):
    """Resolve the recorded field table through its actual fragment ownership."""
    record = FIXTURE["nerf_failed_batch"]
    payload = record["payload"]
    wire = next(
        dict(zip(payload["assertion_columns"], row, strict=True))
        for row in payload["assertions"]
        if row[0] == claim_id
    )
    fragments = {
        row[0]: (unit["source_unit_id"], row[2])
        for unit in payload["source_units"]
        for row in unit["source_fragments"]
        if row[0] is not None
    }
    parent, inherited = fragments[wire["source_fragment_id"]]
    aliases = (
        inherited if wire["evidence_aliases"] is None else wire["evidence_aliases"]
    )
    claim = copy.deepcopy(
        next(c for c in record["original_claims"] if c["claim_id"] == claim_id)
    )
    assert claim["source_unit_id"] == parent
    claim.update(
        required_support_aspects=wire["required_support_aspects"],
        evidence=[{"evidence_alias": alias} for alias in aliases],
    )
    return claim, wire


def original_global_view():
    """Expand recorded coordinates only; keep all original claims and units."""
    payload = copy.deepcopy(FIXTURE["few_shot_first_global"]["payload"])
    parents = {
        fragment["source_fragment_id"]: uid
        for uid, fragments in payload["source_units"].items()
        for fragment in fragments
    }
    payload["source_units"] = [
        {"source_unit_id": uid, "source_fragments": fragments}
        for uid, fragments in payload["source_units"].items()
    ]
    payload["assertions"] = [
        dict(zip(payload["assertion_columns"], row, strict=True))
        for row in payload["assertions"]
    ]
    for claim in payload["assertions"]:
        claim["source_unit_id"] = parents[claim["source_fragment_id"]]
        if claim["kind"] is None:
            claim["kind"] = payload["assertion_kind_default"]
    return payload


def test_recorded_fixture_is_portable_public_data_and_not_a_scientific_gold():
    assert FIXTURE["scientific_gold_standard"] is False
    forbidden = {
        "provider_receipt",
        "headers",
        "authorization",
        "api_key",
        "operation_id",
        "session_id",
        "run_id",
        "trace_id",
        "request_id",
        "budget_snapshot",
    }

    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            assert "/Users/" not in value
            assert "frozen-writer-three-domain-" not in value

    check(FIXTURE)
    for key in ("nerf_failed_batch", "few_shot_first_global"):
        record = FIXTURE[key]
        assert (
            hashlib.sha256(record["raw_response"].encode()).hexdigest()
            == record["raw_response_sha256"]
        )
    payload = FIXTURE["nerf_failed_batch"]["payload"]
    assert len(payload["assertions"]) == 16
    assert len(payload["support_passages"]) == 13
    assert {"p4", "p10"} <= payload["support_passages"].keys()
    assert len(FIXTURE["few_shot_first_global"]["payload"]["assertions"]) == 157


@pytest.mark.parametrize("claim_id", ["claim_87", "claim_89"])
def test_original_nerf_positive_diagnostics_with_negative_status_still_reject(claim_id):
    record = FIXTURE["nerf_failed_batch"]
    before = copy.deepcopy(record)
    raw = next(
        d
        for d in json.loads(record["raw_response"])["decisions"]
        if d["claim_id"] == claim_id
    )
    claim, wire = original_wire_claim(claim_id)
    assert wire["evidence_aliases"] is None
    assert set(wire["allowed_support_ids"]) == {"p4", "p10"}
    decision = PassageDecision.model_validate(raw)
    assert decision.status == "unsupported"
    assert decision.reason_code == "attribution_mismatch"
    assert (
        decision.atomic and decision.classification_valid and decision.source_faithful
    )
    assert all(aspect.supported for aspect in decision.aspect_checks)
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_decision_consistency_invalid"
    ) as error:
        _validate_decision_consistency(claim, decision, record["payload"])
    [diagnostic] = error.value.details["decision_inconsistencies"]
    assert (
        diagnostic["subreason"] == "negative_status_with_complete_positive_diagnostics"
    )
    assert diagnostic["original_decision"] == raw
    assert record == before


def test_original_first_global_keeps_all_four_inventory_findings():
    record = FIXTURE["few_shot_first_global"]
    before = copy.deepcopy(record)
    payload = original_global_view()
    raw = global_review_type(payload).model_validate_json(record["raw_response"])
    # Keep the public coordinate namespace, then use the real response-ID
    # restoration step before canonical anchor ownership validation.
    restored = restore_global_identities(raw, payload)
    result = normalize_global_findings(restored, payload)
    validate_global_findings(result, payload)
    assert len(result.issues) == 4
    assert Counter(issue.code for issue in result.issues) == {
        "non_atomic": 2,
        "missing_assertion": 2,
    }
    assert all(issue.owner == "inventory" for issue in result.issues)
    assert [i.explanation for i in result.issues] == [i.explanation for i in raw.issues]
    assert len(payload["source_units"]) == 33
    assert len(payload["assertions"]) == 157
    assert record == before


@pytest.mark.parametrize("index", range(6))
def test_persistent_atomic_false_remains_unverified_without_changing_original_kind(
    index,
):
    example = FIXTURE["few_shot_atomic_lineage"][index]
    before = copy.deepcopy(example)
    claims = [row["claim"] for row in example["rounds"]]
    assert len({c["claim_text"] for c in claims}) == 1
    assert len({c["source_fragment_id"] for c in claims}) == 1
    for row in example["rounds"]:
        claim = row["claim"]
        decision = PassageDecision.model_validate(row["canonical_batch_decision"])
        assert decision.atomic is False
        [observed] = validate_assessment_decisions(
            [claim],
            [decision],
            ledger={},
            aliases={},
            support_passages={},
            context={"quality_policy": "strict_auto_v2"},
            strict=True,
        )
        assert observed["kind"] == claim["kind"]
        assert observed["status"] == "unverifiable"
        assert observed["local_error"] == "evaluation_atomic_classification_unverified"
    observations = FIXTURE["few_shot_original_kind_observations"]
    assert [r["scientific_count"] for r in observations] == [146, 150, 148]
    assert all(not r["inventory_valid"] for r in observations)
    assert example == before


def select_definition_policy(monkeypatch):
    """Select only the new trusted revision while retaining actual consumers."""
    monkeypatch.setattr(
        stability,
        "SELECTION",
        {**stability.SELECTION, "support_validation_revision": REVISION},
    )
    monkeypatch.setattr(
        structured,
        "FLAGS",
        {
            **structured.FLAGS,
            "support_validation_revision": REVISION,
            "repair_guidance_wire_revision": WIRE_REVISION,
        },
    )
    structured.select_structured_policy(monkeypatch, module=mixed)


def assert_v2_actual_stages(gateway, envelope):
    """Read actual SDK input and native schema rather than a proposed prompt."""
    stages = set()
    for entry in gateway.sent:
        payload = entry["payload"]
        if "allowed_citation_aliases" in payload:
            stage = "inventory"
        elif "cited_materials" in payload:
            stage = "assessment"
        elif "global_output_contract" in payload:
            stage = "global"
        else:
            continue
        stages.add(stage)
        assert envelope["support_validation_revision"] == REVISION
        expected = evaluation_definition_prompt(
            stage,
            {
                **payload,
                "writer_envelope": envelope,
                **({"inventory_recovery": None} if stage == "inventory" else {}),
            },
        )
        request = entry["request"]
        system = request.get("input", request.get("messages"))[0]["content"]
        assert system.startswith(expected)
        if "input" in request:
            schema = request["text"]["format"]["schema"]
            assert request["max_output_tokens"] == 16384
            assert "max_tokens" not in request
            assert "max_completion_tokens" not in request
        else:
            schema = request["response_format"]["json_schema"]["schema"]
            # Evaluation supplies max_tokens to the real Chat SDK; the
            # alternative Chat spelling must not silently change its reserve.
            assert request["max_tokens"] == 16384
            assert "max_completion_tokens" not in request
            assert "max_output_tokens" not in request
        assert schema["description"].startswith("[atomic] ")
        assert "[classification] " in schema["description"]
        assert "[fidelity] " in schema["description"]
        assert request["thinking"] == {"type": "disabled"}
    assert stages == {"inventory", "assessment", "global"}


@pytest.mark.asyncio
@pytest.mark.parametrize("science_failure", [False, True])
async def test_v2_recovery_leaves_one_body_edit_and_requires_complete_reassessment(
    case, tmp_path, monkeypatch, science_failure
):
    select_definition_policy(monkeypatch)
    gateway = stability.StableGateway(case, science_failure=science_failure)
    report, _ = await stability.execute(case, tmp_path, monkeypatch, gateway)
    state, journal, ledger = products(tmp_path)
    history = report["metadata"]["evaluation_history"]
    initial = history["initial_evaluation"]["faithfulness"]["details"]
    recovered = history["attempts"][0]["candidate_evaluation"]["faithfulness"][
        "details"
    ]
    assert initial["inventory_check"]["status"] == "failed"
    assert recovered["inventory_check"]["status"] == "passed"
    assert initial["source_report_sha256"] == recovered["source_report_sha256"]
    negative = [c for c in recovered["assertions"] if "FLOPs by 44%" in c["claim_text"]]
    assert negative and all(c["status"] == "unsupported" for c in negative)
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["purpose"] == mixed.MIXED_CONTRACT
    assert gateway.recovery_requests == 1 and gateway.repair_count == 1
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    guidance_pipeline.assert_complete_model_wire(repairs[0], prior=False)
    assert len(sent_with(gateway, "global_output_contract")) == 3
    assert gateway.inventories == 3
    final = report["evaluation"]["faithfulness"]["details"]
    assert (
        final["coverage"]["complete"] and final["stages"]["global_check"] == "completed"
    )
    assert report["delivery"]["publishable"] is (not science_failure)
    if science_failure:
        assert "FLOPs by 44%" in report["survey"]
        assert not any(c["committed"] for c in journal["candidates"])
    else:
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert final["supported_count"] == final["factual_claim_count"]
    assert ledger["requests_sent"] == ledger["call_count"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]
    assert all(e["status"] == "settled" for e in ledger["entries"].values())
    assert state["inventory_recovery_0"]["feedback"]["preserved_fragments"]
    assert_v2_actual_stages(gateway, ledger["plan"]["envelope"])


@pytest.mark.asyncio
async def test_v2_two_inventory_recoveries_cannot_add_a_third_body_edit(
    case, tmp_path, monkeypatch
):
    select_definition_policy(monkeypatch)
    gateway = mixed.MixedInventoryGateway(case, "repeat")
    stability.read_inventory_catalog_at_provider(gateway)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    assert not report["delivery"]["publishable"]
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 2
    assert gateway.recovery_requests == 2 and gateway.repair_count == 0
    assert not sent_with(gateway, "targets")
    assert not journal["candidates"]
    assert ledger["requests_sent"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]


@pytest.mark.asyncio
async def test_v2_unknown_recovery_holds_usage_and_never_retries(
    case, tmp_path, monkeypatch
):
    select_definition_policy(monkeypatch)
    gateway = stability.StableGateway(case, unknown=True)
    report, _ = await stability.execute(case, tmp_path, monkeypatch, gateway)
    _, _, ledger = products(tmp_path)
    assert not report["delivery"]["publishable"]
    assert gateway.recovery_requests == 1
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert not sent_with(gateway, "targets")
    assert ledger["usage_uncertain"] and ledger["stop_reason"] == "strict_usage_unknown"
    pending = [e for e in ledger["entries"].values() if e.get("inventory_recovery")]
    assert len(pending) == 1 and pending[0]["actual_tokens"] is None
    assert pending[0]["status"] == "interrupted_no_replay"
    assert pending[0]["reserved_tokens"] > 0


class RecordedConflictGateway(stability.FlatGateway):
    """Synthesize a recorded conflict shape before the HTTP mock returns to SDK."""

    def __init__(self, case, raw):
        super().__init__(case)
        self.original = copy.deepcopy(raw)
        self.injected = None
        previous = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await previous(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            claim = next(
                c for c in payload["assertions"] if c["claim_text"].startswith("On D,")
            )
            assert set(claim["required_support_aspects"]) == {
                a["aspect"] for a in self.original["aspect_checks"]
            }
            allowed = {e["evidence_alias"] for e in claim["evidence"]}
            supports = [
                pid
                for pid, passage in payload["support_passages"].items()
                if passage[0] in allowed and passage[1] in payload["original_supports"]
            ]
            assert supports
            decision = copy.deepcopy(self.original)
            decision["claim_id"] = claim["claim_id"]
            if (
                "classification_valid"
                not in payload["schema"]["$defs"]["PassageDecision"]["properties"]
            ):
                # This distinct controlled response follows the current wire;
                # the original recorded raw and its scientific defect stay intact.
                decision.pop("classification_valid")
            decision["support_basis"] = [{"support_id": supports[0]}]
            for aspect in decision["aspect_checks"]:
                aspect["support_ids"] = [supports[0]]
            value = json.loads(response.content)
            value["decisions"] = [
                decision if row["claim_id"] == claim["claim_id"] else row
                for row in value["decisions"]
            ]
            self.injected = decision
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        stability.read_inventory_catalog_at_provider(self)


@pytest.mark.asyncio
@pytest.mark.parametrize("claim_id", ["claim_87", "claim_89"])
async def test_recorded_conflict_shape_stops_actual_v2_sdk_batch_with_usage_retained(
    case, tmp_path, monkeypatch, claim_id
):
    """Project recorded semantics to current HTTP; exact old raw replay is separate."""
    record = FIXTURE["nerf_failed_batch"]
    original = next(
        d
        for d in json.loads(record["raw_response"])["decisions"]
        if d["claim_id"] == claim_id
    )
    case.body["methods"][0]["sentences"][0]["text"] = (
        "On D, Relation networks achieve 30 FPS."
        if claim_id == "claim_87"
        else "On D, Relation networks learn a comparison function."
    )
    select_definition_policy(monkeypatch)
    gateway = RecordedConflictGateway(case, original)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    assert gateway.original == original and gateway.injected is not None
    assert original["classification_valid"] is True
    assert "classification_valid" not in gateway.injected
    assert not report["delivery"]["publishable"]
    assert not sent_with(gateway, "targets")
    assert not sent_with(gateway, "global_output_contract")
    assert len(sent_with(gateway, "cited_materials")) == 1
    progress = json.loads(
        next((tmp_path / "evaluation").glob("*/progress.json")).read_text()
    )["progress"]
    [batch] = progress["batches"]
    assert batch["status"] == "invalid_response"
    assert batch["reason_code"] == "evaluation_decision_consistency_invalid"
    assert batch["usage"]["usage_available"] is True
    [failure] = batch["diagnostics"]["decision_inconsistencies"]
    assert failure["subreason"] == "negative_status_with_complete_positive_diagnostics"
    assert failure["original_decision"]["status"] == original["status"]
    assert failure["original_decision"]["reason_code"] == original["reason_code"]
    assert all(a["supported"] for a in failure["original_decision"]["aspect_checks"])
    saved = json.loads(progress["requests"][-1]["response"])
    assert gateway.injected in saved["decisions"]
    _, journal, ledger = products(tmp_path)
    assert not journal["candidates"]
    assert ledger["requests_sent"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]
    assert all(e["status"] == "settled" for e in ledger["entries"].values())
