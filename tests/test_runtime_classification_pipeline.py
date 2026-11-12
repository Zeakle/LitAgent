"""Complete selected runtime classification without granting body provenance.

The inventories and verdicts are synthetic transport fixtures, not scientific
gold labels. All planning, SDK, assessment, recovery and delivery consumers run.
"""

import copy
import json

import httpx
import pytest
import test_evaluation_stability_pipeline as stability
import test_mixed_inventory_pipeline as mixed
from test_flat_writer_pipeline import FlatGateway
from test_source_map_pipeline import products, sent_with
from test_support_receipts_pipeline import SELECTION
from test_writer_v2 import case  # noqa: F401

from litagent.eval import scoped
from litagent.eval.batched import _validate_decision_consistency
from litagent.quality import digest
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writing_v2 import writer_budget_plan

REVISION = "host-support-receipts-v2"
RUNTIME_TEXT = "The recorded retrieval query was relation networks."


def select_policy(monkeypatch, revision=REVISION, *, current_stack=False):
    """Choose the server-side plan through the actual frozen entry points."""
    if revision is None:
        return
    original_plan, original_run = mixed.plan_frozen_budget, mixed.run_frozen
    flags = {**SELECTION, "support_validation_revision": revision}
    if current_stack:
        flags.update(
            repair_mapping_contract="capacity-source-map-v3",
            repair_guidance_contract="host-structured-repair-v1",
            repair_guidance_wire_revision="host-repair-task-view-v1",
        )

    def plan(*args, **kwargs):
        return original_plan(*args, **{**kwargs, **flags})

    async def run(*args, **kwargs):
        return await original_run(*args, **{**kwargs, **flags})

    monkeypatch.setattr(mixed, "plan_frozen_budget", plan)
    monkeypatch.setattr(mixed, "run_frozen", run)


class RuntimeGateway(FlatGateway):
    """Change only external replies while keeping real request and usage records."""

    def __init__(self, case, fault="recover", *, science_failure=None):
        super().__init__(case)
        self.runtime_fault = fault
        self.science_failure = science_failure
        self.inventories = 0
        self.recovery_requests = 0
        self.repair_count = 0
        self.runtime_decisions = []
        self.runtime_records = []
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                self.inventories += 1
                active = self.inventories == 1 or fault in {
                    "runtime",
                    "repeat",
                    "contradiction",
                    "organization_contradiction",
                }
                if active:
                    for fragment in value["fragments"].values():
                        for claim in fragment["assertions"]:
                            if not (
                                claim["claim_text"].startswith("introduction:")
                                or claim["claim_text"].startswith(RUNTIME_TEXT)
                            ):
                                continue
                            claim["kind"] = (
                                "report_structure"
                                if fault == "organization_contradiction"
                                else "runtime_method"
                            )
                            if fault == "repeat":
                                claim[
                                    "claim_text"
                                ] += f" Extraction-only variant {self.inventories}."
            if "cited_materials" in payload:
                claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
                for decision in value["decisions"]:
                    claim = claims[decision["claim_id"]]
                    if claim["kind"] in {"runtime_method", "report_structure"}:
                        decision.update(
                            correct_kind=claim["kind"],
                            support_basis=[],
                            aspect_checks=[],
                        )
                        if fault != "runtime":
                            decision["correct_kind"] = "scientific"
                            contradictory = fault in {
                                "contradiction",
                                "organization_contradiction",
                            }
                            if contradictory or "classification_valid" in decision:
                                decision["classification_valid"] = contradictory
                        if fault == "repeat":
                            decision["source_faithful"] = False
                        self.runtime_decisions.append(copy.deepcopy(decision))
                        self.runtime_records.append(
                            copy.deepcopy(payload["host_runtime_record"])
                        )
                        assert claim["evidence"] == []
                        assert claim["required_support_aspects"] == []
                    if "FLOPs by" in claim["claim_text"]:
                        decision.update(
                            status="unsupported",
                            reason_code="insufficient_support",
                            support_basis=[],
                            explanation="The cited source gives no FLOPs reduction.",
                        )
                        for aspect in decision["aspect_checks"]:
                            aspect.update(supported=False, support_ids=[])
            if "analyses" in value:
                scientific = {
                    claim["claim_id"]
                    for claim in payload["assertions"]
                    if claim["kind"] == "scientific"
                }
                for row in [
                    *value["analyses"],
                    value["limitations"],
                    *value["coverage"],
                ]:
                    row["claim_ids"] = [
                        cid for cid in row["claim_ids"] if cid in scientific
                    ]
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        stability.read_inventory_catalog_at_provider(self)

    async def __call__(self, request):
        body = json.loads(request.content)
        payload = json.loads(body.get("input", body.get("messages"))[1]["content"])
        recovering = "inventory_recovery" in payload
        if recovering:
            self.recovery_requests += 1
        response = await super().__call__(request)
        if recovering and self.runtime_fault == "unknown":
            value = response.json()
            value.pop("usage", None)
            return httpx.Response(200, json=value)
        return response

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if self.science_failure is not None:
            if payload.get("requested_body_section") == "methods":
                value["units"][0]["text"] = "Relation networks reduce FLOPs by 44%."
            if "targets" in payload:
                self.repair_count += 1
                for patch in value["patches"]:
                    patch.update(
                        text=(
                            "Relation networks reduce FLOPs by 45%."
                            if self.science_failure
                            else "In the cited few-shot setting, Relation networks learn a comparison function."
                        ),
                        fact_card_refs=[self.case.by_eid["b"]],
                    )
        return value


def initial_details(report):
    return report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]


def assert_settled(gateway, ledger):
    assert ledger["requests_sent"] == ledger["call_count"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize("current_stack", [False, True])
async def test_runtime_misclassification_completes_parent_round_then_recovers_same_prose(
    case, tmp_path, monkeypatch, current_stack
):
    select_policy(monkeypatch, current_stack=current_stack)
    gateway = RuntimeGateway(case)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    state, journal, ledger = products(tmp_path)
    initial = initial_details(report)
    [runtime] = [
        claim for claim in initial["assertions"] if claim["kind"] == "runtime_method"
    ]
    assert runtime["classification_valid"] is False
    assert runtime["correct_kind"] == "scientific"
    assert runtime["local_error"] == "evaluation_classification_or_source_invalid"
    assert initial["assessment_stage"] == "host_validated"
    assert initial["stages"]["global_check"] == "completed"
    assert initial["inventory_check"]["status"] == "failed"
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["source_report_sha256"] == initial["source_report_sha256"]
    assert final["factual_claim_count"] == initial["factual_claim_count"] + 1
    assert final["inventory_check"]["status"] == "passed"
    assert final["coverage"]["complete"] and report["delivery"]["publishable"]
    assert gateway.recovery_requests == 1 and gateway.repair_count == 0
    assert len(sent_with(gateway, "global_output_contract")) == 2
    assert state["inventory_recovery_0"]["feedback"]["findings"]
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert not journal["candidates"]
    if current_stack:
        assert (
            ledger["plan"]["envelope"]["repair_mapping_contract"]
            == "capacity-source-map-v3"
        )
        assert (
            ledger["plan"]["envelope"]["repair_guidance_wire_revision"]
            == "host-repair-task-view-v1"
        )
    assert_settled(gateway, ledger)


@pytest.mark.asyncio
async def test_body_runtime_matching_real_host_query_never_gets_host_provenance(
    case, tmp_path, monkeypatch
):
    case.body["introduction"][0]["sentences"][0].update(
        text=RUNTIME_TEXT, fact_card_refs=[]
    )
    select_policy(monkeypatch)
    gateway = RuntimeGateway(case, "runtime")
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    details = report["evaluation"]["faithfulness"]["details"]
    [runtime] = [
        claim for claim in details["assertions"] if claim["kind"] == "runtime_method"
    ]
    assert (
        gateway.runtime_records[0]["search_queries"]["search_fixture"]
        == "relation networks"
    )
    assert gateway.runtime_decisions[0]["status"] == "supported"
    assert runtime["claim_text"] == RUNTIME_TEXT
    assert runtime["status"] == "unverifiable"
    assert runtime["local_error"] == "evaluation_runtime_classification_unverified"
    assert runtime["claim_id"] in details["method_assertion_errors"]
    assert details["stages"]["global_check"] == "completed"
    assert not details["coverage"]["complete"]
    assert report["evaluation"]["faithfulness"]["skipped"] is True
    assert details["unverified_reason"] == "evaluation_assertions_unverifiable"
    assert not report["delivery"]["publishable"]
    assert gateway.recovery_requests == 0 and not sent_with(gateway, "targets")
    assert not journal["candidates"]
    assert_settled(gateway, ledger)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["contradiction", "organization_contradiction"])
async def test_classification_flag_conflict_stops_before_global_and_keeps_usage(
    case, tmp_path, monkeypatch, fault
):
    select_policy(monkeypatch)
    gateway = RuntimeGateway(case, fault)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    progress = json.loads(
        next((tmp_path / "evaluation").glob("*/progress.json")).read_text()
    )["progress"]
    [batch] = progress["batches"]
    assert batch["status"] == "invalid_response"
    # The selected wire forbids this redundant model field altogether. Its raw
    # intrusion stays a protocol failure rather than being normalized or repaired.
    assert batch["reason_code"] == "evaluation_assessment_classification_field_invalid"
    assert batch["usage"]["usage_available"] is True
    assert not sent_with(gateway, "global_output_contract")
    assert gateway.recovery_requests == 0 and not sent_with(gateway, "targets")
    assert not report["delivery"]["publishable"] and not journal["candidates"]
    assert gateway.runtime_decisions[0]["classification_valid"] is True
    assert_settled(gateway, ledger)


@pytest.mark.asyncio
@pytest.mark.parametrize("science_failure", [False, True])
@pytest.mark.parametrize("current_stack", [False, True])
async def test_runtime_recovery_and_prose_edit_share_two_http_attempts(
    case, tmp_path, monkeypatch, science_failure, current_stack
):
    select_policy(monkeypatch, current_stack=current_stack)
    gateway = RuntimeGateway(case, science_failure=science_failure)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["purpose"] == mixed.MIXED_CONTRACT
    assert gateway.recovery_requests == gateway.repair_count == 1
    assert len(sent_with(gateway, "targets")) == 1
    assert len(sent_with(gateway, "global_output_contract")) == 3
    assert len(journal["candidates"]) == 1
    assert journal["candidates"][0]["committed"] is (not science_failure)
    assert report["delivery"]["publishable"] is (not science_failure)
    if current_stack:
        [repair] = sent_with(gateway, "targets")
        assert (
            repair["payload"]["repair_guidance"]["wire_revision"]
            == "host-repair-task-view-v1"
        )
        [call] = [
            item for item in journal["calls"].values() if item.get("stage") == "repair"
        ]
        assert call["guidance_wire_record"]
    if science_failure:
        assert "FLOPs by 44%" in report["survey"]
    assert_settled(gateway, ledger)


@pytest.mark.asyncio
async def test_two_real_runtime_recoveries_cannot_send_a_third_attempt(
    case, tmp_path, monkeypatch
):
    select_policy(monkeypatch)
    gateway = RuntimeGateway(case, "repeat")
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    assert gateway.recovery_requests == 2 and gateway.inventories == 3
    assert len(sent_with(gateway, "global_output_contract")) == 3
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 2
    assert not sent_with(gateway, "targets") and not journal["candidates"]
    assert not report["delivery"]["publishable"]
    assert_settled(gateway, ledger)


@pytest.mark.asyncio
async def test_unknown_runtime_recovery_keeps_hold_and_never_replays(
    case, tmp_path, monkeypatch
):
    select_policy(monkeypatch)
    gateway = RuntimeGateway(case, "unknown")
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    assert gateway.recovery_requests == 1
    assert ledger["usage_uncertain"] and ledger["stop_reason"] == "strict_usage_unknown"
    pending = [
        entry for entry in ledger["entries"].values() if entry.get("inventory_recovery")
    ]
    assert len(pending) == 1 and pending[0]["status"] == "interrupted_no_replay"
    assert pending[0]["actual_tokens"] is None and pending[0]["reserved_tokens"] > 0
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert not sent_with(gateway, "targets") and not journal["candidates"]


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [None, "host-support-receipts-v1"])
async def test_legacy_runtime_branch_still_stops_before_assessment(
    case, tmp_path, monkeypatch, revision
):
    select_policy(monkeypatch, revision)
    gateway = RuntimeGateway(case)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    assert not sent_with(gateway, "cited_materials")
    assert not sent_with(gateway, "global_output_contract")
    assert gateway.recovery_requests == 0
    assert not report["delivery"]["publishable"]


@pytest.mark.parametrize(
    "old_identity",
    [
        "fece6428782b0816de24475a6e34191567a58d6c8e6647cff1db0e7c9413e9d1",
        "3e559ce9fc83869035292dfacec4d431e57d3ab0cebe7b988350d373430d116c",
    ],
)
def test_execution_identity_isolates_prior_v2_without_changing_v1(case, old_identity):
    assert (
        scoped.support_validation_identity()
        == "1cafb459f74f09ce5276aacebabc85bb8dbac3ee092ace9de01aa3039313f607"
    )
    assert scoped.support_validation_identity(REVISION) != old_identity
    flags = {**SELECTION, "support_validation_revision": REVISION}
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **flags,
    )
    assert scoped._runtime_classification_round_enabled(plan["envelope"])
    old = copy.deepcopy(plan["envelope"])
    old["support_validation_identity"] = old_identity
    old["identity"] = digest(
        {key: value for key, value in old.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(old)


@pytest.mark.parametrize("kind", ["scientific", "runtime_method", "report_structure"])
def test_kind_conflict_is_checked_before_every_assigned_kind_early_return(case, kind):
    flags = {**SELECTION, "support_validation_revision": REVISION}
    envelope = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **flags,
    )["envelope"]
    claim = {
        "claim_id": "claim_1",
        "source_unit_id": "u1",
        "source_fragment_id": "u1:fragment:0",
        "kind": kind,
    }
    decision = scoped.PassageDecision(
        claim_id="claim_1",
        correct_kind="runtime_method" if kind == "scientific" else "scientific",
        atomic=True,
        classification_valid=True,
        source_faithful=True,
        status="supported",
        reason_code="supported",
        explanation="Controlled contradictory class.",
        support_basis=[],
        aspect_checks=[],
    )
    original = decision.model_dump()
    with pytest.raises(
        scoped.ScopedEvaluationError, match="evaluation_decision_consistency_invalid"
    ) as error:
        _validate_decision_consistency(claim, decision, {"writer_envelope": envelope})
    [diagnostic] = error.value.details["decision_inconsistencies"]
    assert diagnostic["subreason"] == "classification_flag_kind_mismatch"
    assert diagnostic["original_decision"] == original == decision.model_dump()


@pytest.mark.parametrize(
    "untrusted", [None, False, True, {}, [], 1, "host-support-receipts-v2"]
)
def test_untrusted_context_values_cannot_select_a_runtime_round(untrusted):
    assert scoped._runtime_classification_round_enabled(untrusted) is False


def current_runtime_context(case):
    """Derive the private selector from the real current validated envelope."""
    flags = {
        **SELECTION,
        "support_validation_revision": REVISION,
        "repair_mapping_contract": "capacity-source-map-v3",
        "repair_guidance_contract": "host-structured-repair-v1",
        "repair_guidance_wire_revision": "host-repair-task-view-v1",
    }
    envelope = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **flags,
    )["envelope"]
    validate_writer_envelope(envelope)
    context = {
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "_independent_classification": True,
        "_runtime_classification_round": scoped._runtime_classification_round_enabled(
            envelope
        ),
        "_support_validation_revision": REVISION,
        "review_methodology": {
            "search_queries": {"search_fixture": "relation networks"}
        },
    }
    assert context["_runtime_classification_round"] is True
    return context


def checked_runtime_claim(local_error):
    """Retain a body assertion and a matching record without host provenance."""
    return {
        "claim_id": "claim_1",
        "kind": "runtime_method",
        "claim_text": RUNTIME_TEXT,
        "source_unit_id": "u1",
        "source_fragment_id": "u1:fragment:0",
        "source_start": 0,
        "source_end": len(RUNTIME_TEXT),
        "source_quote": RUNTIME_TEXT,
        "evidence_ids": ["unknown"] if local_error == "unknown_evidence_ref" else [],
        "local_error": local_error,
    }


def runtime_decision(**changes):
    """Supply an independent decision without promoting the host observation."""
    return scoped.PassageDecision(
        **{
            "claim_id": "claim_1",
            "correct_kind": "runtime_method",
            "atomic": True,
            "classification_valid": True,
            "source_faithful": True,
            "status": "supported",
            "reason_code": "supported",
            "explanation": "The supplied record contains this exact retrieval query.",
            "support_basis": [],
            "aspect_checks": [],
            **changes,
        }
    )


@pytest.mark.parametrize(
    "local_error",
    [
        "",
        "unknown_evidence_ref",
        "evaluation_source_scope_unverified",
        "evaluation_atomic_classification_unverified",
    ],
)
def test_runtime_validation_keeps_prior_local_error_and_unverified_provenance(
    case, local_error
):
    context = current_runtime_context(case)
    claim = checked_runtime_claim(local_error)
    decision = runtime_decision()
    original_claim, original_decision = copy.deepcopy(claim), decision.model_dump()
    [observed] = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger={},
        aliases={},
        support_passages={},
        context=context,
        strict=True,
    )
    assert observed["status"] == observed["reason_code"] == "unverifiable"
    assert observed["local_error"] == (
        local_error or "evaluation_runtime_classification_unverified"
    )
    assert (
        observed["claim_text"] == RUNTIME_TEXT and observed["kind"] == "runtime_method"
    )
    assert (
        observed["classification_valid"]
        and observed["source_faithful"]
        and observed["atomic"]
    )
    assert claim == original_claim and decision.model_dump() == original_decision


@pytest.mark.parametrize(
    "change,expected",
    [
        (
            {"classification_valid": False},
            "evaluation_classification_or_source_invalid",
        ),
        ({"source_faithful": False}, "evaluation_classification_or_source_invalid"),
        ({"atomic": False}, "evaluation_atomic_classification_unverified"),
    ],
)
def test_runtime_validation_preserves_prior_classification_fidelity_atomic_priority(
    case, change, expected
):
    decision = runtime_decision(**change)
    [observed] = scoped.validate_assessment_decisions(
        [checked_runtime_claim("unknown_evidence_ref")],
        [decision],
        ledger={},
        aliases={},
        support_passages={},
        context=current_runtime_context(case),
        strict=True,
    )
    assert observed["status"] == observed["reason_code"] == "unverifiable"
    assert observed["local_error"] == expected
    for field, value in change.items():
        assert observed[field] is value
