"""Verify opt-in classification through real writing, assessment and delivery."""

import copy
import json

import pytest
import test_located_evaluation as located
from test_flat_writer_pipeline import FlatGateway
from test_writer_v2 import case  # noqa: F401

from litagent.eval import scoped
from litagent.quality import strict_acceptance
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.workbench import RunOptions
from litagent.writing_v2 import writer_budget_plan

PLAN_TEXT = (
    "This chapter separates method mechanisms from experimental rankings; the label "
    '"outperforms" is discussed only when comparing reported experiments in the next section.'
)


class ClassificationGateway(FlatGateway):
    """Script semantic judgments at HTTP without changing application consumers."""

    def __init__(self, case, fault):
        super().__init__(case)
        self.declined = fault == "uncited_science"
        self.review_fault = fault
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                for fragment in value["fragments"].values():
                    for claim in fragment["assertions"]:
                        if fault == "all_organization":
                            claim["kind"] = "report_structure"
                        if claim["claim_text"].startswith("This chapter"):
                            claim["kind"] = (
                                "scientific"
                                if fault == "uncited_science"
                                else (
                                    "runtime_method"
                                    if fault == "runtime"
                                    else "report_structure"
                                )
                            )
                    if fault == "empty" and any(
                        c["claim_text"].startswith("This chapter")
                        for c in fragment["assertions"]
                    ):
                        fragment.update(
                            assertions=[], exclusion_reason="No assertions."
                        )
            if "cited_materials" in payload:
                by_id = {c["claim_id"]: c for c in payload["assertions"]}
                for decision in value["decisions"]:
                    if fault == "all_organization":
                        decision["correct_kind"] = "report_structure"
                    if not by_id[decision["claim_id"]]["claim_text"].startswith(
                        "This chapter"
                    ):
                        continue
                    decision["correct_kind"] = by_id[decision["claim_id"]]["kind"]
                    if fault == "misclassified":
                        decision.update(
                            correct_kind="scientific", classification_valid=False
                        )
                    if fault == "unfaithful":
                        decision["source_faithful"] = False
            if "analyses" in value:
                science = {
                    c["claim_id"]
                    for c in payload["assertions"]
                    if c["kind"] == "scientific"
                }
                for row in [
                    *value["analyses"],
                    value["limitations"],
                    *value["coverage"],
                ]:
                    row["claim_ids"] = [c for c in row["claim_ids"] if c in science]
                if fault == "global_classification":
                    cid = next(
                        c["claim_id"]
                        for c in payload["assertions"]
                        if c["claim_text"].startswith("This chapter")
                    )
                    value["issues"] = [
                        {
                            "code": "classification",
                            "anchor_ids": [cid],
                            "explanation": "A scientific premise was misclassified by extraction and assessment.",
                        }
                    ]
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


def with_transition(case, *, scientific=False):
    """Add prose without deleting any core science or changing its evidence pool."""
    text = (
        "This chapter finds EViT improves accuracy by 50%." if scientific else PLAN_TEXT
    )
    case.body["methods"].insert(
        0, {"sentences": [{"text": text, "fact_card_refs": []}]}
    )
    for section in case.summaries.values():
        section[0]["sentences"][0]["body_unit_refs"] = ["u4"]
    return text


def recorded_methodology(case, monkeypatch):
    """Provide actual fixture discovery records so unrelated gates can pass."""
    from litagent.review_structure import build_methodology

    def build(query, upstream, **kwargs):
        records = {
            "__search_queries__": {"search_fixture": "relation networks"},
            "search_fixture": list(case.ledger.values()),
            "dedup": list(case.ledger.values()),
            "relevance_gate": list(case.ledger.values()),
        }
        kwargs["source_outcomes"] = [
            {
                "task_id": "search_fixture",
                "source": "offline-public-fixture",
                "status": "success",
                "result_count": len(case.ledger),
            }
        ]
        return build_methodology(query, records, **kwargs)

    monkeypatch.setattr(located, "build_methodology", build)


def test_new_policy_changes_plan_run_and_request_identity_without_changing_old(
    case, monkeypatch
):
    kwargs = dict(methodology={}, budget_policy="planned-budget-v7")
    old = writer_budget_plan(case.packet, case.aliases, **kwargs)
    new = writer_budget_plan(
        case.packet, case.aliases, **kwargs, classification_review=True
    )
    assert old["status"] == new["status"] == "admitted"
    assert "classification_review_contract" not in old["envelope"]
    assert (
        new["envelope"]["classification_review_contract"]
        == scoped.INDEPENDENT_CLASSIFICATION_CONTRACT
    )
    assert new["identity"] != old["identity"] and new["repair_limit"] == 2
    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v7",
        capacity_policy="adaptive-capacity-v1",
        verification={"judge": False, "jev": False},
    )
    original = options.fingerprint()
    current = RunOptions(**{**options.model_dump(), "classification_review": True})
    revised = current.fingerprint()
    assert revised != original
    assert "classification_review" not in options.model_dump()
    monkeypatch.setattr(
        scoped,
        "INDEPENDENT_CLASSIFICATION_INSTRUCTION",
        "Different classification responsibility.",
    )
    assert options.fingerprint() == original
    assert current.fingerprint() != revised
    assert writer_budget_plan(case.packet, case.aliases, **kwargs) == old
    assert (
        writer_budget_plan(
            case.packet, case.aliases, **kwargs, classification_review=True
        )["identity"]
        != new["identity"]
    )


@pytest.mark.parametrize(
    "policy", ["planned-budget-v2", "planned-budget-v5", "planned-budget-v6"]
)
def test_new_contract_never_silently_upgrades_old_policy(case, policy):
    with pytest.raises(ValueError, match="classification_review_policy_invalid"):
        writer_budget_plan(
            case.packet,
            case.aliases,
            methodology={},
            budget_policy=policy,
            classification_review=True,
        )


@pytest.mark.parametrize("fault", ["revision", "parent", "summary"])
def test_invalid_envelope_cannot_enable_semantic_exemption(case, fault):
    envelope = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        classification_review=True,
    )["envelope"]
    if fault == "revision":
        envelope["classification_review_contract"] = "unknown"
    else:
        envelope.pop(
            "summary_constraints_contract"
            if fault == "summary"
            else "global_findings_contract"
        )
    with pytest.raises(ValueError, match="classification_review_invalid"):
        validate_writer_envelope(envelope)


@pytest.mark.asyncio
async def test_complete_pipeline_independently_verifies_quoted_plan_without_false_failure(
    case, tmp_path, monkeypatch
):
    text = with_transition(case)
    recorded_methodology(case, monkeypatch)
    monkeypatch.setattr(located, "FindingsGateway", ClassificationGateway)
    report, gateway = await located.run_case(
        case, tmp_path, monkeypatch, "good", classification_review=True
    )
    details = report["evaluation"]["faithfulness"]["details"]
    assert (
        details["classification_review_contract"]
        == scoped.INDEPENDENT_CLASSIFICATION_CONTRACT
    )
    organization = [c for c in details["assertions"] if c["kind"] == "report_structure"]
    assert len(organization) == 1 and organization[0]["claim_text"] == text
    assert (
        organization[0]["classification_valid"] and organization[0]["source_faithful"]
    )
    assert details["coverage"]["complete"]
    assert report["evaluation"]["faithfulness"]["score"] == 1.0
    assert strict_acceptance(details)["status"] == "passed"
    assert report["delivery"]["publishable"]
    assert not report["metadata"]["targeted_repair_state"]["attempts"]
    assert all(
        row["request"]["model"] == "lite"
        for row in gateway.sent
        if "allowed_citation_aliases" in row["payload"]
        or "cited_materials" in row["payload"]
    )
    assessment = [row for row in gateway.sent if "cited_materials" in row["payload"]]
    assert assessment and any(
        "report_structure" in str(row["payload"]) for row in assessment
    )
    assert any("global_output_contract" in row["payload"] for row in gateway.sent)
    for row in gateway.sent:
        payload = row["payload"]
        if any(
            k in payload
            for k in (
                "allowed_citation_aliases",
                "cited_materials",
                "global_output_contract",
            )
        ):
            prompt = row["request"]["messages"][0]["content"]
            assert "Classify propositions, not quoted labels/plans" in prompt
            if "allowed_citation_aliases" in payload or "cited_materials" in payload:
                assert "Mixed example:" in prompt
                assert "Never exempt the whole mixed sentence" in prompt
            assert set(payload["writer_envelope"]) == {
                "claim_json_bytes",
                "claim_max_json_bytes",
                "diagnostic_json_bytes",
            }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "misclassified",
        "unfaithful",
        "global_classification",
        "uncited_science",
        "runtime",
        "empty",
        "all_organization",
    ],
)
async def test_science_cannot_escape_by_an_organization_label_or_prefix(
    case, tmp_path, monkeypatch, fault
):
    with_transition(case, scientific=fault not in {"unfaithful", "empty"})
    recorded_methodology(case, monkeypatch)
    monkeypatch.setattr(located, "FindingsGateway", ClassificationGateway)
    report, gateway = await located.run_case(
        case, tmp_path, monkeypatch, fault, classification_review=True
    )
    assert not report["delivery"]["publishable"]
    details = report["evaluation"]["faithfulness"]["details"]
    assert strict_acceptance(details)["status"] != "passed"
    if fault == "uncited_science":
        assert details["unsupported_count"] >= 1
        assert any(c["local_error"] == "uncited_claim" for c in details["assertions"])
    elif fault == "empty":
        assert report["evaluation"]["faithfulness"]["skipped"]
        assert details["skipped_reason"] == "evaluation_extraction_incomplete"
    elif fault == "runtime":
        assert "evaluation_runtime_classification_unverified" in str(details)
    elif fault == "all_organization":
        assert details["factual_claim_count"] == 0
        assert not details["core_coverage"][0]["claim_ids"]
    else:
        assert details["inventory_check"]["status"] == "failed"
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    if fault == "uncited_science":
        assert 0 < len(attempts) <= 2
        assert (
            report["metadata"]["evidence_rewrite"]["reason_code"]
            == "writer_repair_declined"
        )
    else:
        assert not attempts
        assert not any("targets" in row["payload"] for row in gateway.sent)


def test_server_manifest_must_explicitly_bind_option_and_plain_api_does_not_expose_it():
    from litagent.api import SurveyRequest
    from scripts import run_frozen_writing as cli

    old = {"writing_profile": cli.ROUND_PROFILE}
    assert cli.classification_review_options(old) == {}
    assert cli.manifest_options(old) == cli.profile_options(
        cli.ROUND_PROFILE
    ).model_dump(mode="json")
    new = {**old, "classification_review": True}
    assert cli.classification_review_options(new) == {"classification_review": True}
    assert cli.manifest_options(new)["classification_review"]
    assert "classification_review" not in SurveyRequest.model_fields
    for value in [False, None, 1, "true"]:
        with pytest.raises(ValueError, match="configuration_invalid"):
            cli.classification_review_options({**old, "classification_review": value})


def test_old_and_new_checkpoint_owners_cannot_restore_each_other(case):
    from litagent.safety.planned_budget import PlannedBudget

    common = {"methodology": {}, "budget_policy": "planned-budget-v7"}
    old = writer_budget_plan(case.packet, case.aliases, **common)
    new = writer_budget_plan(
        case.packet, case.aliases, **common, classification_review=True
    )
    for first, second in [(old, new), (new, old)]:
        owner = PlannedBudget(first)
        with pytest.raises(ValueError, match="identity_conflict"):
            owner.restore(PlannedBudget(second).snapshot())
