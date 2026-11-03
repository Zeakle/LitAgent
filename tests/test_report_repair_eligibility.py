"""Authorize finite host semantic repairs through real report consumers."""

import copy
import json
from collections import Counter

import httpx
import pytest
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from test_complete_native_repair import source_selection
from test_flat_writer_pipeline import FlatGateway
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.quality import build_fact_cards, digest
from litagent.writing_flat import REPAIR_TASK_VIEW_REVISION
from litagent.writing_v2 import BODY, SCOPE, source_packet

SEMANTIC_CASES = {
    "entity": (
        "GhostNet learns a comparison function.",
        "Relation networks learn a comparison function.",
        "Relation networks learn a comparison function.",
        "support_entity_unverified",
    ),
    "relation": (
        "Relation networks reduce parameters.",
        "Relation networks learn a comparison function.",
        "Relation networks learn a comparison function.",
        "support_relation_unverified",
    ),
    "number": (
        "Model-A achieves accuracy of 91%.",
        "Model-A achieves accuracy of 90%.",
        "Model-A achieves accuracy of 90%.",
        "support_number_unverified",
    ),
    "metric": (
        "Latency decreases by 20%.",
        "Throughput increases by 20%.",
        "Throughput increases by 20%.",
        "support_metric_unverified",
    ),
    "direction": (
        "Accuracy improves by 0.3%.",
        "Accuracy decreases by 0.3%.",
        "Accuracy decreases by 0.3%.",
        "support_direction_unverified",
    ),
    "mechanism_attribution": (
        "EViT uses slow-fast token evolution.",
        "EViT uses token reorganization. Evo-ViT uses slow-fast token evolution.",
        "EViT uses token reorganization.",
        "support_mechanism_attribution_unverified",
    ),
    "quantity_role": (
        "UE4-NeRF achieves an average frame rate of 43 FPS.",
        "UE4-NeRF renders at 4K resolution with a frame rate of up to 43 FPS.",
        "UE4-NeRF renders at 4K resolution with a frame rate of up to 43 FPS.",
        "support_quantity_role_unverified",
    ),
    "result_context": (
        "Model-A achieves average accuracy of 90%.",
        "Model-A achieves average accuracy of 80%. Model-A achieves peak accuracy of 90%.",
        "Model-A achieves average accuracy of 80%.",
        "support_result_context_unverified",
    ),
    "numeric_attribution": (
        "ResNet-18 achieves accuracy of 98.93% on TBX11K.",
        "ResNet-18 achieves accuracy of 92.4% on TBX11K. ResNet-12 achieves accuracy of 98.93% on TBX11K.",
        "ResNet-18 achieves accuracy of 92.4% on TBX11K.",
        "support_numeric_attribution_unverified",
    ),
    "binding": (
        "Model-A and Model-B achieve accuracies of 90% and 80%, respectively.",
        "Model-A and Model-B achieve accuracies of 80% and 90%, respectively.",
        "Model-A and Model-B achieve accuracies of 80% and 90%, respectively.",
        "support_binding_unverified",
    ),
    "quantity_basis": (
        "MCTF reduces FLOPs by 44%.",
        "MCTF reduces FLOPs to 44%.",
        "MCTF reduces FLOPs to 44%.",
        "support_quantity_basis_unverified",
    ),
}
PREVIOUS_V2_IDENTITY = (
    "8e020430e6db44dcb3c004e4552f0b59178247aa035c54dc23f158b77e5603af"
)
V1_IDENTITY = "1cafb459f74f09ce5276aacebabc85bb8dbac3ee092ace9de01aa3039313f607"


def prepare_case(case, family="entity", *, units_per_section=1):
    """Build complete current fact cards from synthetic original source material."""
    wrong, source, corrected, issue = SEMANTIC_CASES[family]
    weighted = (
        "Relation networks learn a comparison function for supplied visual examples "
        "under the conditions recorded in the selected source, with the comparison "
        "confined to those examples, the reported relation restricted to that cited "
        "setting, and the interpretation limited to the original source's stated "
        "conditions and visual comparison task."
    )
    distinct_repair = (
        "Under the cited source's visual comparison setting, relation networks "
        "learn a comparison function."
    )
    if units_per_section > 1:
        source += " " + weighted + " " + distinct_repair
    case.ledger["b"].update(text=source, supporting_text=source)
    case.cards = build_fact_cards(case.ledger)
    case.selection = source_selection(case.cards)
    case.selection.section_evidence_ids["methods"] = ["a"]
    case.selection.writing_scope_policy = SCOPE
    case.packet, case.aliases = source_packet(
        case.cards,
        case.selection,
        "Explain mechanisms and limitations.",
        [
            {
                "requirement_id": "q1",
                "kind": "question",
                "text": "Explain mechanisms and limitations.",
            }
        ],
        "en",
    )
    case.by_eid = {
        case.cards[cid]["evidence_id"]: ref for ref, cid in case.aliases.items()
    }
    case.body = {
        "title": "Source bound analysis",
        **{
            key: [
                {
                    "sentences": [
                        {"text": corrected, "fact_card_refs": [case.by_eid["b"]]}
                        for _ in range(units_per_section)
                    ]
                }
            ]
            for key in BODY
        },
    }
    case.body["methods"][0]["sentences"][0]["text"] = wrong
    if units_per_section > 1:
        for unit in case.body["introduction"][0]["sentences"]:
            unit["text"] = weighted
        # Independent paragraphs prevent a whole long source unit from being
        # repeated in both batches, so the real planner can balance text weight.
        for key in BODY:
            case.body[key] = [
                {"sentences": [unit]} for unit in case.body[key][0]["sentences"]
            ]
    case.summaries = {
        key: [{"sentences": [{"text": corrected, "body_unit_refs": ["u3"]}]}]
        for key in ("abstract", "conclusion")
    }
    case.semantic_wrong = wrong
    case.semantic_corrected = distinct_repair if units_per_section > 1 else corrected
    case.semantic_issue = issue
    return case


class SemanticGateway(FlatGateway):
    """Alter external replies while retaining SDK, accounting and host validation."""

    def __init__(self, case, failing_candidates=0, *, fault=None):
        super().__init__(case)
        self.failing_candidates = failing_candidates
        self.semantic_fault = fault
        self.repair_count = 0
        self.inventories = 0
        self.mixed_fragments = set()
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "allowed_citation_aliases" in payload:
                self.inventories += 1
                if fault == "mixed_same_unit":
                    value = json.loads(response.content)
                    self.mixed_fragments = {
                        f["source_fragment_id"]
                        for u in payload["source_units"]
                        for f in u["source_fragments"]
                        if case.semantic_wrong in f["text"]
                    }
                    for fid in self.mixed_fragments:
                        value["fragments"][fid]["assertions"] = [
                            {"kind": "scientific", "claim_text": text}
                            for text in (case.semantic_wrong, case.semantic_corrected)
                        ]
                    response.content = json.dumps(value)
            if "cited_materials" in payload:
                value = json.loads(response.content)
                claims = {c["claim_id"]: c for c in payload["assertions"]}
                for decision in value["decisions"]:
                    claim = claims[decision["claim_id"]]
                    if not claim["evidence"]:
                        decision.update(
                            status="unsupported",
                            reason_code="uncited_claim",
                            support_basis=[],
                        )
                        for aspect in decision["aspect_checks"]:
                            aspect.update(supported=False, support_ids=[])
                    if (
                        fault == "mixed_same_unit"
                        and claim["source_fragment_id"] in self.mixed_fragments
                        and case.semantic_corrected in claim["claim_text"]
                    ):
                        extra = copy.deepcopy(decision["aspect_checks"][0])
                        extra["aspect"] = "causal_relation"
                        decision["aspect_checks"].append(extra)
                    if case.semantic_wrong not in claim["claim_text"]:
                        continue
                    aspects = decision["aspect_checks"]
                    if fault == "extra_aspect":
                        extra = copy.deepcopy(aspects[0])
                        extra["aspect"] = "causal_relation"
                        aspects.append(extra)
                    elif fault == "missing_aspect":
                        aspects.pop()
                    elif fault == "duplicate_aspect":
                        aspects.append(copy.deepcopy(aspects[0]))
                    elif fault == "empty_basis":
                        decision["support_basis"] = []
                    elif fault == "duplicate_basis":
                        decision["support_basis"] *= 2
                    elif fault == "unknown_basis":
                        decision["support_basis"] = [{"support_id": "p99999"}]
                    elif fault == "wrong_paper_basis":
                        own = {e["evidence_alias"] for e in claim["evidence"]}
                        other = next(
                            pid
                            for pid, p in payload["support_passages"].items()
                            if p[0] not in own
                        )
                        decision["support_basis"] = [{"support_id": other}]
                        for aspect in aspects:
                            aspect["support_ids"] = [other]
                    elif fault == "classification":
                        decision["correct_kind"] = "report_structure"
                        if "classification_valid" in decision:
                            decision["classification_valid"] = False
                    elif fault == "source_fidelity":
                        decision["source_faithful"] = False
                    elif fault == "raw_status":
                        decision.update(
                            status="unverifiable", reason_code="unverifiable"
                        )
                if fault == "missing_decision":
                    value["decisions"].pop()
                elif fault == "duplicate_decision":
                    value["decisions"].append(copy.deepcopy(value["decisions"][0]))
                response.content = json.dumps(value)
            if "analyses" in json.loads(response.content):
                if fault == "global_incomplete":
                    value = json.loads(response.content)
                    value["units"][0]["complete"] = False
                    response.content = json.dumps(value)
                elif fault == "global_json":
                    response.content = '{"units":'
            return response

        self.evaluator.chat = chat
        from test_evaluation_stability_pipeline import (
            read_inventory_catalog_at_provider,
        )

        read_inventory_catalog_at_provider(self)

    def answer(self, payload, native_schema):
        """Produce candidates without editing any application state or verdict."""
        value = super().answer(payload, native_schema)
        if "targets" in payload:
            self.repair_count += 1
            for patch in value["patches"]:
                patch["text"] = (
                    f"AnotherGhostNet{self.repair_count} learns a comparison function."
                    if self.repair_count <= self.failing_candidates
                    else self.case.semantic_corrected
                )
                patch["fact_card_refs"] = [self.case.by_eid["b"]]
        return value

    async def __call__(self, request):
        """Keep unknown provider usage unknown at the actual HTTP boundary."""
        response = await super().__call__(request)
        if self.semantic_fault == "unknown_usage":
            body = json.loads(request.content)
            messages = body.get("input", body.get("messages"))
            if "cited_materials" in json.loads(messages[1]["content"]):
                value = json.loads(response.content)
                value.pop("usage", None)
                return httpx.Response(200, json=value)
        return response


def select_current_policy(
    monkeypatch, *, module=None, revision="host-support-receipts-v2"
):
    """Choose the authenticated V2 task view through real frozen entry points."""
    flags = {
        **structured.FLAGS,
        "repair_guidance_wire_revision": REPAIR_TASK_VIEW_REVISION,
    }
    if revision is None:
        flags.pop("support_validation_revision", None)
    else:
        flags["support_validation_revision"] = revision
    monkeypatch.setattr(structured, "FLAGS", flags)
    structured.select_structured_policy(monkeypatch, module=module)


async def run_semantic(
    case,
    tmp_path,
    monkeypatch,
    *,
    failing_candidates=0,
    fault=None,
    revision="host-support-receipts-v2",
):
    """Use Synthesis, Reviewer, report extraction, evaluation and repair unchanged."""
    gateway = SemanticGateway(case, failing_candidates, fault=fault)
    if fault == "unknown_usage":
        import test_mixed_inventory_pipeline as mixed

        select_current_policy(monkeypatch, module=mixed, revision=revision)
        return await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    if revision is not None:
        select_current_policy(monkeypatch, revision=revision)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, failing_candidates)
    return report, gateway


@pytest.mark.asyncio
@pytest.mark.parametrize("family", list(SEMANTIC_CASES))
async def test_legal_host_unverifiable_enters_repair_without_rewriting_old_score(
    case, tmp_path, monkeypatch, family
):
    """A candidate can repair prose while the original remains scientifically unverified."""
    prepare_case(case, family)
    report, gateway = await run_semantic(case, tmp_path, monkeypatch)
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"]
    details = initial["faithfulness"]["details"]
    assert details["coverage"]["complete"] is False
    assert details["inventory_check"]["status"] == "passed"
    assert details["stages"]["global_check"] == "completed"
    failures = [a for a in details["assertions"] if a["status"] == "unverifiable"]
    assert {a["local_error"] for a in failures} == {case.semantic_issue}
    raw = [d for b in details["stages"]["batches"] for d in b["decisions"]]
    assert all(d["status"] == "supported" for d in raw)
    assert gateway.repair_count == 1
    assert gateway.inventories == 2
    assert len(sent_with(gateway, "global_output_contract")) == 2
    assert report["delivery"]["publishable"] is True
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["coverage"]["complete"] is True
    assert final["supported_count"] == final["factual_claim_count"]
    assert final["unsupported_count"] == final["unverified_count"] == 0
    assert details["unverified_count"] > 0 and initial["faithfulness"]["skipped"]
    _, journal, ledger = products(tmp_path)
    assert journal["candidates"][0]["committed"] is True
    assert ledger["usage_uncertain"] is False
    assert all(e["status"] == "settled" for e in ledger["entries"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["extra_aspect", "missing_aspect", "duplicate_aspect"]
)
async def test_illegal_aspect_response_never_authorizes_body_repair(
    case, tmp_path, monkeypatch, fault
):
    """Protocol defects cannot become prose tasks through a valid neighboring fact."""
    prepare_case(case)
    report, gateway = await run_semantic(case, tmp_path, monkeypatch, fault=fault)
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"] is False
    assert gateway.repair_count == 0
    assert not sent_with(gateway, "targets")
    assert report["delivery"]["publishable"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_second_attempt_uses_original_and_complete_reassessment_never_a_third(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Uncommitted semantic failures consume the same two shared HTTP slots."""
    prepare_case(case)
    report, gateway = await run_semantic(
        case, tmp_path, monkeypatch, failing_candidates=failing_candidates
    )
    repairs = sent_with(gateway, "targets")
    expected = min(2, failing_candidates + 1)
    assert len(repairs) == gateway.repair_count == expected
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "global_output_contract")) == expected + 1
    for repair in repairs:
        current = json.dumps(repair["payload"]["current_body"])
        assert case.semantic_wrong in current
        assert "AnotherGhostNet" not in current
    state, journal, ledger = products(tmp_path)
    assert len(state["state"]["attempts"]) == expected
    assert len(journal["candidates"]) == expected
    assert journal["candidates"][0]["committed"] is (failing_candidates == 0)
    assert ledger["usage_uncertain"] is False
    assert all(e["status"] == "settled" for e in ledger["entries"].values())
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    assert journal["candidates"][-1]["committed"] is (failing_candidates < 2)
    if failing_candidates == 2:
        assert case.semantic_wrong in report["survey"]
        assert "AnotherGhostNet" not in report["survey"]
        assert (
            report["evaluation"]["faithfulness"]["details"]["coverage"]["complete"]
            is False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "empty_basis",
        "duplicate_basis",
        "unknown_basis",
        "wrong_paper_basis",
        "classification",
        "source_fidelity",
        "raw_status",
        "missing_decision",
        "duplicate_decision",
        "global_incomplete",
        "global_json",
        "mixed_same_unit",
    ],
)
async def test_protocol_source_and_incomplete_responses_never_become_prose_tasks(
    case, tmp_path, monkeypatch, fault
):
    """A valid semantic record cannot authorize a mixed invalid report."""
    prepare_case(case)
    if fault == "wrong_paper_basis":
        case.body["taxonomy"][0]["sentences"][0] = {
            "text": case.ledger["a"]["supporting_text"],
            "fact_card_refs": [case.by_eid["a"]],
        }
    if fault == "mixed_same_unit":
        case.body["methods"][0]["sentences"][0]["text"] = (
            case.semantic_wrong + " " + case.semantic_corrected
        )
    report, gateway = await run_semantic(case, tmp_path, monkeypatch, fault=fault)
    assert gateway.repair_count == 0
    assert not sent_with(gateway, "targets")
    assert report["delivery"]["publishable"] is False
    if fault == "mixed_same_unit":
        details = report["evaluation"]["faithfulness"]["details"]
        errors = {a.get("local_error") for a in details["assertions"]}
        assert {
            "support_entity_unverified",
            "support_aspect_coverage_invalid",
        } <= errors


@pytest.mark.asyncio
async def test_unknown_assessment_usage_keeps_hold_and_sends_no_repair(
    case, tmp_path, monkeypatch
):
    """Missing provider usage cannot be repaired or released as known accounting."""
    prepare_case(case)
    report, gateway = await run_semantic(
        case, tmp_path, monkeypatch, fault="unknown_usage"
    )
    assert not sent_with(gateway, "targets")
    assert report["delivery"]["publishable"] is False
    _, _, ledger = products(tmp_path)
    assert ledger["usage_uncertain"] is True
    assert ledger["stop_reason"] == "strict_usage_unknown"
    uncertain = [e for e in ledger["entries"].values() if e["status"] != "settled"]
    assert uncertain and all(e["actual_tokens"] is None for e in uncertain)
    assert all(e["reserved_tokens"] > 0 for e in uncertain)


def corrupt_observation(arguments, fault):
    """Damage a copy of authentic host observations, leaving the live owner intact."""
    value = {
        **arguments,
        **{
            key: copy.deepcopy(arguments[key])
            for key in ("report_data", "initial_evaluation", "results")
        },
    }
    report = value["report_data"]
    details = value["initial_evaluation"]["faithfulness"]["details"]
    stages = details["stages"]
    failure = next(a for a in details["assertions"] if a["status"] == "unverifiable")
    if fault == "forged_local_error":
        failure["local_error"] = "support_relation_unverified"
    elif fault == "forged_host_result":
        failure["support_validation"]["entity"] = "ForgedNet"
    elif fault == "canonical_status":
        failure.update(status="unsupported", reason_code="insufficient_support")
    elif fault == "diagnostic_issue":
        details["unsupported_claims"][0]["local_error"] = "support_relation_unverified"
    elif fault == "diagnostic_explanation":
        details["unsupported_claims"][0]["explanation"] = "A forged diagnosis."
    elif fault == "forged_uncited_with_refs":
        stages["inventory_assertions"][0]["local_error"] = "uncited_claim"
    elif fault == "raw_pseudo_code":
        raw = json.loads(stages["requests"][1]["response"])
        raw["decisions"][0]["reason_code"] = "support_entity_unverified"
        stages["requests"][1]["response"] = json.dumps(raw)
    elif fault == "raw_canonical_disagreement":
        stages["batches"][0]["decisions"][0][
            "explanation"
        ] = "A forged canonical value."
    elif fault == "batch_input_sha":
        stages["batches"][0]["input_sha256"] = "0" * 64
    elif fault == "request_input_sha":
        stages["requests"][1]["input_sha256"] = "0" * 64
    elif fault == "source_sha":
        details["source_report_sha256"] = "0" * 64
    elif fault == "binding_sha":
        report["metadata"]["writing_bindings"]["report_sha256"] = "0" * 64
    elif fault == "binding_quote":
        report["metadata"]["writing_bindings"]["units"][0][
            "source_quote"
        ] = "Forged text."
    elif fault == "source_snapshot":
        report["metadata"]["evidence_snapshot_sha256"] = "0" * 64
    elif fault == "native_document":
        report["metadata"]["writer_document"]["body"]["methods"][0]["sentences"][0][
            "text"
        ] = "Forged text."
    elif fault == "provider_usage":
        details["provider"]["usage_complete"] = False
    elif fault == "request_usage":
        stages["requests"][1]["usage"]["usage_available"] = False
    elif fault == "request_usage_sum":
        details["provider"]["usage"]["prompt_tokens"] += 1
    elif fault == "truncated_response":
        stages["requests"][1]["finish_reason"] = "length"
    elif fault == "missing_batch":
        stages["batches"].pop()
    elif fault == "partial_batch":
        stages["batches"][0]["status"] = "started_no_replay"
    elif fault == "missing_id":
        details["assertions"].pop()
    elif fault == "duplicate_id":
        details["assertions"][-1] = copy.deepcopy(details["assertions"][0])
    elif fault == "global_unfinished":
        stages["global_check"] = "not_started"
    elif fault == "old_identity":
        stages["support_validation_identity"] = "0" * 64
    elif fault == "previous_v2_identity":
        stages["support_validation_identity"] = PREVIOUS_V2_IDENTITY
        stages["evaluation_definitions"]["identity"] = PREVIOUS_V2_IDENTITY
        details["evaluation_definitions"]["identity"] = PREVIOUS_V2_IDENTITY
    elif fault == "resealed_caller_records":
        raw = json.loads(stages["requests"][1]["response"])
        raw["decisions"][0]["explanation"] = "Self-consistent forged model explanation."
        stages["requests"][1]["response"] = json.dumps(raw)
        stages["batches"][0]["decisions"][0]["explanation"] = raw["decisions"][0][
            "explanation"
        ]
        stages["requests"][1]["input_sha256"] = "0" * 64
        stages["batches"][0]["input_sha256"] = "0" * 64
        cid = stages["batches"][0]["decisions"][0]["claim_id"]
        next(a for a in details["assertions"] if a["claim_id"] == cid)[
            "explanation"
        ] = raw["decisions"][0]["explanation"]
    elif fault == "raw_duplicate_key":
        raw = stages["requests"][1]["response"]
        stages["requests"][1]["response"] = raw[:-1] + ',"decisions":[]}'
    else:
        raise AssertionError(f"Unknown corruption fixture: {fault}")
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "forged_local_error",
        "forged_host_result",
        "canonical_status",
        "diagnostic_issue",
        "diagnostic_explanation",
        "forged_uncited_with_refs",
        "raw_pseudo_code",
        "raw_canonical_disagreement",
        "batch_input_sha",
        "request_input_sha",
        "source_sha",
        "binding_sha",
        "binding_quote",
        "source_snapshot",
        "native_document",
        "provider_usage",
        "request_usage",
        "request_usage_sum",
        "truncated_response",
        "missing_batch",
        "partial_batch",
        "missing_id",
        "duplicate_id",
        "global_unfinished",
        "old_identity",
        "previous_v2_identity",
        "raw_duplicate_key",
        "resealed_caller_records",
    ],
)
async def test_authentic_completed_round_rejects_corrupt_observation_before_http(
    case, tmp_path, monkeypatch, fault
):
    """Real evidence replay, rather than a trusted error string, grants eligibility."""
    import litagent.writing_v2 as writing

    prepare_case(case)
    original = writing._host_unverified_repair_eligible
    observed = []

    def checked(runner, **arguments):
        # Only this adversarial trust-boundary test injects observation damage.
        # The genuine predicate still performs every check and supplies its result.
        assert original(runner, **arguments) is True
        before = digest(arguments["initial_evaluation"])
        result = original(runner, **corrupt_observation(arguments, fault))
        assert digest(arguments["initial_evaluation"]) == before
        observed.append(result)
        return result

    monkeypatch.setattr(writing, "_host_unverified_repair_eligible", checked)
    report, gateway = await run_semantic(case, tmp_path, monkeypatch)
    assert observed == [False]
    assert not sent_with(gateway, "targets")
    assert (
        report["metadata"]["evidence_rewrite"]["reason_code"]
        == "writer_assessment_incomplete"
    )
    assert report["delivery"]["publishable"] is False


@pytest.mark.asyncio
async def test_semantic_gap_and_actual_uncited_negative_can_be_repaired_together(
    case, tmp_path, monkeypatch
):
    """A complete negative for missing citations remains an ordinary prose target."""
    prepare_case(case)
    case.body["introduction"][0]["sentences"][0]["fact_card_refs"] = []
    report, gateway = await run_semantic(case, tmp_path, monkeypatch)
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    uncited = [
        a for a in initial["assertions"] if a.get("local_error") == "uncited_claim"
    ]
    assert uncited and all(not a["evidence_ids"] for a in uncited)
    assert all(a["status"] == "unsupported" for a in uncited)
    assert initial["unverified_count"] > 0 and initial["unsupported_count"] > 0
    assert initial["coverage"]["complete"] is False
    assert gateway.repair_count == 1 and gateway.inventories == 2
    assert report["delivery"]["publishable"] is True
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["coverage"]["complete"] is True
    assert final["unsupported_count"] == final["unverified_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "missing_id", "duplicate_id"])
async def test_balanced_real_batches_require_one_complete_unique_claim_set(
    case, tmp_path, monkeypatch, fault
):
    """Real batch balancing may reorder groups without losing or duplicating IDs."""
    import litagent.writing_v2 as writing

    prepare_case(case, units_per_section=4)
    original = writing._host_unverified_repair_eligible
    observed = []

    def checked(runner, **arguments):
        details = arguments["initial_evaluation"]["faithfulness"]["details"]
        ids = [a["claim_id"] for a in details["assertions"]]
        groups = [
            cid for batch in details["stages"]["batches"] for cid in batch["claim_ids"]
        ]
        assert len(details["stages"]["batches"]) == 2
        assert groups != ids
        assert Counter(groups) == Counter(ids)
        assert (
            runner._completed_bound_repair(
                arguments["initial_evaluation"], arguments["report_data"]["survey"]
            )
            is False
        )
        assert (
            runner._completed_bound_repair(
                arguments["initial_evaluation"],
                arguments["report_data"]["survey"],
                allow_reordered_batches=True,
            )
            is True
        )
        assert original(runner, **arguments) is True
        result = original(
            runner,
            **(corrupt_observation(arguments, fault) if fault else arguments),
        )
        observed.append(result)
        return result

    monkeypatch.setattr(writing, "_host_unverified_repair_eligible", checked)
    report, gateway = await run_semantic(case, tmp_path, monkeypatch)
    assert observed == [fault is None]
    assert gateway.repair_count == (1 if fault is None else 0)
    assert report["delivery"]["publishable"] is (fault is None)


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [None, "host-support-receipts-v1"])
async def test_unselected_legacy_identity_does_not_gain_semantic_repair(
    case, tmp_path, monkeypatch, revision
):
    """Existing missing and V1 selectors keep their old incomplete-assessment refusal."""
    prepare_case(case)
    report, gateway = await run_semantic(case, tmp_path, monkeypatch, revision=revision)
    assert gateway.repair_count == 0 and not sent_with(gateway, "targets")
    assert report["delivery"]["publishable"] is False
    assert (
        report["evaluation"]["faithfulness"]["details"]["coverage"]["complete"] is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing", "wrong_run", "wrong_allocation", "unreadable"]
)
async def test_current_checkpoint_must_match_actual_run_and_allocation(
    case, tmp_path, monkeypatch, fault
):
    """Only the actual current checkpoint can authenticate replayed observations."""
    import litagent.writing_v2 as writing

    prepare_case(case)
    original = writing._host_unverified_repair_eligible
    observed = []

    def checked(runner, **arguments):
        assert original(runner, **arguments) is True
        allocation = arguments["initial_evaluation"]["faithfulness"]["details"][
            "stages"
        ]["capacity"]["allocation_identity"]
        matches = []
        for path in runner._evaluation_progress_root.glob("*/progress.json"):
            checkpoint = json.loads(path.read_text())
            if (
                checkpoint["progress"].get("capacity", {}).get("allocation_identity")
                == allocation
            ):
                matches.append((path, checkpoint))
        assert len(matches) == 1
        path, checkpoint = matches[0]
        raw = path.read_bytes()
        try:
            if fault == "unreadable":
                read_text = type(path).read_text

                def blocked(selected, *args, **kwargs):
                    if selected == path:
                        raise PermissionError("Current test checkpoint cannot be read.")
                    return read_text(selected, *args, **kwargs)

                with monkeypatch.context() as context:
                    context.setattr(type(path), "read_text", blocked)
                    result = original(runner, **arguments)
            elif fault == "missing":
                path.unlink()
            else:
                if fault == "wrong_run":
                    checkpoint["session_id"] = "another-current-test-run"
                else:
                    checkpoint["progress"]["capacity"]["allocation_identity"] = "0" * 64
                path.write_text(json.dumps(checkpoint))
            if fault != "unreadable":
                result = original(runner, **arguments)
        finally:
            path.write_bytes(raw)
        assert original(runner, **arguments) is True
        observed.append(result)
        return result

    monkeypatch.setattr(writing, "_host_unverified_repair_eligible", checked)
    report, gateway = await run_semantic(case, tmp_path, monkeypatch)
    assert observed == [False]
    assert gateway.repair_count == 0
    assert report["delivery"]["publishable"] is False


def test_previous_v2_is_isolated_while_v1_identity_keeps_its_existing_pin():
    """The new semantic responsibility cannot silently reinterpret prior V2 receipts."""
    from litagent.eval.scoped import support_validation_identity

    assert support_validation_identity("host-support-receipts-v1") == V1_IDENTITY
    assert (
        support_validation_identity("host-support-receipts-v2") != PREVIOUS_V2_IDENTITY
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["partial_prior", "wrong_allocation", "forged_candidate"]
)
async def test_second_attempt_rejects_unrelated_or_unfinished_prior_candidate(
    case, tmp_path, monkeypatch, fault
):
    """An authenticated original does not license a forged failed-candidate history."""
    import litagent.writing_v2 as writing

    prepare_case(case)
    original = writing._host_unverified_repair_eligible
    observed = []

    def checked(runner, **arguments):
        assert original(runner, **arguments) is True
        session = arguments["session"]
        if not session.journal.get("candidates"):
            observed.append(True)
            return True
        copied = copy.copy(session)
        copied.journal = copy.deepcopy(session.journal)
        candidate = copied.journal["candidates"][0]
        if fault == "partial_prior":
            candidate["evaluation"]["faithfulness"]["details"]["stages"][
                "requests"
            ].pop()
        elif fault == "wrong_allocation":
            candidate["evaluation"]["faithfulness"]["details"]["stages"]["capacity"][
                "allocation_identity"
            ] = ("0" * 64)
        else:
            candidate["survey"] += "\nForged candidate text."
        result = original(runner, **{**arguments, "session": copied})
        observed.append(result)
        return result

    monkeypatch.setattr(writing, "_host_unverified_repair_eligible", checked)
    report, gateway = await run_semantic(
        case, tmp_path, monkeypatch, failing_candidates=1
    )
    assert observed == [True, False]
    assert gateway.repair_count == 1 and gateway.inventories == 2
    assert report["delivery"]["publishable"] is False
    assert case.semantic_wrong in report["survey"]
