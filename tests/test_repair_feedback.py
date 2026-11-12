"""Replay 034 repair feedback through exact host mappings and real consumers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.body_quality import bound_repair_diagnostics
from litagent.capacity_repair import (
    capacity_targets,
    normalize_edit_map,
    validate_edit_records,
)
from litagent.quality import digest
from litagent.safety.planned_budget import EnvelopeExceeded
from litagent.writing_flat import parse as parse_flat
from litagent.writing_v2 import (
    BODY,
    WriterError,
    compile_document,
    patch_provenance,
    repair_feedback_directory,
    repair_request_input,
)


@pytest.fixture
def real_034():
    """Restore a public, losslessly shared fixture without local run dependencies."""
    stored = json.loads(
        (Path(__file__).parent / "fixtures/repair-feedback-034.json").read_text()
    )
    assert stored["storage"] == "lossless-shared-strings-v1"

    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"$shared_string"}:
                return stored["strings"][value["$shared_string"]]
            return {key: restore(child) for key, child in value.items()}
        if isinstance(value, list):
            return [restore(child) for child in value]
        return value

    value = restore(stored["data"])
    assert digest(value) == stored["expanded_sha256"]
    return value


def replay_034(value):
    """Use the received raw repair and compiler to recreate the failed candidate."""
    original = value["original"]
    targets = capacity_targets(
        original["body"],
        value["capacity"],
        maximum=value["second_envelope"]["body_max_units"],
    )
    assert targets == value["targets"]
    _, _, schema = repair_request_input(
        original["packet"],
        original["body"],
        targets,
        original["aliases"],
        cause="capacity",
        previous_failure=None,
        writing_plan=original["generation_writing_plan"],
        envelope=value["candidate_envelope"],
        wire_contract="flat-writer-v1",
        capacity_diagnostic=value["capacity_diagnostic"],
    )
    receipt = value["repair_receipt"]
    parsed = parse_flat(receipt["raw"], schema)
    assert hashlib.sha256(receipt["raw"].encode()).hexdigest() == receipt["raw_sha256"]
    assert digest(schema) == receipt["schema_sha256"]
    assert digest(parsed["canonical"]) == receipt["canonical_sha256"]
    assert digest(parsed["bindings"]) == receipt["wire_bindings_sha256"]
    canonical = normalize_edit_map(parsed["canonical"], targets)
    validate_edit_records(canonical, targets)
    replacements = {row["target_id"]: row["sentences"] for row in canonical["patches"]}
    candidate_body = copy.deepcopy(original["body"])
    for tid in sorted(
        targets, key=lambda key: tuple(targets[key]["path"]), reverse=True
    ):
        section, pi, si = targets[tid]["path"]
        end = targets[tid].get("end", si + 1)
        candidate_body[section][pi]["sentences"][si:end] = replacements[tid]
    for section in BODY:
        candidate_body[section] = [
            paragraph for paragraph in candidate_body[section] if paragraph["sentences"]
        ]
    assert digest(candidate_body) == value["source"]["candidate_body_sha256"]
    provenance = patch_provenance(
        original["body"], targets, replacements, candidate_body, cause="capacity"
    )
    original_survey, original_bindings = compile_document(
        original["body"],
        original["summaries"],
        value["fact_cards"],
        original["aliases"],
        value["language"],
        value["methodology"],
        writing_plan=original["generation_writing_plan"],
        envelope=original["budget_envelope"],
    )
    candidate_survey, candidate_bindings = compile_document(
        candidate_body,
        value["candidate_summary"],
        value["fact_cards"],
        original["aliases"],
        value["language"],
        value["methodology"],
        writing_plan=original["generation_writing_plan"],
        envelope=value["candidate_envelope"],
    )
    assert digest(original_bindings) == value["source"]["original_bindings_sha256"]
    assert digest(candidate_bindings) == value["source"]["candidate_bindings_sha256"]
    assert (
        hashlib.sha256(original_survey.encode()).hexdigest()
        == value["source"]["original_report_sha256"]
    )
    assert (
        hashlib.sha256(candidate_survey.encode()).hexdigest()
        == value["source"]["candidate_report_sha256"]
    )
    prior = {
        "source_report_sha256": value["source"]["original_report_sha256"],
        "report_sha256": value["source"]["candidate_report_sha256"],
        "committed": False,
        "survey": candidate_survey,
        "bindings": candidate_bindings,
        "writer_document": {
            **original,
            "body": candidate_body,
            "summaries": value["candidate_summary"],
            "budget_envelope": value["candidate_envelope"],
        },
        "evaluation": {
            "faithfulness": {"details": value["candidate_evaluation_details"]}
        },
        "repair_provenance": provenance,
        "repair_receipt": {
            "authorized_targets": targets,
            "schema": schema,
            **{
                key: receipt[key]
                for key in ["raw", "raw_sha256", "canonical_sha256", "input_sha256"]
            },
        },
    }
    return original_survey, original_bindings, prior, targets, replacements


def feedback_directory(value):
    """Build a trusted directory from the actual failed patch and both report hashes."""
    survey, bindings, prior, targets, _ = replay_034(value)
    directory = repair_feedback_directory(
        value["original"]["packet"],
        survey,
        value["original"],
        bindings,
        prior,
        value["previous_failure"],
        targets,
    )
    return directory, prior, targets


def test_034_fixture_retains_canonical_diagnostics_receipt_and_candidate(real_034):
    original = copy.deepcopy(real_034)
    failures = bound_repair_diagnostics(real_034["candidate_evaluation_details"])
    assert failures == real_034["previous_failure"]["diagnostics"]
    assert digest(failures) == real_034["source"]["full_candidate_diagnostics_sha256"]
    assert len(failures) == 25
    assert real_034["source"]["candidate_science_counts"] == {
        "factual_claim_count": 73,
        "supported_count": 49,
        "unsupported_count": 9,
        "unverified_count": 15,
    }
    _, _, prior, targets, replacements = replay_034(real_034)
    assert prior["report_sha256"] != prior["source_report_sha256"]
    assert prior["committed"] is False
    assert len(targets) == len(replacements) == 23
    assert real_034 == original


def test_034_old_unmarked_guidance_still_rejects_exact_96996_over_12288(real_034):
    original = real_034["original"]
    assert "repair_feedback_contract" not in real_034["second_envelope"]
    with pytest.raises(EnvelopeExceeded) as caught:
        repair_request_input(
            original["packet"],
            original["body"],
            real_034["targets"],
            original["aliases"],
            cause="capacity",
            previous_failure=real_034["previous_failure"],
            writing_plan=original["generation_writing_plan"],
            envelope=real_034["second_envelope"],
            wire_contract="flat-writer-v1",
            capacity_diagnostic=real_034["capacity_diagnostic"],
        )
    assert caught.value.diagnostics == [
        {
            "stage": "repair",
            "field": "diagnostic_json_bytes",
            "actual": 96996,
            "maximum": 12288,
        }
    ]


@pytest.mark.parametrize(
    "fault",
    ["missing_replacement", "overlap", "bool_range", "original", "candidate"],
)
def test_provenance_rejects_unreplayable_edit_mapping(case, fault):
    body = copy.deepcopy(case.body)
    targets = capacity_targets(body, {})
    replacements = {tid: copy.deepcopy(t["original"]) for tid, t in targets.items()}
    candidate = copy.deepcopy(body)
    tid = next(iter(targets))
    if fault == "missing_replacement":
        replacements.pop(tid)
    elif fault == "overlap":
        targets["overlapping"] = copy.deepcopy(targets[tid])
        replacements["overlapping"] = copy.deepcopy(replacements[tid])
    elif fault == "bool_range":
        targets[tid]["path"][1] = False
    elif fault == "original":
        targets[tid]["original"][0]["text"] += " Changed source."
    else:
        candidate[targets[tid]["path"][0]][0]["sentences"][0][
            "text"
        ] += " Changed candidate."
    with pytest.raises(WriterError, match="writer_feedback_mapping_invalid"):
        patch_provenance(body, targets, replacements, candidate, cause="capacity")


def test_provenance_keeps_same_text_positions_and_compacts_deleted_paragraphs(case):
    body = copy.deepcopy(case.body)
    original_unit = body["methods"][0]["sentences"][0]
    body["methods"].append({"sentences": [copy.deepcopy(original_unit)]})
    body["methods"].append({"sentences": [copy.deepcopy(original_unit)]})
    targets = capacity_targets(body, {})
    replacements = {tid: copy.deepcopy(t["original"]) for tid, t in targets.items()}
    removed = next(tid for tid, t in targets.items() if t["path"][:2] == ["methods", 1])
    divided = next(tid for tid, t in targets.items() if t["path"][:2] == ["methods", 2])
    replacements[removed] = []
    replacements[divided] = [copy.deepcopy(original_unit), copy.deepcopy(original_unit)]
    candidate = copy.deepcopy(body)
    candidate["methods"] = [
        copy.deepcopy(body["methods"][0]),
        {"sentences": copy.deepcopy(replacements[divided])},
    ]
    result = patch_provenance(body, targets, replacements, candidate, cause="capacity")
    origins = [row for row in result["units"] if row["path"][0] == "methods"]
    assert [row["path"] for row in origins] == [
        ["methods", 0, 0],
        ["methods", 1, 0],
        ["methods", 1, 1],
    ]
    assert [row["source_range"] for row in origins] == [
        ["methods", 0, 0, 1],
        ["methods", 2, 0, 1],
        ["methods", 2, 0, 1],
    ]
    assert len({tuple(row["path"]) for row in origins}) == 3
    assert result["identity"] == digest(
        {k: v for k, v in result.items() if k != "identity"}
    )


def test_034_new_feedback_preserves_all_failed_science_and_exact_support(real_034):
    from litagent.eval.scoped import build_support_passages
    from litagent.writing_flat import project_repair_feedback

    directory, prior, targets = feedback_directory(real_034)
    original = copy.deepcopy((real_034, directory, prior, targets))
    packet = real_034["original"]["packet"]
    wire = project_repair_feedback(
        real_034["previous_failure"],
        host_directory=directory,
        packet=packet,
        current_body=real_034["original"]["body"],
        targets=targets,
    )
    assert wire["current_report_sha256"] == real_034["source"]["original_report_sha256"]
    assert wire["prior_report_sha256"] == real_034["source"]["candidate_report_sha256"]
    assert wire["current_body_sha256"] != wire["prior_body_sha256"]
    assert (
        wire["diagnostics_sha256"]
        == real_034["source"]["full_candidate_diagnostics_sha256"]
    )
    assert len(wire["findings"]) == 25 and len(wire["fragments"]) == 20
    cards = packet["cards"]
    by_evidence = {row["evidence_id"]: ref for ref, row in cards.items()}
    passages = build_support_passages(
        {
            eid: {
                "supporting_text": packet["sources"][
                    cards[by_evidence[eid]]["source_ref"]
                ]["source_quote"]
            }
            for eid in directory["evidence_aliases"].values()
        },
        {eid: alias for alias, eid in directory["evidence_aliases"].items()},
        original_only=True,
    )

    def explanation(value):
        return (
            value["explanation"]
            if "explanation" in value
            else wire["explanations"][value["explanation_ref"]]
        )

    def support_signature(sid):
        source = passages[sid]
        return by_evidence[source["evidence_id"]], [
            source["source_start"],
            source["source_end"],
        ]

    def wire_support(ref):
        value = wire["support_refs"][ref]
        return value["card_ref"], value["range"]

    for diagnostic, finding in zip(
        real_034["previous_failure"]["diagnostics"], wire["findings"], strict=True
    ):
        assert finding["claim_id"] == diagnostic["claim_id"]
        fragment = wire["fragments"][finding["fragment_ref"]]
        quote = fragment["text"] + (
            " "
            + " ".join(
                f"[E:{cards[ref]['evidence_id']}]"
                for ref in fragment["cited_card_refs"]
            )
            if fragment["cited_card_refs"]
            else ""
        )
        assert quote == diagnostic["source_quote"]
        claim = finding.get(
            "claim_text",
            wire["fragments"].get(finding.get("claim_text_ref"), {}).get("text"),
        )
        assert claim == diagnostic["claim_text"]
        assert explanation(finding) == diagnostic["explanation"]
        judgment = wire["judgments"][finding["judgment_ref"]]
        assert (
            judgment["kind"] == diagnostic["kind"]
            and judgment["status"] == diagnostic["status"]
        )
        assert judgment.get("atomic", True) is diagnostic["atomic"]
        assert (
            judgment.get("classification_valid", True)
            is diagnostic["classification_valid"]
        )
        assert judgment.get("source_faithful", True) is diagnostic["source_faithful"]
        assert (
            judgment.get("correct_kind", judgment["kind"]) == diagnostic["correct_kind"]
        )
        assert (
            judgment.get("reason_code", judgment["status"]) == diagnostic["reason_code"]
        )
        expected = {row["aspect"]: row for row in diagnostic["aspect_checks"]}
        found = {}
        for state, groups in finding.get("aspects", {}).items():
            for group in groups:
                for name in group["names"]:
                    assert name not in found
                    actual = expected[name]
                    assert actual["supported"] is (state == "supported")
                    assert [wire_support(ref) for ref in group["basis_refs"]] == [
                        support_signature(sid) for sid in actual["support_ids"]
                    ]
                    if not actual["supported"]:
                        assert explanation(group) == actual["explanation"]
                    found[name] = True
        assert set(found) == set(expected)
        assert [wire_support(ref) for ref in finding.get("support_basis_refs", [])] == [
            support_signature(row["support_id"]) for row in diagnostic["support_basis"]
        ]
        validation = diagnostic.get("support_validation")
        if validation is not None:
            actual = finding["support_validation"]
            for key, value in validation.items():
                if key == "spans":
                    assert [wire_support(ref) for ref in actual[key]] == [
                        support_signature(span["support_id"]) for span in value
                    ]
                elif key == "support_id":
                    assert (
                        wire_support(actual["support_ref"])
                        if value is not None
                        else actual["support_ref"]
                    ) == (support_signature(value) if value is not None else None)
                elif key == "actual_evidence_id":
                    assert actual["actual_card_ref"] == by_evidence[value]
                elif key == "expected_evidence_ids":
                    assert actual["expected_card_refs"] == [
                        by_evidence[eid] for eid in value
                    ]
                else:
                    assert actual[key] == value
    assert (real_034, directory, prior, targets) == original


@pytest.mark.parametrize(
    "fault",
    [
        "source_sha",
        "candidate_sha",
        "source_range",
        "source_quote",
        "evidence",
        "raw_receipt",
        "raw_hash",
        "provenance",
        "native_path",
    ],
)
def test_real_directory_rejects_wrong_sha_range_receipt_or_lineage(real_034, fault):
    survey, bindings, prior, targets, _ = replay_034(real_034)
    previous = copy.deepcopy(real_034["previous_failure"])
    if fault == "source_sha":
        prior["source_report_sha256"] = "0" * 64
    elif fault == "candidate_sha":
        prior["report_sha256"] = "0" * 64
    elif fault in {"source_range", "source_quote", "evidence"}:
        entry = prior["evaluation"]["faithfulness"]["details"]["unsupported_claims"][0]
        if fault == "source_range":
            entry["source_start"] += 1
        elif fault == "source_quote":
            entry["source_quote"] += " A forged condition."
        else:
            entry["evidence_ids"] = []
        previous["diagnostics"] = bound_repair_diagnostics(
            prior["evaluation"]["faithfulness"]["details"]
        )
    elif fault == "raw_receipt":
        prior["repair_receipt"]["raw"] = '{"outcome":"declined"}'
        prior["repair_receipt"]["raw_sha256"] = hashlib.sha256(
            prior["repair_receipt"]["raw"].encode()
        ).hexdigest()
    elif fault == "raw_hash":
        prior["repair_receipt"]["raw_sha256"] = "0" * 64
    elif fault == "native_path":
        prior["bindings"]["units"][0]["path"] = copy.deepcopy(
            prior["bindings"]["units"][1]["path"]
        )
    else:
        prior["repair_provenance"]["units"][0]["source_range"][-1] += 1
        prior["repair_provenance"]["identity"] = digest(
            {k: v for k, v in prior["repair_provenance"].items() if k != "identity"}
        )
    with pytest.raises(WriterError):
        repair_feedback_directory(
            real_034["original"]["packet"],
            survey,
            real_034["original"],
            bindings,
            prior,
            previous,
            targets,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_field",
        "unknown_aspect",
        "wrong_source_hash",
        "directory_hash",
        "current_body",
        "unknown_target",
        "support_range",
    ],
)
def test_new_projection_rejects_unknown_variants_or_forged_sources(real_034, fault):
    from litagent.writing_flat import project_repair_feedback

    directory, _, targets = feedback_directory(real_034)
    previous = copy.deepcopy(real_034["previous_failure"])
    packet = copy.deepcopy(real_034["original"]["packet"])
    body = copy.deepcopy(real_034["original"]["body"])
    if fault == "unknown_field":
        previous["diagnostics"][0]["ignore_scientific_error"] = True
    elif fault == "unknown_aspect":
        previous["diagnostics"][0]["aspect_checks"][0]["aspect"] = "unknown"
    elif fault == "wrong_source_hash":
        next(iter(packet["sources"].values()))["content_sha256"] = "0" * 64
    elif fault == "directory_hash":
        directory["identity"] = "0" * 64
    elif fault == "current_body":
        body["methods"][0]["sentences"][0]["text"] += " Changed current body."
    elif fault == "unknown_target":
        next(iter(directory["fragments"].values()))["current_target_ids"] = [
            "not-authorized"
        ]
    else:
        entry = next(
            d
            for d in previous["diagnostics"]
            if d.get("support_validation", {}).get("spans")
        )
        entry["support_validation"]["spans"][0]["source_end"] += 1
    if fault != "directory_hash":
        directory["diagnostics_sha256"] = digest(previous["diagnostics"])
        directory["packet_sha256"] = digest(packet)
        directory["identity"] = digest(
            {k: v for k, v in directory.items() if k != "identity"}
        )
    with pytest.raises(WriterError, match="writer_feedback_(identity|variant)_invalid"):
        project_repair_feedback(
            previous,
            host_directory=directory,
            packet=packet,
            current_body=body,
            targets=targets,
        )


@pytest.mark.parametrize("reject", [False, True])
def test_034_real_request_guard_counts_complete_feedback_and_native_schema(
    real_034, reject
):
    from litagent.safety.budget import strict_request_reservation
    from litagent.safety.planned_budget import CONTEXT_LIMIT
    from litagent.writing_flat import REPAIR_FEEDBACK_CONTRACT
    from litagent.writing_v2 import (
        prepare_planned_input,
        provider_response_format,
        request_messages,
    )

    directory, _, targets = feedback_directory(real_034)
    original = real_034["original"]
    envelope = copy.deepcopy(real_034["second_envelope"])
    envelope["repair_feedback_contract"] = REPAIR_FEEDBACK_CONTRACT
    if reject:
        envelope["request_caps"]["repair"] = 100000
    envelope["identity"] = digest(
        {k: v for k, v in envelope.items() if k != "identity"}
    )
    args = dict(
        cause="capacity",
        previous_failure=real_034["previous_failure"],
        writing_plan=original["generation_writing_plan"],
        envelope=envelope,
        wire_contract="flat-writer-v1",
        capacity_diagnostic=real_034["capacity_diagnostic"],
        feedback_context=directory,
    )
    if reject:
        with pytest.raises(
            WriterError, match="writer_feedback_capacity_exceeded"
        ) as caught:
            repair_request_input(
                original["packet"],
                original["body"],
                targets,
                original["aliases"],
                **args,
            )
        diagnostic = caught.value.diagnostics[0]
        assert diagnostic["request_sent"] is False
        assert diagnostic["finding_count"] == 25
        assert diagnostic["maximum_tokens"] == 100000
        assert diagnostic["reserved_output_tokens"] == 20000
        return
    system, payload, schema = repair_request_input(
        original["packet"], original["body"], targets, original["aliases"], **args
    )
    system, payload = prepare_planned_input(
        system,
        payload,
        original["packet"],
        original["aliases"],
        original["generation_writing_plan"],
        envelope,
    )
    reservation = strict_request_reservation(
        request_messages(system, payload, schema, wire_contract="flat-writer-v1"),
        20000,
        provider_response_format("repair", schema, wire_contract="flat-writer-v1"),
    )
    assert len(payload["previous_local_failure"]["findings"]) == 25
    assert reservation <= min(envelope["request_caps"]["repair"], CONTEXT_LIMIT)
    assert reservation > 110000


async def run_feedback_case(
    case, tmp_path, monkeypatch, failing_candidates, *, config=None
):
    """Mock external HTTP only while retaining the full transactional consumer chain."""
    from model_config_helpers import load_previous_default_config as load_config
    from test_native_responses import owner_for
    from test_scientific_repair_pipeline import ScientificRepairGateway

    from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
    from litagent.output_capacity import configured_output_tokens
    from litagent.review_structure import build_methodology
    from litagent.runner import LitAgent, RunPolicy
    from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN

    gateway = ScientificRepairGateway(case, failing_candidates)
    owner, _, _ = await owner_for(gateway)
    legacy_output = config is None
    config = (
        load_config("config/default.yaml")
        if config is None
        else config.model_copy(deep=True)
    )
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    if legacy_output:
        config.llm.max_tokens = 20000
        config.adversarial.review_max_tokens = 4096
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**kwargs):
        owner._model = kwargs["model"]
        owner._cost_budget = kwargs["cost_budget"]
        owner._trace_hook = kwargs["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {
                "paper_id": item["paper_id"],
                "claims": [item["text"]],
                "evidence_items": [item],
            }
            for item in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            {
                "__search_queries__": {"search_fixture": "relation networks"},
                "search_fixture": list(case.ledger.values()),
                "dedup": list(case.ledger.values()),
                "relevance_gate": list(case.ledger.values()),
            },
            source_outcomes=[
                {
                    "task_id": "search_fixture",
                    "source": "offline-public-fixture",
                    "query": "relation networks",
                    "status": "completed",
                    "result_count": len(case.ledger),
                }
            ],
            started_at=1700000000,
        ),
        "fact_cards_sha256": digest(case.cards),
    }
    before = copy.deepcopy(bundle)
    plan = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        repair_feedback=True,
        **(
            {"output_tokens": configured_output_tokens(config)}
            if not legacy_output
            else {}
        ),
    )
    try:
        report = await run_frozen(
            agent,
            bundle,
            "repair-feedback-offline",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            repair_feedback=True,
        )
        assert bundle == before
        assert not agent._cost_budget.snapshot()["usage_uncertain"]
        return report, gateway
    finally:
        await agent.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [1, 2])
async def test_failed_candidate_feedback_reaches_second_real_sdk_repair_and_full_assessment(
    case, tmp_path, monkeypatch, failing_candidates
):
    report, gateway = await run_feedback_case(
        case, tmp_path, monkeypatch, failing_candidates
    )
    repairs = [row["payload"] for row in gateway.sent if "targets" in row["payload"]]
    assert gateway.repair_count == len(repairs) == 2
    assert gateway.inventories == 3
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 2
    assert repairs[0]["previous_local_failure"] is None
    failure = repairs[1]["previous_local_failure"]
    assert failure["contract"] == "host-repair-feedback-v1"
    assert failure["findings"]
    assert failure["current_report_sha256"] != failure["prior_report_sha256"]
    for payload in repairs:
        current = json.dumps(payload["current_body"])
        assert "FLOPs by 44%" in current
        assert "FLOPs by 45%" not in current
    final = report["evaluation"]["faithfulness"]["details"]
    if failing_candidates == 1:
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


def test_summary_only_failed_patch_keeps_edit_lineage_separate_from_body_premises(
    case, real_034
):
    from litagent.report_render import build_report_source_units
    from litagent.writing_flat import encode, project_repair_feedback
    from litagent.writing_v2 import (
        RANGE_WRITING_PLAN,
        body_units,
        bound_fragments,
        schema_for,
        writer_budget_plan,
    )

    summaries = copy.deepcopy(case.summaries)
    original_unit = copy.deepcopy(summaries["abstract"][0]["sentences"][0])
    targets = {
        "summary_range": {
            "path": ["abstract", 0, 0],
            "end": 1,
            "original": [original_unit],
            "maximum": 2,
            "explanation": "Repair only this abstract range, preserving its scientific premises.",
        }
    }
    replacements = {
        "summary_range": [
            {
                "text": "Relation networks reduce FLOPs by 44%.",
                "body_unit_refs": ["u1"],
            },
            {
                "text": "Relation networks reduce FLOPs by 44%.",
                "body_unit_refs": ["u3"],
            },
        ]
    }
    candidate_summaries = copy.deepcopy(summaries)
    candidate_summaries["abstract"][0]["sentences"] = replacements["summary_range"]
    schema = schema_for("repair", body_units(case.body), targets, summary_repair=True)
    response = {
        "outcome": "patched",
        "explanation": "Controlled first summary repair.",
        "patches": [
            {"target_id": "summary_range", "sentences": replacements["summary_range"]}
        ],
    }
    raw = json.dumps(encode(response, schema), ensure_ascii=False)
    canonical = parse_flat(raw, schema)["canonical"]
    original_survey, original_bindings = compile_document(
        case.body,
        summaries,
        case.cards,
        case.aliases,
        "en",
        {},
        writing_plan=RANGE_WRITING_PLAN,
    )
    candidate_survey, candidate_bindings = compile_document(
        case.body,
        candidate_summaries,
        case.cards,
        case.aliases,
        "en",
        {},
        writing_plan=RANGE_WRITING_PLAN,
    )
    fragments = bound_fragments(candidate_survey, candidate_bindings)
    source_units = {
        unit.unit_id: unit for unit in build_report_source_units(candidate_survey)
    }
    diagnostics = []
    for uid, group in fragments.items():
        for fragment in group:
            if ":native:writer:abstract:" not in fragment["source_fragment_id"]:
                continue
            unit = source_units[uid]
            start, end = unit.start + fragment["start"], unit.start + fragment["end"]
            diagnostic = copy.deepcopy(real_034["previous_failure"]["diagnostics"][0])
            diagnostic.update(
                claim_id=f"summary_claim_{len(diagnostics)}",
                statement_id=f"summary_claim_{len(diagnostics)}",
                source_unit_id=uid,
                source_fragment_id=fragment["source_fragment_id"],
                source_start=start,
                source_end=end,
                source_quote=candidate_survey[start:end],
                evidence_ids=fragment["citation_aliases"],
                section="Abstract",
                claim_text="Relation networks reduce FLOPs by 44%.",
                explanation="The cited complete original contains no numerical FLOPs result.",
                local_error=None,
                atomic=True,
                status="unsupported",
                reason_code="insufficient_support",
                support_basis=[],
                aspect_checks=[],
            )
            diagnostic.pop("support_validation", None)
            diagnostics.append(diagnostic)
    assert len(diagnostics) == 2
    document = {
        "body": case.body,
        "summaries": summaries,
        "aliases": case.aliases,
        "packet": case.packet,
        "generation_writing_plan": RANGE_WRITING_PLAN,
    }
    previous = {
        "code": "writer_candidate_failed",
        "family": "quality",
        "candidate_sha256": hashlib.sha256(candidate_survey.encode()).hexdigest(),
        "diagnostics": diagnostics,
    }
    prior = {
        "source_report_sha256": hashlib.sha256(original_survey.encode()).hexdigest(),
        "report_sha256": previous["candidate_sha256"],
        "committed": False,
        "survey": candidate_survey,
        "bindings": candidate_bindings,
        "writer_document": {**document, "summaries": candidate_summaries},
        "evaluation": {
            "faithfulness": {"details": {"unsupported_claims": diagnostics}}
        },
        "repair_provenance": patch_provenance(
            summaries, targets, replacements, candidate_summaries, cause="evaluation"
        ),
        "repair_receipt": {
            "authorized_targets": targets,
            "schema": schema,
            "raw": raw,
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "canonical_sha256": digest(canonical),
            "input_sha256": digest(targets),
        },
    }
    directory = repair_feedback_directory(
        case.packet,
        original_survey,
        document,
        original_bindings,
        prior,
        previous,
        targets,
    )
    rows = list(directory["fragments"].values())
    assert [row["current_target_ids"] for row in rows] == [
        ["summary_range"],
        ["summary_range"],
    ]
    assert [row["native_paths"] for row in rows] == [
        [["abstract", 0, 0]],
        [["abstract", 0, 1]],
    ]
    assert rows[0]["source_start"] != rows[1]["source_start"]
    expected_premises = [[body_units(case.body)[ref]["path"]] for ref in ["u1", "u3"]]
    assert [row["body_premise_paths"] for row in rows] == expected_premises
    assert directory["current_body_sha256"] == digest(summaries)
    assert directory["prior_body_sha256"] == digest(candidate_summaries)
    wire = project_repair_feedback(
        previous,
        host_directory=directory,
        packet=case.packet,
        current_body=summaries,
        targets=targets,
    )
    assert len(wire["findings"]) == 2
    assert all(
        row["current_targets"] == ["summary_range"]
        for row in wire["fragments"].values()
    )
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        repair_feedback=True,
    )
    assert plan["status"] == "admitted"
    _, payload, second_schema = repair_request_input(
        case.packet,
        summaries,
        targets,
        body_units(case.body),
        cause="evaluation",
        previous_failure=previous,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=plan["envelope"],
        wire_contract="flat-writer-v1",
        summary_body=case.body,
        feedback_context=directory,
    )
    assert len(payload["previous_local_failure"]["findings"]) == 2
    assert payload["summary_source_body_sha256"] == digest(case.body)
    assert (
        "body_unit_refs"
        in second_schema["properties"]["patches"]["items"]["properties"]["sentences"][
            "items"
        ]["properties"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text_kind", ["ascii", "cjk", "escaped"])
async def test_unique_128_by_2000_feedback_stops_before_attempt_reserve_or_http(
    real_034, tmp_path, text_kind
):
    from test_native_responses import owner_for

    from litagent.llm.routing import ModelBoundClient
    from litagent.writing_flat import project_repair_feedback
    from litagent.writing_v2 import (
        EOF_CONTAINER_POLICY,
        RANGE_WRITING_PLAN,
        WriterSession,
        writer_budget_plan,
    )

    survey, bindings, prior, targets, _ = replay_034(real_034)
    template = copy.deepcopy(real_034["previous_failure"]["diagnostics"][0])
    filler = {"ascii": "a", "cjk": "条", "escaped": '"\\\n\t'}[text_kind]
    diagnostics = []
    for index in range(128):
        prefix = f"Unique diagnostic {index:03d}: condition {index}; "
        explanation = (prefix + filler * 2000)[:2000]
        diagnostic = copy.deepcopy(template)
        diagnostic.update(
            claim_id=f"capacity_claim_{index}",
            statement_id=f"capacity_claim_{index}",
            explanation=explanation,
        )
        diagnostics.append(diagnostic)
    assert len({row["explanation"] for row in diagnostics}) == 128
    assert all(len(row["explanation"]) == 2000 for row in diagnostics)
    previous = {
        **real_034["previous_failure"],
        "diagnostics": diagnostics,
    }
    prior["evaluation"] = {
        "faithfulness": {"details": {"unsupported_claims": diagnostics}}
    }
    original = real_034["original"]
    directory = repair_feedback_directory(
        original["packet"], survey, original, bindings, prior, previous, targets
    )
    projection = project_repair_feedback(
        previous,
        host_directory=directory,
        packet=original["packet"],
        current_body=original["body"],
        targets=targets,
    )
    assert len(projection["findings"]) == 128
    for expected, actual in zip(diagnostics, projection["findings"], strict=True):
        assert actual["claim_id"] == expected["claim_id"]
        explanation = actual.get("explanation")
        if explanation is None:
            explanation = projection["explanations"][actual["explanation_ref"]]
        assert explanation == expected["explanation"]

    sent = []

    def unexpected_http(request):
        sent.append(request)
        raise AssertionError("An over-capacity repair must fail before transport.")

    owner, budget, events = await owner_for(unexpected_http)
    plan = writer_budget_plan(
        original["packet"],
        original["aliases"],
        methodology=real_034["methodology"],
        budget_policy="planned-budget-v7",
        repair_feedback=True,
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "repair.json",
        "feedback-capacity-offline",
        original["packet"],
        original["aliases"],
        thinking_mode="disabled",
        provider_api="responses",
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )
    session.journal["failure"] = previous
    before = copy.deepcopy(budget.snapshot())
    try:
        with pytest.raises(
            WriterError, match="writer_feedback_capacity_exceeded"
        ) as caught:
            await session.patch(
                original["body"],
                targets,
                cause="capacity",
                claim_count=128,
                capacity_diagnostic=real_034["capacity_diagnostic"],
                feedback_context=directory,
            )
        diagnostic = caught.value.diagnostics[0]
        assert diagnostic["finding_count"] == 128
        assert diagnostic["reserved_output_tokens"] == 20000
        assert diagnostic["required_tokens"] > diagnostic["maximum_tokens"]
        assert diagnostic["request_sent"] is False
        assert session.state["attempts"] == []
        assert session.journal["calls"] == {}
        assert budget.snapshot() == before
        assert sent == events == []
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "body", [None, [], {"introduction": None}, {"introduction": [{"sentences": None}]}]
)
def test_provenance_invalid_native_containers_raise_located_writer_error(body):
    with pytest.raises(WriterError, match="writer_feedback_mapping_invalid"):
        patch_provenance(body, {}, {}, body, cause="evaluation")
