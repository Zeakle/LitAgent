"""Keep complete local diagnostics while projecting authenticated typed repair work."""

import copy
import hashlib
import json

import pytest
from test_support_receipt_identity import selected_envelope

import litagent.writing_flat as flat
from litagent.capacity_repair import capacity_targets
from litagent.quality import digest
from litagent.writing_v2 import BODY, WriterError


@pytest.fixture
def guidance_case():
    """Build synthetic positional contracts, not a scientific gold standard."""
    claim = "Method A works only under condition X."
    quote = "Method A works only under condition X. A numeric result is 10 percent."
    body = {
        "title": "Typed diagnostic contract",
        **{
            key: [{"sentences": [{"text": claim, "fact_card_refs": ["c1"]}]}]
            for key in BODY
        },
    }
    targets = capacity_targets(body, {"reason": "capacity"}, maximum=66)
    packet = {
        "cards": {
            "c1": {
                "evidence_id": "E1",
                "source_ref": "s1",
                "semantic_status": "not_verified",
                "finding_candidate": "LOCAL-CANDIDATE-ONLY",
                "candidate_fields": {"method": "unverified"},
            }
        },
        "sources": {
            "s1": {
                "source_quote": quote,
                "content_sha256": hashlib.sha256(quote.encode()).hexdigest(),
            }
        },
    }
    text = claim + " [E:E1]"
    fid = "unit-1:native:writer:introduction:0:0"
    fragment = {
        "source_unit_id": "unit-1",
        "text": text,
        "source_start": 10,
        "source_end": 10 + len(text),
        "evidence_ids": ["E1"],
        "native_paths": [["introduction", 0, 0]],
        "body_premise_paths": [],
        "current_target_ids": ["r1"],
    }
    diagnostic = {
        "claim_id": "claim-1",
        "kind": "scientific",
        "claim_text": claim,
        "source_unit_id": "unit-1",
        "source_fragment_id": fid,
        "source_quote": text,
        "source_start": 10,
        "source_end": 10 + len(text),
        "section": "Introduction",
        "evidence_ids": ["E1"],
        "local_error": None,
        "correct_kind": "scientific",
        "atomic": True,
        "classification_valid": True,
        "source_faithful": True,
        "status": "unsupported",
        "reason_code": "insufficient_support",
        "explanation": "LOCAL-ONLY-MAIN-" + "z" * 2000,
        "support_basis": [{"support_id": "e1:full_original"}],
        "aspect_checks": [
            {
                "aspect": "conditions",
                "supported": False,
                "support_ids": ["e1:full_original"],
                "explanation": "LOCAL-ONLY-ASPECT-" + "q" * 2000,
            }
        ],
        "statement_id": "claim-1",
        "alignment_status": "exact",
        "citation_scope": "exact",
        "diagnostic_source": "survey-evaluation-v3",
        "support_validation": {
            "status": "unverified",
            "issue": "support_aspect_coverage_invalid",
            "required_aspects": ["predicate", "conditions"],
            "actual_aspects": ["conditions"],
            "spans": [
                {
                    "support_id": "e1:full_original",
                    "evidence_id": "E1",
                    "source": "original_support",
                    "quote": quote,
                    "source_start": 0,
                    "source_end": len(quote),
                }
            ],
        },
    }
    return body, targets, packet, fid, fragment, diagnostic


def context_for(
    case, diagnostics=None, *, category="evaluation", origin="current", reportless=False
):
    """Model the trusted host directory independently of the wire projection."""
    body, _, packet, fid, fragment, diagnostic = case
    diagnostics = copy.deepcopy(
        diagnostics if diagnostics is not None else [diagnostic]
    )
    fragments = (
        {} if category in {"local", "capacity"} else {fid: copy.deepcopy(fragment)}
    )
    context = {
        "contract": flat.STRUCTURED_REPAIR_GUIDANCE_CONTEXT,
        "origin": origin,
        "current_report_sha256": None if reportless else "a" * 64,
        "current_body_sha256": digest(body),
        "source_report_sha256": (
            None if reportless else ("a" if origin == "current" else "b") * 64
        ),
        "source_body_sha256": digest(body) if origin == "current" else "c" * 64,
        "source_bindings_sha256": None if reportless else "d" * 64,
        "diagnostics_sha256": digest(diagnostics),
        "packet_sha256": digest(packet),
        "evidence_aliases": {"e1": "E1"},
        "fragments": fragments,
        "issues": [
            {
                "diagnostic_index": index,
                "diagnostic_sha256": digest(raw),
                "category": category,
                "fragment_ids": list(fragments),
                "target_ids": ["r1"],
                "body_paths": [["introduction", 0, 0]] if not fragments else [],
            }
            for index, raw in enumerate(diagnostics)
        ],
    }
    context["identity"] = digest(context)
    return {"diagnostics": diagnostics, "host_directory": context}


def rebind(wrapper):
    """Recompute data hashes while leaving semantic and positional guards active."""
    d = wrapper["host_directory"]
    d["diagnostics_sha256"] = digest(wrapper["diagnostics"])
    for index, issue in enumerate(d["issues"]):
        issue["diagnostic_sha256"] = digest(wrapper["diagnostics"][index])
    d["identity"] = digest({k: v for k, v in d.items() if k != "identity"})


def project(case, wrapper):
    body, targets, packet, *_ = case
    return flat.project_structured_repair_guidance(
        wrapper["diagnostics"],
        host_directory=wrapper["host_directory"],
        packet=packet,
        current_body=body,
        targets=targets,
    )


def test_complete_typed_fields_survive_and_all_free_explanations_stay_local(
    guidance_case,
):
    wrapper = context_for(guidance_case)
    before = copy.deepcopy((guidance_case, wrapper))
    result = project(guidance_case, wrapper)
    issue = result["issues"][0]
    assert issue["claim_text"] == guidance_case[-1]["claim_text"]
    assert result["fragments"]["f1"]["text"] == guidance_case[-2]["text"]
    assert issue["status"] == "unsupported"
    assert issue["reason_code"] == "insufficient_support"
    assert issue["aspect_checks"] == [
        {"aspect": "conditions", "supported": False, "support_refs": ["s1"]}
    ]
    assert issue["support_validation"]["required_aspects"] == [
        "predicate",
        "conditions",
    ]
    assert issue["support_validation"]["actual_aspects"] == ["conditions"]
    assert result["support_refs"]["s1"] == {
        "card_ref": "c1",
        "range": [0, len(guidance_case[2]["sources"]["s1"]["source_quote"])],
    }
    assert result["coverage"] == {
        "diagnostics_total": 1,
        "issues_total": 1,
        "complete": True,
    }
    assert "LOCAL-ONLY-" not in json.dumps(result)
    assert (guidance_case, wrapper) == before


@pytest.mark.parametrize(
    "reason",
    ["evaluation_report_cross_section_contradiction", "body_analysis_incomplete"],
)
def test_host_report_diagnostics_keep_supported_status_without_reclassification(
    guidance_case, reason
):
    raw = copy.deepcopy(guidance_case[-1])
    raw.update(status="supported", reason_code=reason, diagnostic_owner="report")
    if reason == "body_analysis_incomplete":
        raw.update(diagnostic_type="analysis_depth", preserve_supported_claim=True)
    result = project(guidance_case, context_for(guidance_case, [raw]))
    assert result["issues"][0]["status"] == "supported"
    assert result["issues"][0]["reason_code"] == reason


def test_reviewer_is_only_a_located_request_for_independent_review(guidance_case):
    raw = {
        "unit_id": "writer:introduction:0:0",
        "explanation": "LOCAL-ONLY-Reviewer claims a numerical error.",
    }
    result = project(
        guidance_case, context_for(guidance_case, [raw], category="reviewer")
    )
    issue = result["issues"][0]
    assert (
        issue["code"] == "reviewer_concern" and issue["need_independent_review"] is True
    )
    assert not {"status", "reason_code", "expected", "actual"} & issue.keys()
    assert "LOCAL-ONLY-" not in json.dumps(result)


@pytest.mark.parametrize("category", ["local", "capacity"])
def test_reportless_host_observations_use_body_paths_without_fabricated_offsets(
    guidance_case, category
):
    raw = {
        "code": "writer_reference_invalid",
        "path": ["introduction", 0, 0],
        "actual": 70,
        "maximum": 66,
        "explanation": "LOCAL-ONLY-host prose",
        "diagnostics": [
            {
                "field": "reference",
                "reference": "bad",
                "explanation": "LOCAL-ONLY-nested",
            }
        ],
    }
    result = project(
        guidance_case,
        context_for(guidance_case, [raw], category=category, reportless=True),
    )
    assert (
        result["current_report_sha256"] is None
        and result["source_bindings_sha256"] is None
    )
    assert result["fragments"] == {}
    assert result["issues"][0]["body_paths"] == [["introduction", 0, 0]]
    assert not {"source_start", "source_end"} & result["issues"][0].keys()
    assert "LOCAL-ONLY-" not in json.dumps(result)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "wrong_source",
        "wrong_offset",
        "missing_fragment",
        "lost_issue",
        "bool_index",
        "unknown_aspect",
        "unknown_validation",
        "extra_raw",
        "no_location",
    ],
)
def test_no_unrecognized_or_unbound_problem_is_silently_filtered(guidance_case, fault):
    wrapper = context_for(guidance_case)
    d, raw = wrapper["host_directory"], wrapper["diagnostics"][0]
    if fault == "unknown":
        d["issues"][0]["category"] = "guessed"
    elif fault == "wrong_source":
        raw["evidence_ids"] = ["unknown"]
    elif fault == "wrong_offset":
        raw["source_start"] += 1
    elif fault == "missing_fragment":
        d["fragments"] = {}
    elif fault == "lost_issue":
        d["issues"] = []
    elif fault == "bool_index":
        d["issues"][0]["diagnostic_index"] = False
    elif fault == "unknown_aspect":
        raw["aspect_checks"][0]["aspect"] = "guessed"
    elif fault == "unknown_validation":
        raw["support_validation"]["guessed"] = "not dropped"
    elif fault == "extra_raw":
        raw["future_field"] = "not dropped"
    elif fault == "no_location":
        d["issues"][0].update(fragment_ids=[], body_paths=[])
        d["fragments"] = {}
    rebind(wrapper)
    before = copy.deepcopy(wrapper)
    with pytest.raises(WriterError, match="writer_guidance_"):
        project(guidance_case, wrapper)
    assert wrapper == before


def selected_scope():
    """Use the actual full-parent envelope producer for the selected repair path."""
    return selected_envelope(
        repair_mapping_contract="capacity-source-map-v3",
        repair_guidance_contract=flat.STRUCTURED_REPAIR_GUIDANCE_CONTRACT,
    )


def request_payload(case, previous=None):
    body, targets, packet, *_ = case
    return {
        **packet,
        "current_body": copy.deepcopy(body),
        "body_sha256": digest(body),
        "targets": {"r1": copy.deepcopy(targets["r1"])},
        "cause": "evaluation",
        "previous_local_failure": previous,
    }


def test_selected_first_request_has_no_target_or_previous_free_prose(guidance_case):
    wrapper = context_for(guidance_case)
    payload = request_payload(
        guidance_case,
        {
            "code": "writer_local_failure",
            "family": "quality",
            "explanation": "LOCAL-ONLY-previous",
        },
    )
    payload["targets"]["r1"]["explanation"] = "LOCAL-ONLY-target"
    before = copy.deepcopy((payload, wrapper))
    result = flat.repair_projection(
        payload,
        selected_scope(),
        packet=guidance_case[2],
        guidance_context={"current": wrapper, "prior_candidate": None},
    )
    assert result["repair_guidance"]["current"]["issues"][0]["target_ids"] == ["r1"]
    assert not {"issue_refs", "task"} & result["targets"]["r1"].keys()
    assert result["cards"] == {
        "c1": {
            "source_ref": "s1",
            "evidence_id": "E1",
            "semantic_status": "not_verified",
        }
    }
    assert result["sources"] == guidance_case[2]["sources"]
    assert "LOCAL-CANDIDATE-ONLY" not in json.dumps(result)
    assert "explanation" not in result["targets"]["r1"]
    assert result["previous_local_failure"] == {
        "code": "writer_local_failure",
        "family": "quality",
        "guidance_ref": "current",
    }
    assert "LOCAL-ONLY-" not in json.dumps(result)
    assert (payload, wrapper) == before


def prior_context(case):
    prior = context_for(case, origin="prior_candidate")
    d = prior["host_directory"]
    old = {
        "contract": flat.REPAIR_FEEDBACK_DIRECTORY,
        "current_report_sha256": d["current_report_sha256"],
        "current_body_sha256": d["current_body_sha256"],
        "prior_report_sha256": d["source_report_sha256"],
        "prior_body_sha256": d["source_body_sha256"],
        "diagnostics_sha256": d["diagnostics_sha256"],
        "packet_sha256": d["packet_sha256"],
        "fragments": d["fragments"],
        "evidence_aliases": d["evidence_aliases"],
        "provenance_identity": "e" * 64,
    }
    old["identity"] = digest(old)
    return prior, old


def test_second_request_keeps_both_complete_issue_origins_and_prior_authentication(
    guidance_case,
):
    current = context_for(guidance_case)
    prior, old = prior_context(guidance_case)
    failure = {
        "code": "writer_candidate_failed",
        "family": "quality",
        "candidate_sha256": "b" * 64,
        "diagnostics": prior["diagnostics"],
    }
    result = flat.repair_projection(
        request_payload(guidance_case, failure),
        selected_scope(),
        packet=guidance_case[2],
        host_directory=old,
        guidance_context={"current": current, "prior_candidate": prior},
    )
    assert result["repair_guidance"]["current"]["issues"][0]["target_ids"] == ["r1"]
    assert result["repair_guidance"]["prior_candidate"]["issues"][0]["target_ids"] == [
        "r1"
    ]
    assert not {"issue_refs", "task"} & result["targets"]["r1"].keys()
    assert result["repair_guidance"]["current"]["coverage"]["diagnostics_total"] == 1
    assert (
        result["repair_guidance"]["prior_candidate"]["coverage"]["diagnostics_total"]
        == 1
    )
    assert "LOCAL-ONLY-" not in json.dumps(result)
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        flat.repair_projection(
            request_payload(guidance_case, failure),
            selected_scope(),
            packet=guidance_case[2],
            guidance_context={"current": current, "prior_candidate": prior},
        )


def test_complete_aggregate_overflow_is_unrepairable_without_subset_or_mutation(
    guidance_case,
):
    raw = copy.deepcopy(guidance_case[-1])
    raw["claim_text"] = "汉" * 12000
    wrapper = context_for(guidance_case, [raw])
    before = copy.deepcopy(wrapper)
    with pytest.raises(
        WriterError, match="writer_guidance_unrepairable_capacity"
    ) as raised:
        flat.repair_projection(
            request_payload(guidance_case),
            selected_scope(),
            packet=guidance_case[2],
            guidance_context={"current": wrapper, "prior_candidate": None},
        )
    assert (
        raised.value.diagnostics[0]["actual"]
        > raised.value.diagnostics[0]["maximum"]
        == 12288
    )
    assert wrapper == before


def test_guidance_identity_does_not_change_existing_source_or_feedback_identities(
    monkeypatch,
):
    before = flat.structured_repair_guidance_identity()
    assert (
        flat.source_map_wire_identity()
        == "03a9471fc5e12065a4624b364dd1bedd5f8edface3df25cf8edca9c731a60929"
    )
    assert (
        flat.repair_feedback_identity()
        == "12dc1548f64557da7f6e9e63da70be509e6d1cdc3261e8ad698569ed6e7755f6"
    )
    monkeypatch.setattr(
        flat,
        "STRUCTURED_REPAIR_GUIDANCE_INSTRUCTION",
        flat.STRUCTURED_REPAIR_GUIDANCE_INSTRUCTION + " New selected rule.",
    )
    assert flat.structured_repair_guidance_identity() != before
    assert (
        flat.source_map_wire_identity()
        == "03a9471fc5e12065a4624b364dd1bedd5f8edface3df25cf8edca9c731a60929"
    )
    assert (
        flat.repair_feedback_identity()
        == "12dc1548f64557da7f6e9e63da70be509e6d1cdc3261e8ad698569ed6e7755f6"
    )


@pytest.mark.parametrize("field", ["actual", "expected"])
def test_local_nested_free_prose_is_unrepairable_instead_of_typed_by_name(
    guidance_case, field
):
    raw = {
        "code": "writer_local_failure",
        field: {"explanation": "Not typed by its parent name."},
    }
    wrapper = context_for(guidance_case, [raw], category="local", reportless=True)
    with pytest.raises(WriterError, match="writer_guidance_unrepairable"):
        project(guidance_case, wrapper)


def test_rehashed_fragment_and_diagnostic_cannot_change_current_body(guidance_case):
    wrapper = context_for(guidance_case)
    d = wrapper["host_directory"]
    fid = next(iter(d["fragments"]))
    d["fragments"][fid]["text"] = "X" + d["fragments"][fid]["text"][1:]
    wrapper["diagnostics"][0]["source_quote"] = d["fragments"][fid]["text"]
    rebind(wrapper)
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        project(guidance_case, wrapper)


@pytest.mark.parametrize("value", [True, -1, "128", 1.0])
def test_capacity_numbers_are_observations_not_coerced_values(guidance_case, value):
    wrapper = context_for(
        guidance_case,
        [
            {
                "code": "capacity_required",
                "actual_assertions": value,
                "target_assertions": 128,
            }
        ],
        category="capacity",
        reportless=True,
    )
    with pytest.raises(WriterError, match="writer_guidance_unrepairable"):
        project(guidance_case, wrapper)


def test_capacity_stage_metadata_is_explicitly_typed_and_preserved(guidance_case):
    from litagent.capacity_evaluation import capacity_diagnostic

    raw = capacity_diagnostic(
        {
            "inventory_assertions": [
                {"source_fragment_id": "f1", "section": "Methods"}
            ],
            "authored_unit_counts": {"body": 66, "summary": 4},
        },
        "capacity_remaining_workflow_unaffordable",
        batches=1,
    )
    raw.update(
        budget_policy="planned-budget-v7",
        source_report_sha256="a" * 64,
        initial_assertions=128,
        effective_assertions=176,
        assessment_reserved_tokens=100000,
        allocation_identity="e" * 64,
        reserved_with_margin=4400000,
        fixed_allowance=4480000,
        known_tokens_at_allocation=40000,
        future_locked_tokens=4300000,
        settled_reservation_savings=10000,
        round={
            "number": 1,
            "report_sha256": "a" * 64,
            "mode": "refine",
            "status": "superseded_unverified",
            "assertions": 176,
            "batches": 11,
            "remaining_repairs": 2,
            "offsets": {"assessment": 0, "global": 0},
            "requests": [
                {
                    "stage": "assessment",
                    "wire_sha256": "f" * 64,
                    "reserved_tokens": 100000,
                    "output_tokens": 16384,
                }
            ],
        },
    )
    result = project(
        guidance_case, context_for(guidance_case, [raw], category="capacity")
    )
    assert result["issues"][0]["host_fields"] == raw
    assert result["issues"][0]["code"] == raw["reason"]
    assert "status" not in result["issues"][0]


def test_unverified_science_is_not_reclassified_as_an_evaluation_error(guidance_case):
    raw = copy.deepcopy(guidance_case[-1])
    for key in ("atomic", "classification_valid", "source_faithful", "correct_kind"):
        raw.pop(key)
    wrapper = context_for(guidance_case, [raw])
    with pytest.raises(WriterError, match="writer_guidance_unrepairable"):
        project(guidance_case, wrapper)


@pytest.mark.parametrize("field", ["source_start", "source_end"])
def test_rehashed_evaluation_offsets_cannot_use_python_numeric_aliases(
    guidance_case, field
):
    wrapper = context_for(guidance_case)
    fragment = next(iter(wrapper["host_directory"]["fragments"].values()))
    if field == "source_start":
        fragment["source_start"] = 0
        fragment["source_end"] = len(fragment["text"])
        wrapper["diagnostics"][0]["source_start"] = False
        wrapper["diagnostics"][0]["source_end"] = fragment["source_end"]
    else:
        wrapper["diagnostics"][0]["source_end"] = float(fragment["source_end"])
    rebind(wrapper)
    with pytest.raises(WriterError, match="writer_guidance_"):
        project(guidance_case, wrapper)


@pytest.mark.parametrize("nested", [False, True])
def test_local_body_path_bool_cannot_alias_zero_even_inside_host_diagnostics(
    guidance_case, nested
):
    bad = {"path": ["introduction", False, 0]}
    raw = {
        "code": "writer_local_contract_issue",
        **({"diagnostics": [bad]} if nested else bad),
    }
    wrapper = context_for(guidance_case, [raw], category="local", reportless=True)
    with pytest.raises(WriterError, match="writer_guidance_"):
        project(guidance_case, wrapper)


def test_prior_authentication_compares_json_types_not_python_numeric_equality(
    guidance_case,
):
    current = context_for(guidance_case)
    prior, old = prior_context(guidance_case)
    fragment = next(iter(prior["host_directory"]["fragments"].values()))
    fragment["source_start"] = 0
    fragment["source_end"] = len(fragment["text"])
    prior["diagnostics"][0].update(source_start=0, source_end=fragment["source_end"])
    rebind(prior)
    old["fragments"] = copy.deepcopy(prior["host_directory"]["fragments"])
    next(iter(old["fragments"].values()))["source_start"] = False
    old["diagnostics_sha256"] = prior["host_directory"]["diagnostics_sha256"]
    old["identity"] = digest({k: v for k, v in old.items() if k != "identity"})
    previous = {
        "code": "writer_candidate_failed",
        "family": "quality",
        "candidate_sha256": "b" * 64,
        "diagnostics": prior["diagnostics"],
    }
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        flat.repair_projection(
            request_payload(guidance_case, previous),
            selected_scope(),
            packet=guidance_case[2],
            host_directory=old,
            guidance_context={"current": current, "prior_candidate": prior},
        )


def test_rehashed_current_ids_cannot_authorize_another_native_section(guidance_case):
    wrapper = context_for(guidance_case)
    d = wrapper["host_directory"]
    next(iter(d["fragments"].values()))["current_target_ids"] = ["r2"]
    d["issues"][0]["target_ids"] = ["r2"]
    rebind(wrapper)
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        flat.project_structured_repair_guidance(
            wrapper["diagnostics"],
            host_directory=d,
            packet=guidance_case[2],
            current_body=guidance_case[0],
            targets={"r2": guidance_case[1]["r2"]},
        )


@pytest.mark.parametrize("wrong_target", [False, True])
def test_current_summary_origins_follow_exact_body_premise_paths(
    guidance_case, wrong_target
):
    wrapper = context_for(guidance_case)
    d = wrapper["host_directory"]
    fragment = next(iter(d["fragments"].values()))
    fragment["native_paths"] = [["abstract", 0, 0]]
    fragment["body_premise_paths"] = [["introduction", 0, 0]]
    if wrong_target:
        fragment["current_target_ids"] = ["r2"]
        d["issues"][0]["target_ids"] = ["r2"]
    rebind(wrapper)
    if wrong_target:
        with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
            project(guidance_case, wrapper)
    else:
        assert project(guidance_case, wrapper)["issues"][0]["target_ids"] == ["r1"]


def test_summary_cannot_use_another_summary_as_a_body_premise(guidance_case):
    wrapper = context_for(guidance_case)
    fragment = next(iter(wrapper["host_directory"]["fragments"].values()))
    fragment["native_paths"] = [["abstract", 0, 0]]
    fragment["body_premise_paths"] = [["conclusion", 0, 0]]
    rebind(wrapper)
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        project(guidance_case, wrapper)


def test_capacity_wire_uses_one_exact_issue_authority_and_one_count_record(
    guidance_case,
):
    raw = {
        "reason": "capacity_remaining_workflow_unaffordable",
        "actual_assertions": 184,
        "target_assertions": 176,
        "target_batches": 11,
        "assertions_by_fragment": {"unit-1:native:writer:introduction:0:0": 3},
    }
    wrapper = context_for(guidance_case, [raw], category="capacity")
    before = copy.deepcopy(wrapper)
    canonical = project(guidance_case, wrapper)
    assert canonical["issues"][0]["body_paths"] == [["introduction", 0, 0]]
    payload = request_payload(guidance_case)
    payload.update(
        cause="capacity",
        capacity_diagnostic={
            "reason": raw["reason"],
            "actual_assertions": 184,
            "target_assertions": 176,
            "target_batches": 11,
            "assertions_by_native_unit": {"writer:introduction:0:0": 3},
        },
    )
    result = flat.repair_projection(
        payload,
        selected_scope(),
        packet=guidance_case[2],
        guidance_context={"current": wrapper, "prior_candidate": None},
    )
    issue = result["repair_guidance"]["current"]["issues"][0]
    assert issue["host_fields"] == raw
    assert issue["target_ids"] == ["r1"] and "body_paths" not in issue
    assert "capacity_diagnostic" not in result
    assert not {"task", "issue_refs", "explanation"} & result["targets"]["r1"].keys()
    assert wrapper == before


def test_capacity_duplicate_mismatch_is_not_silently_removed(guidance_case):
    raw = {
        "reason": "capacity_required",
        "actual_assertions": 184,
        "target_assertions": 176,
    }
    wrapper = context_for(guidance_case, [raw], category="capacity")
    payload = request_payload(guidance_case)
    payload.update(
        cause="capacity", capacity_diagnostic={**raw, "actual_assertions": 183}
    )
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        flat.repair_projection(
            payload,
            selected_scope(),
            packet=guidance_case[2],
            guidance_context={"current": wrapper, "prior_candidate": None},
        )


@pytest.mark.parametrize("field", ["sources", "cards"])
def test_lookup_projection_cannot_sanitize_a_mismatched_input_packet(
    guidance_case, field
):
    wrapper = context_for(guidance_case)
    payload = request_payload(guidance_case)
    payload[field] = {}
    with pytest.raises(WriterError, match="writer_guidance_identity_invalid"):
        flat.repair_projection(
            payload,
            selected_scope(),
            packet=guidance_case[2],
            guidance_context={"current": wrapper, "prior_candidate": None},
        )


def test_legacy_repair_keeps_full_card_and_explanation_bytes(guidance_case):
    payload = request_payload(guidance_case)
    payload["targets"]["r1"]["explanation"] = "LEGACY-EXPLANATION"
    scope = selected_envelope(repair_mapping_contract="capacity-source-map-v3")
    result = flat.repair_projection(payload, scope, packet=guidance_case[2])
    assert result["cards"] == guidance_case[2]["cards"]
    assert result["target_explanations"] == {"g1": "LEGACY-EXPLANATION"}
    assert "repair_guidance" not in result
