"""Exercise real host guards with controlled decisions and unaltered 054 excerpts."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_evaluation_dimensions_identity import dimension_envelope

from litagent.eval import batched, dimensions, scoped


@pytest.fixture
def host_envelope():
    """Use the real frozen envelope constructor rather than a selector mock."""
    return dimension_envelope()


@pytest.fixture
def historical_cases():
    """Load self-contained excerpts without reading private run artifacts."""
    path = Path(__file__).with_name("fixtures") / "evaluation-054-dimensions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["gold"] is False
    return {case["case"]: case for case in value["cases"]}


def controlled_case(text, source, envelope, *, report_text=None):
    """Construct complete owned originals and explicitly controlled model checks."""
    report_text = text if report_text is None else report_text
    claim = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": text,
        "source_unit_id": "unit:0",
        "source_fragment_id": "fragment:0",
        "source_quote": report_text,
        "source_start": 0,
        "source_end": len(report_text),
        "section": "methods",
        "evidence_ids": ["owned"],
        "local_error": None,
    }
    ledger = {"owned": {"text": source, "supporting_text": source}}
    aliases = {"owned": "e1"}
    passages = scoped.build_support_passages(ledger, aliases, original_only=True)
    full = [pid for pid in passages if pid.endswith(":full_original")]
    assert len(full) == 1
    raw = {
        "claim_id": claim["claim_id"],
        "correct_kind": "scientific",
        "atomic": True,
        "source_faithful": True,
        "status": "supported",
        "reason_code": "supported",
        "explanation": "Controlled model fixture; not a scientific gold label.",
        "support_basis": [{"support_id": pid} for pid in full],
        "aspect_checks": [
            {
                "aspect": aspect,
                "supported": True,
                "support_ids": full,
                "explanation": "Controlled aspect, independently checked by the host.",
            }
            for aspect in scoped.required_support_aspects(
                text,
                applicability=True,
                support_validation_revision=envelope["support_validation_revision"],
            )
        ],
    }
    return claim, ledger, aliases, passages, raw


def decode(raw, claim, envelope):
    """Pass controlled native fields through the actual selected decoder."""
    decision = copy.deepcopy(raw)
    if dimensions.selected_dimensions(envelope):
        decision = {
            "claim_id": decision.pop("claim_id"),
            "report_check": {
                key: decision.pop(key)
                for key in ("correct_kind", "atomic", "source_faithful", "explanation")
            },
            "evidence_check": decision,
        }
    content = json.dumps({"units": [], "decisions": [decision], "conflicts": []})
    result = batched.decode_assessment_response(
        content,
        {
            "writer_envelope": envelope,
            "assertions": [{"claim_id": claim["claim_id"], "kind": claim["kind"]}],
        },
    )
    assert result.decisions[0].classification_valid is (
        raw["correct_kind"] == claim["kind"]
    )
    return result.decisions[0]


def check_basis(case, decision, envelope):
    """Exercise every existing strict support guard after relation dispatch."""
    claim, ledger, aliases, passages, _ = case
    return scoped.validate_support_basis(
        claim,
        decision,
        ledger,
        aliases,
        passages=passages,
        original_only=True,
        local_numeric_bindings=True,
        chinese_comparatives=True,
        applicability=True,
        metric_inflections=True,
        protocol_numeric_scope=True,
        support_validation_revision=envelope["support_validation_revision"],
        evaluation_dimensions_envelope=(
            envelope if dimensions.selected_dimensions(envelope) else None
        ),
    )


def check_decision(case, decision, envelope):
    """Run the canonical consumer that withholds support from negative checks."""
    claim, ledger, aliases, passages, _ = case
    return scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases=aliases,
        support_passages=passages,
        context={
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "_independent_classification": True,
            "_support_validation_revision": envelope["support_validation_revision"],
            "_evaluation_dimensions_envelope": (
                envelope if dimensions.selected_dimensions(envelope) else None
            ),
            "_evaluation_progress": {
                "global_findings_contract": "anchored-global-findings-v2"
            },
        },
        strict=True,
    )[0]


def test_parameterization_removes_only_the_selected_early_veto(host_envelope):
    text = "NeRF被参数化为连续5D神经场。"
    source = "NeRF represents a scene as a continuous 5D neural field."
    before = controlled_case(text, source, host_envelope)
    decision = decode(before[-1], before[0], host_envelope)
    assert check_basis(before, decision, host_envelope)["status"] == "passed"
    assert check_decision(before, decision, host_envelope)["status"] == "supported"

    legacy = dimension_envelope(evaluation_dimensions_contract=None)
    old = controlled_case(text, source, legacy)
    old_decision = decode(old[-1], old[0], legacy)
    checked = check_decision(old, old_decision, legacy)
    assert checked["status"] == "unverifiable"
    assert checked["support_validation"]["aspect"] == "parameter_count"


@pytest.mark.parametrize(
    "text,source,issue,aspect",
    [
        (
            "NeRF被参数化为连续6D神经场。",
            "NeRF represents a continuous 5D neural field.",
            "support_number_unverified",
            None,
        ),
        (
            "Model-A has 8M parameters.",
            "Model-A has 8B parameters.",
            "support_relation_unverified",
            "parameter_count",
        ),
        (
            "Model-A has 8M parameters.",
            "Model-A has 9M parameters.",
            "support_relation_unverified",
            "parameter_count",
        ),
        (
            "Model-A被参数化并用于边缘部署。",
            "Model-A represents a continuous neural field.",
            "support_relation_unverified",
            "edge_deployment",
        ),
        (
            "Model-A被参数化并通过 distillation 提升效率。",
            "Model-B uses distillation. Model-A represents a continuous field.",
            "support_relation_unverified",
            "mechanism_subject",
        ),
        (
            "Model-A优于Model-B，并使用参数化表示。",
            "Model-B outperforms Model-A.",
            "support_relation_unverified",
            "comparison_direction",
        ),
        (
            "Model-A在1-shot任务上的accuracy为80%。",
            "Model-A has 80% accuracy on 5-shot tasks.",
            "support_number_unverified",
            None,
        ),
        (
            "Model-A has 8M parameters.",
            "Model-B has 8M parameters.",
            "support_entity_unverified",
            None,
        ),
    ],
)
def test_selected_relation_dispatch_keeps_wrong_science_blocked(
    host_envelope, text, source, issue, aspect
):
    case = controlled_case(text, source, host_envelope)
    raw_before = copy.deepcopy(case[-1])
    decision = decode(case[-1], case[0], host_envelope)
    basis = check_basis(case, decision, host_envelope)
    checked = check_decision(case, decision, host_envelope)
    assert basis["status"] == "unverified" and basis["issue"] == issue
    assert checked["status"] == "unverifiable" and checked["local_error"] == issue
    if aspect:
        assert basis["aspect"] == checked["support_validation"]["aspect"] == aspect
    assert case[-1] == raw_before and decision.status == "supported"


def test_faithful_extraction_does_not_promote_a_negative_paper_check(host_envelope):
    case = controlled_case(
        "Model-A uses distillation.", "Model-A uses attention.", host_envelope
    )
    raw = case[-1]
    raw.update(status="unsupported", reason_code="insufficient_support")
    raw["aspect_checks"][0]["supported"] = False
    before = copy.deepcopy(raw)
    decision = decode(raw, case[0], host_envelope)
    checked = check_decision(case, decision, host_envelope)
    assert decision.source_faithful is True and decision.status == "unsupported"
    assert checked["status"] == "unsupported"
    assert checked["aspect_checks"][0]["supported"] is False
    assert raw == before


@pytest.mark.parametrize(
    "report,extracted,paper",
    [
        (
            "Model-A reduces FLOPs by 30%.",
            "Model-A reduces FLOPs by 60%.",
            "Model-A reduces FLOPs by 60%.",
        ),
        (
            "Model-A is evaluated on 1-shot tasks.",
            "Model-A is evaluated on 5-shot tasks.",
            "Model-A is evaluated on 5-shot tasks.",
        ),
    ],
)
def test_paper_supported_extraction_cannot_override_a_false_report_check(
    host_envelope, report, extracted, paper
):
    case = controlled_case(extracted, paper, host_envelope, report_text=report)
    case[-1]["source_faithful"] = False
    before = copy.deepcopy(case[-1])
    decision = decode(case[-1], case[0], host_envelope)
    assert check_basis(case, decision, host_envelope)["status"] == "passed"
    checked = check_decision(case, decision, host_envelope)
    assert checked["status"] == "unverifiable"
    assert checked["local_error"] == "evaluation_classification_or_source_invalid"
    assert checked["source_faithful"] is False
    assert decision.status == "supported" and decision.source_faithful is False
    assert case[-1] == before


@pytest.mark.parametrize("negative", ["atomic", "classification", "aspect"])
def test_positive_support_cannot_override_other_negative_required_checks(
    host_envelope, negative
):
    text = "NeRF被参数化为连续5D神经场。"
    case = controlled_case(
        text, "NeRF represents a continuous 5D field.", host_envelope
    )
    raw = case[-1]
    if negative == "atomic":
        raw["atomic"] = False
    elif negative == "classification":
        raw["correct_kind"] = "report_structure"
    else:
        raw["aspect_checks"][0]["supported"] = False
    decision = decode(raw, case[0], host_envelope)
    checked = check_decision(case, decision, host_envelope)
    assert checked["status"] == "unverifiable"
    assert (
        checked["local_error"]
        == {
            "atomic": "evaluation_atomic_classification_unverified",
            "classification": "evaluation_classification_or_source_invalid",
            "aspect": "support_aspect_unverified",
        }[negative]
    )


def test_public_054_fixture_preserves_complete_source_hashes(historical_cases):
    assert len(historical_cases) == 3
    for case in historical_cases.values():
        assert case["gold"] is False
        report = case["report"]
        assert (
            hashlib.sha256(report["source_quote"].encode()).hexdigest()
            == report["source_quote_sha256"]
        )
        assert report["source_end"] - report["source_start"] == len(
            report["source_quote"]
        )
        for source in case["own_sources"]:
            assert len(source["source_quote"]) == source["full_source_characters"]
            assert (
                hashlib.sha256(source["source_quote"].encode()).hexdigest()
                == source["full_source_sha256"]
            )


def test_054_parameterization_reproduces_only_the_mechanical_boundary(
    host_envelope, historical_cases
):
    case = historical_cases["nerf_parameterization_vs_parameter_count"]
    before = copy.deepcopy(case)
    quotes = [source["source_quote"] for source in case["own_sources"]]
    assert case["raw_decision"]["status"] == "supported"
    assert case["post_host_status"] == "unverifiable"
    assert scoped._RELATION_SCOPES["parameter_count"] == case["rule_literal"]
    assert scoped._dimension_relation_issue(
        case["claim_text"], quotes, quotes, None
    ) == {
        "issue": case["post_host_local_error"],
        "aspect": case["post_host_aspect"],
    }
    # Absence of this particular veto does not reassess the historical science.
    assert (
        scoped._dimension_relation_issue(
            case["claim_text"], quotes, quotes, host_envelope
        )
        is None
    )
    assert case == before and case["gold"] is False


def test_054_false_fidelity_and_unsupported_raw_are_preserved(historical_cases):
    saved = historical_cases["vit_report_fidelity_vs_paper_support"]
    before = copy.deepcopy(saved)
    legacy = dimension_envelope(evaluation_dimensions_contract=None)
    case = controlled_case(
        saved["claim_text"],
        saved["own_sources"][0]["source_quote"],
        legacy,
        report_text=saved["report"]["source_quote"],
    )
    claim = case[0]
    claim["claim_id"] = saved["claim_id"]
    decision = decode(saved["raw_decision"], claim, legacy)
    assert decision.source_faithful is False and decision.status == "unsupported"
    checked = check_decision(case, decision, legacy)
    assert checked["status"] == saved["post_host_status"] == "unverifiable"
    assert checked["local_error"] == saved["post_host_local_error"]
    assert checked["source_faithful"] is False
    assert saved == before and saved["gold"] is False


def test_054_atomic_flags_do_not_claim_a_new_structural_split(historical_cases):
    saved = historical_cases["vit_atomic_flag_is_not_a_structural_split"]
    before = copy.deepcopy(saved)
    legacy = dimension_envelope(evaluation_dimensions_contract=None)
    flags = []
    for text, raw in zip(saved["claim_texts"], saved["raw_decisions"], strict=True):
        decision = decode(
            raw, {"claim_id": saved["claim_id"], "kind": "scientific"}, legacy
        )
        assert decision.source_faithful is raw["source_faithful"]
        assert decision.status == raw["status"] and text
        flags.append(decision.atomic)
    assert flags == [False, True]
    assert saved["same_fragment_assertion_counts"] == [3, 3]
    assert saved["whole_inventory_counts"] == [185, 185]
    assert saved == before and saved["gold"] is False


def partial_case(envelope):
    """Build one completed batch without declaring the whole report verified."""
    case = controlled_case(
        "NeRF被参数化为连续5D神经场。",
        "NeRF represents a continuous 5D neural field.",
        envelope,
    )
    claim, ledger, aliases, passages, raw = case
    decision = decode(raw, claim, envelope)
    pair = dimensions.dimensions_fields(envelope)
    context = {
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "_support_validation_revision": envelope["support_validation_revision"],
        "_evaluation_dimensions_envelope": envelope if pair else None,
        "_host_validation_inputs": {
            "claims": [claim],
            "ledger": ledger,
            "aliases": aliases,
            "support_passages": passages,
            "support_validation_revision": envelope["support_validation_revision"],
            "support_validation_identity": envelope["support_validation_identity"],
            **pair,
        },
    }
    progress = {
        "support_validation_revision": envelope["support_validation_revision"],
        "support_validation_identity": envelope["support_validation_identity"],
        **pair,
        "batches": [
            {
                "status": "completed",
                "claim_ids": [claim["claim_id"]],
                "decisions": [decision.model_dump()],
            }
        ],
    }
    return context, progress


def test_partial_results_keep_the_same_bound_host_policy(host_envelope):
    context, progress = partial_case(host_envelope)
    result = scoped.partial_assessment(progress, context)
    assert result["overall_valid"] is False
    assert result["supported_count"] == result["processed_assertions"] == 1
    assert result["unverified_count"] == result["unsupported_count"] == 0
    assert dimensions.dimensions_fields(host_envelope).items() <= result.items()
    legacy = dimension_envelope(evaluation_dimensions_contract=None)
    old_context, old_progress = partial_case(legacy)
    old = scoped.partial_assessment(old_progress, old_context)
    assert old["overall_valid"] is False and old["unverified_count"] == 1
    assert not set(dimensions.FIELDS).intersection(old)


@pytest.mark.parametrize("location", ["inputs", "progress", "context"])
@pytest.mark.parametrize("damage", ["missing", "forged", "null"])
def test_partial_dimensions_pair_cannot_be_stripped_or_changed(
    host_envelope, location, damage
):
    context, progress = partial_case(host_envelope)
    target = (
        context["_host_validation_inputs"]
        if location == "inputs"
        else progress if location == "progress" else context
    )
    key = (
        "_evaluation_dimensions_envelope"
        if location == "context"
        else "evaluation_dimensions_identity"
    )
    if damage == "missing":
        target.pop(key)
    elif damage == "null":
        target[key] = None
    else:
        target[key] = {} if location == "context" else "f" * 64
    with pytest.raises(
        ValueError, match="evaluation_partial_dimensions_identity_invalid"
    ):
        scoped.partial_assessment(progress, context)
