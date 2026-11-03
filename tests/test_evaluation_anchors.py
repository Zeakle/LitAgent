"""Verify single-anchor findings, historical refusal and real evaluation consumers."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_writer_v2 import case  # noqa: F401

from litagent.eval.batched import (
    ANCHORED_FINDINGS_CONTRACT,
    GLOBAL_FINDINGS_CONTRACT,
    AnchoredAnalysisGlobalReview,
    AnchoredGlobalIssue,
    LocatedAnalysisGlobalReview,
    global_prompt,
    global_review_type,
    located_findings_enabled,
    normalize_global_findings,
    validate_global_findings,
)
from litagent.writing_flat import restore_global_identities


@pytest.fixture
def original():
    """Retain the real invalid response, its parent identities and report wording."""
    return json.loads(
        (Path(__file__).parent / "fixtures/run028-global-findings.json").read_text()
    )


def payload(original, contract=ANCHORED_FINDINGS_CONTRACT):
    """Use actual host parents without deriving identities from report wording."""
    return {
        "global_findings_contract": contract,
        "body_analysis_contract": "body-analysis-v1",
        "source_units": [{"source_unit_id": uid} for uid in original["units"]],
        "assertions": original["assertions"],
    }


def review(original, issues):
    """Construct independent new-contract output without adapting the old response."""
    value = copy.deepcopy(original["response"])
    value["issues"] = issues
    return AnchoredAnalysisGlobalReview.model_validate(value)


def test_original_028_is_still_rejected_under_both_contracts(original):
    before = copy.deepcopy(original)
    value = LocatedAnalysisGlobalReview.model_validate(original["response"])
    with pytest.raises(ValueError, match="finding_identity_invalid"):
        validate_global_findings(value, payload(original, GLOBAL_FINDINGS_CONTRACT))
    with pytest.raises(ValidationError):
        AnchoredAnalysisGlobalReview.model_validate(original["response"])
    assert original == before
    assert not original["human_gold"]


def test_single_claim_anchor_has_exactly_one_host_parent(original):
    result = review(
        original,
        [
            {
                "code": "source_fidelity",
                "anchor_ids": ["claim:a41"],
                "explanation": "Synthetic extraction defect, not a scientific verdict.",
            }
        ],
    )
    before = result.model_dump_json()
    normalized = normalize_global_findings(result, payload(original))
    validate_global_findings(normalized, payload(original))
    assert [(i.source_unit_id, i.claim_ids, i.owner) for i in normalized.issues] == [
        ("v9", ["a41"], "inventory")
    ]
    assert result.model_dump_json() == before


def test_multi_unit_finding_preserves_identical_text_at_distinct_locations(original):
    claims = original["assertions"]
    first = claims[0]
    second = next(c for c in claims if c["source_unit_id"] != first["source_unit_id"])
    host = copy.deepcopy(payload(original))
    next(c for c in host["assertions"] if c["claim_id"] == second["claim_id"])[
        "claim_text"
    ] = first["claim_text"]
    result = review(
        original,
        [
            {
                "code": "contradiction",
                "anchor_ids": [
                    "claim:" + first["claim_id"],
                    "claim:" + second["claim_id"],
                ],
                "explanation": "Both report locations require inspection.",
            }
        ],
    )
    normalized = normalize_global_findings(result, host)
    validate_global_findings(normalized, host)
    assert [i.source_unit_id for i in normalized.issues] == [
        first["source_unit_id"],
        second["source_unit_id"],
    ]
    assert all(i.owner == "report" for i in normalized.issues)


def test_missing_assertion_uses_original_unit_even_without_a_claim(original):
    result = review(
        original,
        [
            {
                "code": "missing_assertion",
                "anchor_ids": ["unit:v12"],
                "explanation": "A proposition is missing from this unit's inventory.",
            }
        ],
    )
    normalized = normalize_global_findings(result, payload(original))
    validate_global_findings(normalized, payload(original))
    assert normalized.issues[0].source_unit_id == "v12"
    assert normalized.issues[0].claim_ids == []
    assert normalized.issues[0].owner == "inventory"


@pytest.mark.parametrize(
    "code,anchors",
    [
        ("source_fidelity", ["unit:v9"]),
        ("classification", ["unit:v9"]),
        ("non_atomic", ["unit:v9"]),
        ("missing_assertion", ["claim:a41"]),
        ("analysis", ["claim:missing"]),
        ("repetition", ["unit:missing"]),
        ("source_fidelity", []),
        ("repetition", ["claim:a41", "claim:a41"]),
    ],
)
def test_invalid_anchors_never_become_accepted_findings(original, code, anchors):
    result = review(
        original, [{"code": code, "anchor_ids": anchors, "explanation": "bad"}]
    )
    with pytest.raises(ValueError, match="finding_identity_invalid"):
        normalize_global_findings(result, payload(original))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_unit_id", "v12"),
        ("claim_ids", ["a41"]),
        ("owner", "report"),
    ],
)
def test_model_cannot_reintroduce_redundant_identity_fields(field, value):
    with pytest.raises(ValidationError):
        AnchoredGlobalIssue.model_validate(
            {
                "code": "repetition",
                "anchor_ids": ["a41"],
                "explanation": "test",
                field: value,
            }
        )


def test_alias_restore_is_bijective_and_does_not_search_text(original):
    host = payload(original)
    result = review(
        original,
        [
            {
                "code": "contradiction",
                "anchor_ids": ["a41", "v12"],
                "explanation": "test",
            }
        ],
    )
    before = result.model_dump_json()
    restored = restore_global_identities(result, host)
    assert restored.issues[0].anchor_ids == ["claim:a41", "unit:v12"]
    assert result.model_dump_json() == before
    result.issues[0].anchor_ids = ["claim:a41"]
    with pytest.raises(ValueError, match="anchor_alias_invalid"):
        restore_global_identities(result, host)


def test_schema_and_prompt_keep_distinct_fidelity_and_support_responsibilities(
    original,
):
    host = payload(original)
    assert global_review_type(host) is AnchoredAnalysisGlobalReview
    assert (
        global_review_type(payload(original, GLOBAL_FINDINGS_CONTRACT))
        is LocatedAnalysisGlobalReview
    )
    prompt = global_prompt(host)
    assert "Each issue must name an actual source_unit_id" not in prompt
    assert "source_fidelity compares extraction with report prose" in prompt
    fields = AnchoredGlobalIssue.model_json_schema()["properties"]
    assert set(fields) == {"code", "anchor_ids", "explanation"}


@pytest.mark.parametrize("marker", [None, [], {}, 1, "unknown-version"])
def test_invalid_marker_does_not_raise_an_unrelated_type_error(marker):
    assert not located_findings_enabled({"global_findings_contract": marker})


@pytest.mark.parametrize("stage", ["inventory", "assessment", "global"])
def test_duplicate_json_keys_are_never_silently_overwritten(stage):
    from litagent.eval.scoped import _inventory_json

    raw = '{"anchor_ids":["a41"],"anchor_ids":["v12"]}'
    with pytest.raises(ValueError, match=f"evaluation_{stage}_duplicate_key"):
        _inventory_json(raw, stage=stage)
    assert raw == '{"anchor_ids":["a41"],"anchor_ids":["v12"]}'


def test_current_schema_and_instruction_are_bound_to_run_identity(monkeypatch):
    from litagent.eval import batched
    from litagent.workbench import RunOptions

    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v7",
        capacity_policy="adaptive-capacity-v1",
        verification={"judge": False, "jev": False},
    )
    before = options.fingerprint()
    old = options.model_copy(update={"budget_policy": "planned-budget-v6"})
    previous = old.fingerprint()
    identity = batched.global_findings_identity()
    monkeypatch.setattr(batched, "ANCHORED_FINDINGS_INSTRUCTION", "changed instruction")
    assert batched.global_findings_identity() != identity
    assert options.fingerprint() != before
    assert old.fingerprint() == previous


@pytest.mark.asyncio
async def test_historical_v1_still_runs_through_real_consumers(
    case, tmp_path, monkeypatch
):
    import test_located_evaluation as located

    monkeypatch.setattr(
        "litagent.eval.batched.CURRENT_FINDINGS_CONTRACT", GLOBAL_FINDINGS_CONTRACT
    )
    report, _ = await located.run_case(case, tmp_path, monkeypatch, "inventory")
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["global_findings_contract"] == GLOBAL_FINDINGS_CONTRACT
    assert details["inventory_check"]["status"] == "failed"
    assert not report["delivery"]["publishable"]
