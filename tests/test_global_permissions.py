"""Bind actual global output permissions without relaxing historical verdicts."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_evaluation_definition_contract import prompt_for
from test_evaluation_dimensions_identity import dimension_envelope
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_global_kind_transport import mixed_global

from litagent.eval import batched
from litagent.eval import decision_output as decisions
from litagent.eval import global_permissions as permissions
from litagent.eval import scoped
from litagent.eval.results import CONTRACT, EvaluationRunResult
from litagent.quality import digest
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_accounting import _envelope as historical_envelope
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writing_flat import restore_global_identities


@pytest.fixture
def global_payload(canonical):
    """Freeze the real current parent and preserve full original report input."""
    value = mixed_global(canonical)
    value["writer_envelope"] = dimension_envelope(evaluation_result_contract=CONTRACT)
    return value


def issue_result(issue):
    """Construct a complete region shell for the unchanged anchor normalizer."""
    return batched.AnchoredAnalysisGlobalReview(
        units=[],
        conflicts=[],
        issues=[issue],
        coverage=[],
        analyses=[],
        limitations={"adequate": False, "claim_ids": [], "explanation": "Control."},
    )


def actual_schema(payload):
    messages, native = scoped.build_evaluation_request(
        prompt_for("global", payload), payload
    )
    return messages, native, native["json_schema"]["schema"]


def historical_parent(payload):
    """Retain the explicit pre-codec parent, whose absence keeps old behavior."""
    value = copy.deepcopy(payload)
    owner = value["writer_envelope"]
    for field in decisions.FIELDS:
        owner.pop(field)
    owner["identity"] = digest({k: v for k, v in owner.items() if k != "identity"})
    validate_writer_envelope(owner)
    return value


@pytest.mark.parametrize("code", list(permissions.ANCHOR_PERMISSIONS))
@pytest.mark.parametrize("anchors", [["v1"], ["a1"], ["v1", "a1"], [], ["v999"]])
def test_native_permissions_agree_with_unchanged_host(global_payload, code, anchors):
    _, _, schema = actual_schema(global_payload)
    item_schema = schema["properties"]["issues"]["items"]
    issue = {"code": code, "anchor_ids": anchors, "explanation": "Located fact."}
    valid = bool(anchors) and "v999" not in anchors
    if code == "missing_assertion" and "a1" in anchors:
        valid = False
    assert (
        Draft202012Validator({"$defs": schema["$defs"], **item_schema}).is_valid(issue)
        is valid
    )
    if valid:
        result = restore_global_identities(issue_result(issue), global_payload)
        located = batched.normalize_global_findings(result, global_payload)
        assert located.issues
        assert all(row.source_unit_id == "u1" for row in located.issues)
        assert located.issues[0].code == code
    else:
        # Unknown aliases are rejected by restoration before normalization.
        with pytest.raises(ValueError, match="(identity|alias)_invalid"):
            result = restore_global_identities(issue_result(issue), global_payload)
            batched.normalize_global_findings(result, global_payload)


@pytest.mark.parametrize(
    "anchors,explanation", [(["v1", "v1"], "Duplicate."), (["v1"], " ")]
)
def test_schema_does_not_replace_host_duplicate_or_explanation_checks(
    global_payload, anchors, explanation
):
    issue = {
        "code": "missing_assertion",
        "anchor_ids": anchors,
        "explanation": explanation,
    }
    value = restore_global_identities(issue_result(issue), global_payload)
    with pytest.raises(ValueError, match="finding_identity_invalid"):
        batched.normalize_global_findings(value, global_payload)


def test_actual_sender_keeps_complete_input_and_measures_final_schema(global_payload):
    before = copy.deepcopy(global_payload)
    messages, native, schema = actual_schema(global_payload)
    Draft202012Validator.check_schema(schema)
    assert len(schema["properties"]["issues"]["items"]["anyOf"]) == 2
    assert strict_request_reservation(messages, 16384, native) > 16384
    assert global_payload == before
    old = historical_parent(before)
    old_messages, old_native, _ = actual_schema(old)
    from test_global_local_input import flatten

    local = json.loads(messages[1]["content"])
    previous = json.loads(old_messages[1]["content"])
    rows = flatten(local)
    assert len(rows) == sum(len(v) for v in previous["assertions_by_kind"].values())
    for kind, entries in previous["assertions_by_kind"].items():
        for entry in entries:
            original = dict(zip(previous["assertion_columns"], entry, strict=True))
            current = next(
                row for row in rows if row["claim_id"] == original["claim_id"]
            )
            assert {key: current[key] for key in original} == original
            assert current["kind"] == kind
    # Other ID fields now gain exact permissions too. The original response
    # grammar and host input remain intact; historical owners retain their schema.
    assert set(schema["properties"]) == set(
        old_native["json_schema"]["schema"]["properties"]
    )
    assert schema["required"] == old_native["json_schema"]["schema"]["required"]
    assert actual_schema(old)[1] == old_native


def test_original_060_findings_remain_rejected():
    path = Path(__file__).parent / "fixtures/run060-global-anchors.json"
    raw = path.read_bytes()
    fixture = json.loads(raw)
    assert fixture["scientific_gold_standard"] is False
    for row in fixture["global_findings_rejected"]:
        assert row["code"] == "missing_assertion"
        result = issue_result(
            {
                "code": row["code"],
                "anchor_ids": row["restored_anchor_ids"],
                "explanation": row["explanation"],
            }
        )
        payload = {
            "source_units": [{"source_unit_id": row["restored_anchor_ids"][0][5:]}],
            "assertions": [
                {
                    "claim_id": anchor[6:],
                    "source_unit_id": row["restored_anchor_ids"][0][5:],
                }
                for anchor in row["restored_anchor_ids"][1:]
            ],
        }
        with pytest.raises(ValueError, match="finding_identity_invalid"):
            batched.normalize_global_findings(result, payload)
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "fault", ["properties", "issues", "items", "code", "anchors", "codes", "stage"]
)
def test_unexpected_producer_schema_fails_before_transmission(global_payload, fault):
    # Capture the real pre-permission format, rather than building a second schema.
    owner = global_payload["writer_envelope"]
    old = historical_parent(global_payload)
    _, native, schema = actual_schema(old)
    issues = schema["properties"]["issues"]
    issue = decisions._resolve(schema, issues["items"])
    if fault == "properties":
        schema["properties"] = []
    elif fault == "issues":
        schema["properties"]["issues"] = None
    elif fault == "items":
        issues["items"] = {"$ref": "#/external"}
    elif fault == "code":
        issue["properties"]["code"] = None
    elif fault == "anchors":
        issue["properties"]["anchor_ids"] = []
    elif fault == "codes":
        issue["properties"]["code"]["enum"].append("non_atomic")
    else:
        global_payload["global_output_contract"] = "wrong-stage"
    assert decisions.selected(owner)
    with pytest.raises(ValueError, match="global_permission_schema_invalid"):
        permissions.project_format(native, global_payload)


def test_old_owner_is_not_silently_upgraded(global_payload):
    old = copy.deepcopy(global_payload)
    old["writer_envelope"] = dimension_envelope()
    marker = {"historical_format": True}
    assert permissions.project_format(marker, old) is marker
    validate_writer_envelope(old["writer_envelope"])


def test_selected_sender_never_silently_falls_back_to_unbound_schema(
    global_payload, monkeypatch
):
    monkeypatch.setattr(scoped, "evaluation_response_format", lambda *_: None)
    with pytest.raises(ValueError, match="global_permission_schema_invalid"):
        scoped.build_evaluation_request(
            prompt_for("global", global_payload), global_payload
        )


def test_060_results_and_usage_remain_readable_but_not_live_resumable():
    path = Path("artifacts/runs/frozen-writer-single-20261104-060-few-shot-metric.json")
    ledger = Path(
        "artifacts/targeted-repairs/frozen-writer-single-20261104-060-few-shot-metric.json.budget-ledger.json"
    )
    if not path.exists() or not ledger.exists():
        pytest.skip("Private real-run artifacts are not distributed")
    raw, budget_raw = path.read_bytes(), ledger.read_bytes()
    report = json.loads(raw)["report"]
    details = report["evaluation"]["faithfulness"]["details"]
    run = EvaluationRunResult.from_details(
        details, hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    assert run is not None
    assert report["evaluation"]["faithfulness"]["score"] is None
    owner = json.loads(budget_raw)["plan"]["envelope"]
    historical_envelope(owner)
    assert owner[decisions.FIELDS[1]] != decisions.contract_identity()
    with pytest.raises(
        ValueError, match="assessment_decision|writer_source_view_policy_invalid"
    ):
        validate_writer_envelope(owner)
    assert path.read_bytes() == raw and ledger.read_bytes() == budget_raw
