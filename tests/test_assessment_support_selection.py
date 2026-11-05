"""Keep aspect-owned source selections strict without duplicate model fields."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_claim_aspect_contract import aspect_payload  # noqa: F401
from test_claim_response_contract import permissions  # noqa: F401
from test_claim_response_contract import allowed_for
from test_evaluation_dimensions_contract import nested_reply
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_reason_only_decisions import reason_payload, rehash, wire_reply  # noqa: F401

from litagent.eval import batched
from litagent.eval import decision_output as codec
from litagent.eval import scoped
from litagent.safety.planned_accounting import _envelope as historical_envelope
from litagent.safety.planned_budget import validate_writer_envelope


def decode_on_wire(payload, reply, monkeypatch, *, alias_wire=False):
    """Use the actual sender, decoder and canonical passage restoration."""
    monkeypatch.setattr(
        scoped,
        "strict_request_reservation",
        lambda messages, output, response_format: int(
            ("assessment_input_contract" in json.loads(messages[1]["content"]))
            is not alias_wire
        ),
    )
    messages, native = scoped.build_evaluation_request(
        batched.batch_prompt(payload), payload
    )
    Draft202012Validator(native["json_schema"]["schema"]).validate(reply)
    result = batched.decode_assessment_response(json.dumps(reply), payload)
    return batched.restore_batch_support_ids(result, payload), messages


@pytest.mark.parametrize("alias_wire", [False, True])
def test_distinct_aspect_sources_form_one_exact_host_basis(
    reason_payload, monkeypatch, alias_wire
):
    reply, _, aliases = wire_reply(reason_payload, alias_wire)
    claim = reason_payload["assertions"][0]
    sources = allowed_for(reason_payload, claim)[:2]
    assert len(sources) == 2
    to_wire = {v: k for k, v in (aliases or {}).items()}
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    assert len(checks) >= 2
    checks["predicate"]["support_ids"] = [to_wire.get(sources[0], sources[0])]
    checks["conditions"]["support_ids"] = [to_wire.get(sources[1], sources[1])]
    before, raw = copy.deepcopy(reason_payload), copy.deepcopy(reply)
    result, messages = decode_on_wire(
        reason_payload, reply, monkeypatch, alias_wire=alias_wire
    )
    decision = result.decisions[0]
    assert [s.support_id for s in decision.support_basis] == sources
    assert decision.aspect_checks[0].support_ids == [sources[0]]
    assert decision.aspect_checks[1].support_ids == [sources[1]]
    assert decision.status == "supported"
    assert reason_payload == before and reply == raw
    assert "using selected basis IDs" not in messages[0]["content"]
    assert "from this claim's selected support_basis" not in messages[0]["content"]
    assert (
        "Within each aspect, support_ids must contain no duplicates"
        in messages[0]["content"]
    )
    assert codec.INSTRUCTION.strip() in messages[0]["content"]


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_aspect",
        "missing_aspect",
        "unknown_aspect",
        "duplicate_source",
        "empty_positive",
        "boolean",
        "fields",
    ],
)
def test_structural_faults_have_claim_locations_before_scientific_repair(
    reason_payload, fault
):
    reply, _, _ = wire_reply(reason_payload)
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    if fault == "duplicate_aspect":
        reply["decisions"][0]["evidence_check"]["aspect_checks"] = [
            {"aspect": "predicate", **checks["predicate"]}
        ] * 2
    elif fault == "missing_aspect":
        checks.pop("conditions")
    elif fault == "unknown_aspect":
        checks["plausibility"] = checks.pop("predicate")
    elif fault == "duplicate_source":
        checks["predicate"]["support_ids"] *= 2
    elif fault == "empty_positive":
        checks["predicate"]["support_ids"] = []
    elif fault == "boolean":
        checks["predicate"]["supported"] = 1
    else:
        checks["predicate"]["unrequested"] = "extra"
    with pytest.raises(
        scoped.ScopedEvaluationError, match="support_selection_invalid"
    ) as exc:
        batched.decode_assessment_response(json.dumps(reply), reason_payload)
    assert exc.value.details["claim_id"] == reason_payload["assertions"][0]["claim_id"]
    assert "source_fragment_id" in exc.value.details


@pytest.mark.parametrize("alias_wire", [False, True])
def test_other_claims_source_is_rejected_after_exact_alias_restoration(
    reason_payload, monkeypatch, alias_wire
):
    reply, _, aliases = wire_reply(reason_payload, alias_wire)
    wrong = allowed_for(reason_payload, reason_payload["assertions"][2])[0]
    to_wire = {v: k for k, v in (aliases or {}).items()}
    reply["decisions"][0]["evidence_check"]["aspect_checks"]["predicate"][
        "support_ids"
    ] = [to_wire.get(wrong, wrong)]
    monkeypatch.setattr(
        scoped,
        "strict_request_reservation",
        lambda messages, output, response_format: int(
            ("assessment_input_contract" in json.loads(messages[1]["content"]))
            is not alias_wire
        ),
    )
    result = batched.decode_assessment_response(json.dumps(reply), reason_payload)
    with pytest.raises(
        scoped.ScopedEvaluationError, match="support_selection_invalid"
    ) as exc:
        batched.restore_batch_support_ids(result, reason_payload)
    assert exc.value.details["subreason"] == "support_ownership"


def test_negative_aspects_remain_negative_with_partial_selected_evidence(
    reason_payload, monkeypatch
):
    reply, _, _ = wire_reply(reason_payload)
    evidence = reply["decisions"][0]["evidence_check"]
    evidence["reason_code"] = "insufficient_support"
    evidence["aspect_checks"]["conditions"].update(supported=False, support_ids=[])
    result, _ = decode_on_wire(reason_payload, reply, monkeypatch)
    decision = result.decisions[0]
    assert decision.status == "unsupported"
    assert decision.aspect_checks[1].supported is False
    assert decision.support_basis


def test_basis_order_uses_required_aspect_order_and_deduplicates_only_transmission(
    reason_payload,
):
    claim = reason_payload["assertions"][0]
    reply, _, _ = wire_reply(reason_payload)
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    checks = [{"aspect": key, **value} for key, value in checks.items()]
    sources = allowed_for(reason_payload, claim)[:2]
    checks[0]["support_ids"] = sources[:1]
    checks[1]["support_ids"] = list(reversed(sources))
    expected = [{"support_id": p} for p in sources]
    assert codec._derive_basis(list(reversed(checks)), claim) == expected
    assert len(checks) == 2  # Independent aspect judgments were not merged.


def test_union_above_existing_basis_capacity_is_rejected_without_truncation(
    reason_payload,
):
    claim = reason_payload["assertions"][0]
    reply, _, _ = wire_reply(reason_payload)
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    checks = [{"aspect": key, **value} for key, value in checks.items()]
    checks[0]["support_ids"] = [f"source-{i}" for i in range(16)]
    checks[1]["support_ids"] = ["seventeenth"]
    with pytest.raises(scoped.ScopedEvaluationError) as exc:
        codec._derive_basis(checks, claim)
    assert exc.value.details["subreason"] == "support_capacity"
    assert len(checks[0]["support_ids"]) == 16


def test_old_owner_is_readable_for_costs_but_cannot_select_new_live_decoder(
    reason_payload,
):
    owner = copy.deepcopy(reason_payload["writer_envelope"])
    owner[codec.FIELDS[0]], owner[codec.FIELDS[1]] = "reason-only-decision-v1", "a" * 64
    rehash(owner)
    historical_envelope(owner)
    with pytest.raises(ValueError, match="assessment_decision_contract_invalid"):
        validate_writer_envelope(owner)


def test_run063_original_conflicts_are_not_reclassified_as_success(reason_payload):
    path = Path(__file__).parent / "fixtures/run063-source-selections.json"
    raw = path.read_bytes()
    fixture = json.loads(raw)
    assert fixture["scientific_gold_standard"] is False
    assert len(fixture["decisions"]) == 2
    for old in fixture["decisions"]:
        basis = {b["support_id"] for b in old["support_basis"]}
        assert any(
            not set(a["support_ids"]).issubset(basis) for a in old["aspect_checks"]
        )
        value = nested_reply({"units": [], "decisions": [old], "conflicts": []})
        value["decisions"][0]["evidence_check"].pop("status")
        with pytest.raises(ValueError, match="assessment_decision_response_invalid"):
            codec.decode_response(json.dumps(value), reason_payload)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest()
    )


def test_private_run063_costs_and_failed_scores_still_read_without_migration():
    """Keep original paid receipts and null science results as historical data."""
    from litagent.eval.results import EvaluationRunResult
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    root = Path("artifacts/acceptance/frozen-writer-three-domain-20261105-063")
    if not (root / "state.json").exists():
        pytest.skip("Private real-run accounting is not distributed")
    paths = [root / "state.json", root / "manifest.json"]
    archive = Path("artifacts/runs") / (root.name + "-few-shot-metric.json")
    paths.append(archive)
    raw = {p: p.read_bytes() for p in paths}
    state, manifest = json.loads(raw[paths[0]]), json.loads(raw[paths[1]])
    for row in state["cases"]:
        validate_accounting_snapshot(
            row["budget_snapshot"],
            expected_plan=manifest["budget_plans"][row["case_id"]],
        )
    assert sum(row["requests"] for row in state["cases"]) == 31
    assert sum(row["unknown_hold"] for row in state["cases"]) == 78765
    report = json.loads(raw[archive])["report"]
    result = EvaluationRunResult.from_details(
        report["evaluation"]["faithfulness"]["details"],
        hashlib.sha256(report["survey"].encode()).hexdigest(),
    )
    assert result.metrics()["faithfulness"]["score"] is None
    assert all(p.read_bytes() == content for p, content in raw.items())
