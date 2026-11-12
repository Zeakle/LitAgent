"""Bind native response permissions without granting scientific support."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_evaluation_wire_projection import canonical  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.eval import batched, scoped
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import make_writer_envelope


def response_envelope(*, selected=True, revision="host-support-receipts-v2"):
    """Obtain the complete selected envelope from its real host producer."""
    return make_writer_envelope(
        body_units=66,
        assertions=128,
        body_bytes=32000,
        source_identity="a" * 64,
        builder_identity="b" * 64,
        contract="flat-writer-envelope-v1",
        capacity=CapacityConfig(),
        round_budget=True,
        global_findings_contract="anchored-global-findings-v2",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        summary_constraints=True,
        classification_review=True,
        repair_feedback=True,
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
        support_validation_revision=revision,
        **(
            {"evaluation_response_contract": batched.EVALUATION_RESPONSE_CONTRACT}
            if selected
            else {}
        ),
    )


@pytest.fixture
def permissions(canonical):
    """Keep shared and disjoint claim permissions under one cited fragment."""
    value = copy.deepcopy(canonical)
    value["writer_envelope"] = response_envelope()
    value["evaluation_input_contract"] = "flat-evaluation-input-v1"
    value["assertions"] = value["assertions"][:5]
    value["cited_materials"]["e2"] = {"original_support_ref": "s2"}
    value["original_supports"]["s2"] = copy.deepcopy(value["original_supports"]["s1"])
    unit = value["source_units"][0]
    unit["citation_aliases"] = ["e1", "e2"]
    unit["source_fragments"][0].update(
        citation_aliases=["e1", "e2"],
        text="A method has a source-bound limitation. [E:e1] [E:e2]",
    )
    value["support_passages"] = {
        "passage:" + "a" * 64: ["e1", "s1", 0, 29],
        "passage:" + "b" * 64: ["e1", "s1", 0, 8],
        "passage:" + "c" * 64: ["e2", "s2", 0, 29],
    }
    value["assertions"][2]["evidence"] = [
        {"evidence_alias": "e2", "material_ref": "e2"}
    ]
    value["assertions"][3]["evidence"] = []
    value["assertions"][4].update(
        kind="report_structure", evidence=[], required_support_aspects=[]
    )
    value["schema"] = batched.assessment_schema(value)
    return value


def allowed_for(payload, claim):
    """Resolve permissions from the actual citation and passage directories."""
    cited = {entry["evidence_alias"] for entry in claim["evidence"]}
    return [
        pid
        for pid, passage in payload["support_passages"].items()
        if claim["kind"] == "scientific" and passage[0] in cited
    ]


def unbound_native(payload, aliases=None):
    """Use the real native and description producers before the new binder."""
    projected = batched.batch_alias_input(payload) if aliases is not None else payload
    native = scoped.evaluation_response_format(projected)
    native = copy.deepcopy(native)
    native["json_schema"]["schema"] = scoped.evaluation_definition_schema(
        native["json_schema"]["schema"],
        stage="assessment",
        host_kind_agreement=True,
    )
    return native


def complete_reply(payload, native, aliases=None):
    """Build full shape controls without treating their labels as scientific gold."""
    to_wire = {canonical: alias for alias, canonical in (aliases or {}).items()}
    schema = native["json_schema"]["schema"]
    item = schema["properties"]["decisions"]["items"]
    if "anyOf" in item:
        item = item["anyOf"][0]
    if "$ref" in item:
        item = schema["$defs"][item["$ref"].removeprefix("#/$defs/")]
    properties = item["properties"]
    decisions = []
    for claim in payload["assertions"]:
        allowed = allowed_for(payload, claim)
        selected = [to_wire.get(allowed[0], allowed[0])] if allowed else []
        negative = claim["kind"] == "scientific" and not selected
        decision = {
            "claim_id": claim["claim_id"],
            "correct_kind": claim["kind"],
            "atomic": True,
            "source_faithful": True,
            "status": "unverifiable" if negative else "supported",
            "reason_code": "unverifiable" if negative else "supported",
            "explanation": "Controlled schema choice; not a scientific gold label.",
            "support_basis": [{"support_id": pid} for pid in selected],
            "aspect_checks": [
                {
                    "aspect": name,
                    "supported": bool(selected),
                    "support_ids": selected,
                    "explanation": "Controlled aspect shape.",
                }
                for name in claim["required_support_aspects"]
            ],
        }
        if "classification_valid" in properties:
            decision["classification_valid"] = True
        decisions.append(decision)
    return {
        "units": [
            {
                "source_unit_id": unit["source_unit_id"],
                "complete": True,
                "explanation": "All controlled assertions are present.",
            }
            for unit in payload["source_units"]
        ],
        "decisions": decisions,
        "conflicts": [],
    }


@pytest.mark.parametrize("alias_wire", [False, True])
def test_binder_keeps_legal_subsets_and_claims_with_shared_permissions(
    permissions, alias_wire
):
    aliases = batched.batch_support_aliases(permissions) if alias_wire else None
    native = unbound_native(permissions, aliases)
    before = copy.deepcopy((native, permissions, aliases))
    selected = batched.bind_assessment_schema(
        native, permissions, passage_aliases=aliases
    )
    reply = complete_reply(permissions, selected, aliases)
    Draft202012Validator.check_schema(selected["json_schema"]["schema"])
    Draft202012Validator(selected["json_schema"]["schema"]).validate(reply)
    assert (
        reply["decisions"][0]["support_basis"] == reply["decisions"][1]["support_basis"]
    )
    assert len(reply["decisions"][0]["support_basis"]) == 1
    assert len(allowed_for(permissions, permissions["assertions"][0])) == 2
    assert selected is not native
    assert (native, permissions, aliases) == before
    assert batched.selected_response_contract(permissions["writer_envelope"])
    assert (
        permissions["writer_envelope"]["evaluation_response_identity"]
        == batched.response_contract_identity()
    )


@pytest.mark.parametrize("alias_wire", [False, True])
@pytest.mark.parametrize("location", ["basis", "aspect", "mixed_basis", "mixed_aspect"])
def test_native_schema_rejects_other_claim_passages_in_both_response_locations(
    permissions, alias_wire, location
):
    aliases = batched.batch_support_aliases(permissions) if alias_wire else None
    selected = batched.bind_assessment_schema(
        unbound_native(permissions, aliases),
        permissions,
        passage_aliases=aliases,
    )
    reply = complete_reply(permissions, selected, aliases)
    validator = Draft202012Validator(selected["json_schema"]["schema"])
    validator.validate(reply)
    foreign = reply["decisions"][2]["support_basis"][0]["support_id"]
    original = copy.deepcopy(reply)
    claim = reply["decisions"][0]
    if "basis" in location:
        if location == "basis":
            claim["support_basis"] = []
        claim["support_basis"].append({"support_id": foreign})
    else:
        if location == "aspect":
            claim["aspect_checks"][0]["support_ids"] = []
        claim["aspect_checks"][0]["support_ids"].append(foreign)
    assert list(validator.iter_errors(reply))
    assert original["decisions"][2] == reply["decisions"][2]
    assert foreign in {
        selection["support_id"]
        for row in original["decisions"]
        for selection in row["support_basis"]
    }


@pytest.mark.parametrize("status", ["unsupported", "unverifiable"])
def test_native_schema_keeps_explicit_scientific_negative_with_legal_basis(
    permissions, status
):
    selected = batched.bind_assessment_schema(unbound_native(permissions), permissions)
    reply = complete_reply(permissions, selected)
    row = reply["decisions"][0]
    row.update(
        status=status,
        reason_code=(
            "insufficient_support" if status == "unsupported" else "unverifiable"
        ),
    )
    for aspect in row["aspect_checks"]:
        aspect.update(supported=False, support_ids=[])
    Draft202012Validator(selected["json_schema"]["schema"]).validate(reply)
    assert row["support_basis"]
    assert row["status"] == status
    assert all(not aspect["supported"] for aspect in row["aspect_checks"])


@pytest.mark.parametrize("index", [3, 4])
@pytest.mark.parametrize("location", ["basis", "aspect"])
def test_empty_allowlist_never_gains_batch_permission_or_an_empty_enum(
    permissions, index, location
):
    selected = batched.bind_assessment_schema(unbound_native(permissions), permissions)
    reply = complete_reply(permissions, selected)
    validator = Draft202012Validator(selected["json_schema"]["schema"])
    validator.validate(reply)
    row = reply["decisions"][index]
    assert not row["support_basis"]
    foreign = reply["decisions"][0]["support_basis"][0]["support_id"]
    if location == "basis":
        row["support_basis"] = [{"support_id": foreign}]
    else:
        if not row["aspect_checks"]:
            row["aspect_checks"] = [
                copy.deepcopy(reply["decisions"][0]["aspect_checks"][0])
            ]
        row["aspect_checks"][0]["support_ids"] = [foreign]
    assert list(validator.iter_errors(reply))
    assert '"enum": []' not in json.dumps(selected)


@pytest.mark.parametrize(
    "selection", [None, "host-support-receipts-v1", "host-support-receipts-v2"]
)
def test_unselected_response_binding_is_an_exact_noop(permissions, selection):
    payload = copy.deepcopy(permissions)
    if selection is None:
        payload.pop("writer_envelope")
    else:
        payload["writer_envelope"] = response_envelope(
            selected=False, revision=selection
        )
    payload["schema"] = batched.assessment_schema(payload)
    native = scoped.evaluation_response_format(payload)
    before = copy.deepcopy((native, payload))
    result = batched.bind_assessment_schema(native, payload)
    assert result is native
    assert (native, payload) == before
    assert not batched.selected_response_contract(payload.get("writer_envelope"))


def test_both_actual_cost_candidates_are_bound_before_reservation(
    permissions, monkeypatch
):
    measured = []
    original = scoped.strict_request_reservation
    before = copy.deepcopy(permissions)

    def measure(messages, output, native):
        cost = original(messages, output, native)
        measured.append((copy.deepcopy(messages), copy.deepcopy(native), cost))
        return cost

    monkeypatch.setattr(scoped, "strict_request_reservation", measure)
    messages, native = scoped.build_evaluation_request(
        batched.batch_prompt(permissions), permissions
    )
    assert len(measured) == 2
    aliases = batched.batch_support_aliases(permissions)
    seen_namespaces = []
    for candidate_messages, candidate_native, _ in measured:
        wire = json.loads(candidate_messages[1]["content"])
        names = set(wire["support_passages"])
        if names == set(permissions["support_passages"]):
            selected_aliases = None
            seen_namespaces.append("canonical")
        else:
            assert names == set(aliases)
            selected_aliases = aliases
            seen_namespaces.append("alias")
        schema = candidate_native["json_schema"]["schema"]
        assert schema["description"].startswith("[atomic] ")
        reply = complete_reply(permissions, candidate_native, selected_aliases)
        validator = Draft202012Validator(schema)
        validator.validate(reply)
        foreign = reply["decisions"][2]["support_basis"][0]["support_id"]
        reply["decisions"][0]["support_basis"] = [{"support_id": foreign}]
        assert list(validator.iter_errors(reply))
        assert wire["original_supports"] == permissions["original_supports"]
    assert set(seen_namespaces) == {"canonical", "alias"}
    chosen = strict_request_reservation(messages, 0, native)
    assert chosen == min(value[2] for value in measured)
    assert (messages, native) in [(value[0], value[1]) for value in measured]
    assert strict_request_reservation(messages, 16384, native) == chosen + 16384
    assert permissions == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_response_identity", "0" * 64),
        ("evaluation_response_contract", "host-bound-evaluation-future"),
        ("support_validation_revision", "host-support-receipts-v1"),
        ("transport_revision", "host-lossless-transport-v0"),
        ("contract", "grounded-writer-envelope-v1"),
        ("capacity", None),
    ],
)
def test_bound_response_rejects_wrong_identity_or_parent_before_changing_schema(
    permissions, field, value
):
    native = unbound_native(permissions)
    invalid = copy.deepcopy(permissions)
    invalid["writer_envelope"][field] = value
    before = copy.deepcopy((native, invalid))
    with pytest.raises(ValueError):
        batched.selected_response_contract(invalid["writer_envelope"])
    with pytest.raises(ValueError):
        batched.bind_assessment_schema(native, invalid)
    assert (native, invalid) == before


@pytest.mark.parametrize(
    "field", ["evaluation_response_contract", "evaluation_response_identity"]
)
def test_bound_response_pair_cannot_be_half_selected(permissions, field):
    native = unbound_native(permissions)
    invalid = copy.deepcopy(permissions)
    invalid["writer_envelope"].pop(field)
    before = copy.deepcopy((native, invalid))
    with pytest.raises(ValueError, match="evaluation_response_contract_invalid"):
        batched.bind_assessment_schema(native, invalid)
    assert (native, invalid) == before


def located_global(
    permissions, *, complete, neighbor_complete=False, code="missing_assertion"
):
    """Use canonical typed findings at the actual post-alias validation boundary."""
    payload = copy.deepcopy(permissions)
    payload["global_findings_contract"] = "anchored-global-findings-v2"
    unit_id = payload["source_units"][0]["source_unit_id"]
    payload["source_units"].append(
        {
            "source_unit_id": "neighbor-unit",
            "citation_aliases": [],
            "source_fragments": [
                {
                    "source_fragment_id": "neighbor-fragment",
                    "text": "Independent adjacent report prose.",
                    "citation_aliases": [],
                }
            ],
        }
    )
    claim_ids = (
        []
        if code in {None, "missing_assertion", "repetition"}
        else [payload["assertions"][0]["claim_id"]]
    )
    result = batched.LocatedAnalysisGlobalReview.model_validate(
        {
            "units": [
                {
                    "source_unit_id": unit_id,
                    "complete": complete,
                    "explanation": "Controlled own unit decision.",
                },
                {
                    "source_unit_id": "neighbor-unit",
                    "complete": neighbor_complete,
                    "explanation": "Independent adjacent unit decision.",
                },
            ],
            "conflicts": [],
            "issues": (
                []
                if code is None
                else [
                    {
                        "source_unit_id": unit_id,
                        "claim_ids": claim_ids,
                        "code": code,
                        "explanation": "Original unresolved finding; never rewritten.",
                    }
                ]
            ),
            "coverage": [],
            "limitations": {
                "adequate": False,
                "claim_ids": [],
                "explanation": "No limitations are asserted by this shape fixture.",
            },
            "analyses": [],
        }
    )
    return payload, result


@pytest.mark.parametrize(
    "complete,code",
    [
        (False, "missing_assertion"),
        (True, None),
        (True, "repetition"),
        (True, "non_atomic"),
    ],
)
def test_global_guard_keeps_valid_missing_and_other_findings_without_rewriting(
    permissions, complete, code
):
    payload, result = located_global(permissions, complete=complete, code=code)
    before = result.model_dump_json()
    payload_before = copy.deepcopy(payload)
    batched.validate_global_findings(result, payload)
    assert result.model_dump_json() == before
    assert payload == payload_before
    if code:
        assert len(result.issues) == 1 and result.issues[0].code == code
    if code == "missing_assertion":
        assert result.issues[0].owner == "inventory"
        assert result.units[0].complete is False


@pytest.mark.parametrize("reverse_order", [False, True])
def test_global_missing_cannot_borrow_incompleteness_from_an_adjacent_unit(
    permissions, reverse_order
):
    payload, result = located_global(
        permissions, complete=True, neighbor_complete=False
    )
    if reverse_order:
        result.units.reverse()
    before = result.model_dump_json()
    with pytest.raises(ValueError, match="^evaluation_global_completeness_conflict$"):
        batched.validate_global_findings(result, payload)
    assert result.model_dump_json() == before
    assert result.issues[0].code == "missing_assertion"
    assert any(
        unit.source_unit_id == "neighbor-unit" and unit.complete is False
        for unit in result.units
    )


def test_global_missing_uses_its_own_false_decision_even_when_neighbor_is_complete(
    permissions,
):
    payload, result = located_global(
        permissions, complete=False, neighbor_complete=True
    )
    before = result.model_dump_json()
    batched.validate_global_findings(result, payload)
    assert result.model_dump_json() == before


@pytest.mark.parametrize("has_envelope", [False, True])
def test_unselected_global_keeps_original_missing_and_complete_true(
    permissions, has_envelope
):
    payload, result = located_global(permissions, complete=True)
    if has_envelope:
        payload["writer_envelope"] = response_envelope(selected=False)
    else:
        payload.pop("writer_envelope")
    before = result.model_dump_json()
    batched.validate_global_findings(result, payload)
    assert result.model_dump_json() == before
    assert result.units[0].complete is True
    assert result.issues[0].code == "missing_assertion"


@pytest.mark.parametrize(
    "fault", ["arbitrary_keys", "permutation", "incomplete", "duplicate_values"]
)
def test_response_alias_map_must_be_the_actual_complete_batch_directory(
    permissions, fault
):
    aliases = batched.batch_support_aliases(permissions)
    if fault == "arbitrary_keys":
        aliases = {
            "invented-" + alias: canonical for alias, canonical in aliases.items()
        }
    elif fault == "permutation":
        aliases["p1"], aliases["p3"] = aliases["p3"], aliases["p1"]
    elif fault == "incomplete":
        aliases.pop("p3")
    else:
        aliases["p2"] = aliases["p1"]
    native = unbound_native(permissions)
    before = copy.deepcopy((native, permissions, aliases))
    with pytest.raises(
        ValueError, match="^evaluation_response_alias_identity_invalid$"
    ):
        batched.bind_assessment_schema(native, permissions, passage_aliases=aliases)
    assert (native, permissions, aliases) == before


@pytest.mark.parametrize("fault", ["missing", "unknown"])
def test_selected_actual_builder_rejects_unbound_canonical_fallback(permissions, fault):
    invalid = copy.deepcopy(permissions)
    if fault == "missing":
        invalid.pop("assessment_input_contract")
    else:
        invalid["assessment_input_contract"] = "adaptive-input-future"
    before = copy.deepcopy(invalid)
    with pytest.raises(
        ValueError, match="^evaluation_response_input_contract_invalid$"
    ):
        scoped.build_evaluation_request(batched.batch_prompt(invalid), invalid)
    assert invalid == before
