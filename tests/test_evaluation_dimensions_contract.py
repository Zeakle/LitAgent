"""Keep nested report and evidence checks reversible, strict and independently owned."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_claim_aspect_contract import aspect_payload  # noqa: F401
from test_claim_response_contract import permissions  # noqa: F401
from test_claim_response_contract import complete_reply, unbound_native
from test_evaluation_wire_projection import canonical  # noqa: F401

from litagent.eval import batched, dimensions, scoped


def selected_envelope():
    """Build the static selector input without pretending it is a complete owner."""
    return {
        **dimensions._PARENTS,
        "capacity": {},
        dimensions.FIELDS[0]: dimensions.CONTRACT,
        dimensions.FIELDS[1]: dimensions.dimensions_identity(),
    }


@pytest.fixture
def dimensions_payload(aspect_payload):
    """Select dimensions on the existing complete production assessment fixture."""
    payload = copy.deepcopy(aspect_payload)
    payload["writer_envelope"].update(
        {
            dimensions.FIELDS[0]: dimensions.CONTRACT,
            dimensions.FIELDS[1]: dimensions.dimensions_identity(),
        }
    )
    return payload


def native_for(payload, alias_wire=False):
    """Bind the original claim and aspect permissions before the new projection."""
    aliases = batched.batch_support_aliases(payload) if alias_wire else None
    original = batched.bind_assessment_schema(
        unbound_native(payload, aliases), payload, passage_aliases=aliases
    )
    return original, dimensions.project_dimensions_format(original, payload), aliases


def nested_reply(flat):
    """Move known fixture fields without interpreting any diagnostic."""
    value = copy.deepcopy(flat)
    for decision in value["decisions"]:
        fields = dict(decision)
        decision.clear()
        decision["claim_id"] = fields.pop("claim_id")
        decision["report_check"] = {
            key: fields.pop(key)
            for key in (
                "correct_kind",
                "atomic",
                "classification_valid",
                "source_faithful",
                "explanation",
            )
            if key in fields
        }
        decision["evidence_check"] = fields
    return value


def schema_branches(native):
    """Read the actual bound decision groups independently of the projector."""
    schema = native["json_schema"]["schema"]
    item = schema["properties"]["decisions"]["items"]
    branches = item.get("anyOf", [item])
    return [
        schema["$defs"][branch["$ref"].split("/")[-1]] if "$ref" in branch else branch
        for branch in branches
    ]


def check_group(native, branch, field):
    """Resolve a pooled group while preserving its actual native schema."""
    group = branch["properties"][field]
    if "$ref" in group:
        return native["json_schema"]["schema"]["$defs"][group["$ref"].split("/")[-1]]
    return group


def test_none_keeps_original_objects_and_does_not_infer_selection():
    """Missing pairs retain historical values even when other revisions are new."""
    for envelope in (None, {}, {**dimensions._PARENTS, "capacity": {}}):
        payload = {"writer_envelope": envelope}
        raw, native = "unchanged invalid historical JSON", {"arbitrary": object()}
        assert dimensions.selected_dimensions(envelope) is False
        assert dimensions.dimensions_fields(envelope) == {}
        assert dimensions.decode_dimensions_response(raw, payload) is raw
        assert dimensions.project_dimensions_format(native, payload) is native


@pytest.mark.parametrize("field", dimensions.FIELDS)
@pytest.mark.parametrize("bad", [None, False, 0, "", "future", [], {}])
def test_malformed_or_half_pairs_are_rejected(field, bad):
    """Presence of either new field requires its exact authenticated pair."""
    envelope = selected_envelope()
    envelope[field] = bad
    with pytest.raises(ValueError, match="evaluation_dimensions_policy_invalid"):
        dimensions.selected_dimensions(envelope)
    envelope = selected_envelope()
    envelope.pop(field)
    with pytest.raises(ValueError, match="evaluation_dimensions_policy_invalid"):
        dimensions.dimensions_fields(envelope)


@pytest.mark.parametrize("field", [*dimensions._PARENTS, "capacity"])
def test_selection_requires_every_static_parent_without_owner_recursion(field):
    """The selector never manufactures a missing prerequisite."""
    envelope = selected_envelope()
    assert dimensions.selected_dimensions(envelope) is True
    assert dimensions.dimensions_fields(envelope) == {
        key: envelope[key] for key in dimensions.FIELDS
    }
    envelope.pop(field)
    with pytest.raises(ValueError, match="evaluation_dimensions_policy_invalid"):
        dimensions.selected_dimensions(envelope)


def test_source_view_and_other_optional_writer_policies_do_not_select_dimensions():
    """The Evaluation choice stays independent of the writer source view."""
    envelope = selected_envelope()
    envelope.update(writer_source_view="source-only-writer-v1")
    assert dimensions.selected_dimensions(envelope) is True
    for key in dimensions.FIELDS:
        envelope.pop(key)
    assert dimensions.selected_dimensions(envelope) is False


@pytest.mark.parametrize("alias_wire", [False, True])
def test_final_projection_moves_fields_once_and_preserves_all_permissions(
    dimensions_payload, alias_wire
):
    """Both native candidates retain exact original field schemas and refs."""
    original, projected, aliases = native_for(dimensions_payload, alias_wire)
    original_before = copy.deepcopy(original)
    old_schema = original["json_schema"]["schema"]
    new_schema = projected["json_schema"]["schema"]
    Draft202012Validator.check_schema(new_schema)
    for old, new in zip(
        schema_branches(original), schema_branches(projected), strict=True
    ):
        assert set(new["properties"]) == {"claim_id", "report_check", "evidence_check"}
        report = check_group(projected, new, "report_check")
        evidence = check_group(projected, new, "evidence_check")
        assert report["additionalProperties"] is False
        assert evidence["additionalProperties"] is False
        assert {
            "claim_id": new["properties"]["claim_id"],
            **report["properties"],
            **evidence["properties"],
        } == old["properties"]
        assert set(report["required"]) | set(evidence["required"]) | {
            "claim_id"
        } == set(old["required"])
        assert set(report["properties"]) == set(dimensions._REPORT_FIELDS)
        assert set(evidence["properties"]) == set(dimensions._EVIDENCE_FIELDS)
    assert {key: val for key, val in old_schema.items() if key != "$defs"} == {
        key: val for key, val in new_schema.items() if key != "$defs"
    }
    for name, node in old_schema["$defs"].items():
        if node not in schema_branches(original):
            assert new_schema["$defs"][name] == node
    flat = complete_reply(dimensions_payload, original, aliases)
    nested = nested_reply(flat)
    Draft202012Validator(new_schema).validate(nested)
    assert (
        json.loads(
            dimensions.decode_dimensions_response(
                json.dumps(nested), dimensions_payload
            )
        )
        == flat
    )
    assert original == original_before
    assert not Draft202012Validator(new_schema).is_valid(flat)
    assert not Draft202012Validator(old_schema).is_valid(nested)


@pytest.mark.parametrize("alias_wire", [False, True])
def test_projection_preserves_negative_and_cross_claim_source_permissions(
    dimensions_payload, alias_wire
):
    """Nested grouping cannot grant a foreign source or a positive report flag."""
    original, native, aliases = native_for(dimensions_payload, alias_wire)
    flat = complete_reply(dimensions_payload, original, aliases)
    flat["decisions"][0].update(
        source_faithful=False, status="unsupported", reason_code="insufficient_support"
    )
    flat["decisions"][0]["aspect_checks"][0]["supported"] = False
    negative = nested_reply(flat)
    validator = Draft202012Validator(native["json_schema"]["schema"])
    assert validator.is_valid(negative)
    assert (
        json.loads(
            dimensions.decode_dimensions_response(
                json.dumps(negative), dimensions_payload
            )
        )
        == flat
    )
    foreign = copy.deepcopy(negative)
    foreign["decisions"][0]["evidence_check"]["support_basis"] = copy.deepcopy(
        foreign["decisions"][2]["evidence_check"]["support_basis"]
    )
    assert not validator.is_valid(foreign)


@pytest.mark.parametrize(
    "faithful,status,reason",
    [
        (True, "unsupported", "insufficient_support"),
        (True, "unsupported", "uncited_claim"),
        (False, "supported", "supported"),
        (False, "unverifiable", "unverifiable"),
    ],
)
def test_decoder_never_promotes_report_or_evidence_diagnostics(
    dimensions_payload, faithful, status, reason
):
    """Transport controls cover independent dimensions without making gold labels."""
    original, _, _ = native_for(dimensions_payload)
    flat = complete_reply(dimensions_payload, original)
    flat["decisions"][0].update(
        source_faithful=faithful,
        status=status,
        reason_code=reason,
        explanation="Report wording has these settings; this explains report checks only.",
    )
    raw = json.dumps(nested_reply(flat), ensure_ascii=False)
    raw_before = raw
    canonical = dimensions.decode_dimensions_response(raw, dimensions_payload)
    assert json.loads(canonical) == flat
    old_payload = copy.deepcopy(dimensions_payload)
    for key in dimensions.FIELDS:
        old_payload["writer_envelope"].pop(key)
    restored = batched.decode_assessment_response(canonical, old_payload)
    decision = restored.decisions[0]
    assert decision.source_faithful is faithful
    assert (decision.status, decision.reason_code) == (status, reason)
    assert raw == raw_before


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_outer",
        "missing_report",
        "mixed_flat",
        "unknown_report",
        "missing_fidelity",
        "unknown_evidence",
        "missing_aspects",
        "string_bool",
        "integer_fidelity",
        "unknown_kind",
        "unknown_status",
        "wrong_basis_type",
        "extra_basis_field",
        "string_aspect_bool",
        "extra_aspect_field",
        "invalid_unit_bool",
        "extra_root",
    ],
)
def test_decoder_rejects_mixed_missing_extra_and_wrong_typed_shapes(
    dimensions_payload, fault
):
    """Malformed nested responses fail before the canonical decoder sees them."""
    original, _, _ = native_for(dimensions_payload)
    nested = nested_reply(complete_reply(dimensions_payload, original))
    decision = nested["decisions"][0]
    report, evidence = decision["report_check"], decision["evidence_check"]
    if fault == "unknown_outer":
        decision["extra"] = True
    elif fault == "missing_report":
        decision.pop("report_check")
    elif fault == "mixed_flat":
        decision["status"] = "supported"
    elif fault == "unknown_report":
        report["paper_support"] = True
    elif fault == "missing_fidelity":
        report.pop("source_faithful")
    elif fault == "unknown_evidence":
        evidence["source_faithful"] = True
    elif fault == "missing_aspects":
        evidence.pop("aspect_checks")
    elif fault == "string_bool":
        report["atomic"] = "true"
    elif fault == "integer_fidelity":
        report["source_faithful"] = 1
    elif fault == "unknown_kind":
        report["correct_kind"] = "organization"
    elif fault == "unknown_status":
        evidence["status"] = "passed"
    elif fault == "wrong_basis_type":
        evidence["support_basis"] = {}
    elif fault == "extra_basis_field":
        evidence["support_basis"][0]["quote"] = "not a schema field"
    elif fault == "string_aspect_bool":
        evidence["aspect_checks"][0]["supported"] = "false"
    elif fault == "extra_aspect_field":
        evidence["aspect_checks"][0]["reason"] = "duplicate explanation"
    elif fault == "invalid_unit_bool":
        nested["units"][0]["complete"] = 1
    else:
        nested["extra"] = []
    with pytest.raises(ValueError, match="evaluation_dimensions_response_invalid"):
        dimensions.decode_dimensions_response(json.dumps(nested), dimensions_payload)


@pytest.mark.parametrize(
    "raw",
    [
        "null",
        "[]",
        "{not-json",
        '{"units":[],"decisions":[],"conflicts":[],"units":[]}',
        '{"units":[],"decisions":[{"claim_id":"a","claim_id":"b"}],"conflicts":[]}',
        '{"units":[],"decisions":[],"conflicts":[NaN]}',
    ],
)
def test_duplicate_keys_and_non_json_constants_never_disappear(dimensions_payload, raw):
    """Strict parsing retains rejection evidence rather than accepting last-key wins."""
    with pytest.raises(ValueError, match="evaluation_dimensions_response_invalid"):
        dimensions.decode_dimensions_response(raw, dimensions_payload)


@pytest.mark.parametrize(
    "fault",
    ["extra_field", "missing_field", "optional_field", "unknown_modifier", "cycle"],
)
def test_unfamiliar_native_decision_shapes_are_rejected(dimensions_payload, fault):
    """The schema adapter never silently drops a constraint it cannot relocate."""
    original, _, _ = native_for(dimensions_payload)
    node = schema_branches(original)[0]
    if fault == "extra_field":
        node["properties"]["unknown"] = {"type": "boolean"}
        node["required"].append("unknown")
    elif fault == "missing_field":
        node["properties"].pop("source_faithful")
        node["required"].remove("source_faithful")
    elif fault == "optional_field":
        node["required"].remove("atomic")
    elif fault == "unknown_modifier":
        node["dependentRequired"] = {"atomic": ["source_faithful"]}
    else:
        schema = original["json_schema"]["schema"]
        schema["$defs"]["Cycle"] = {"$ref": "#/$defs/Cycle"}
        schema["properties"]["decisions"]["items"] = {"$ref": "#/$defs/Cycle"}
    with pytest.raises(ValueError, match="evaluation_dimensions_format_invalid"):
        dimensions.project_dimensions_format(original, dimensions_payload)


def test_existing_classification_field_is_preserved_when_present_in_original_schema(
    dimensions_payload,
):
    """The representation helper does not invent or remove a legacy primitive."""
    original, _, _ = native_for(dimensions_payload)
    node = schema_branches(original)[0]
    node["properties"]["classification_valid"] = {"type": "boolean"}
    node["required"].append("classification_valid")
    nested = dimensions.project_dimensions_format(original, dimensions_payload)
    report = check_group(nested, schema_branches(nested)[0], "report_check")
    assert report["properties"]["classification_valid"] == {"type": "boolean"}
    assert "classification_valid" in report["required"]


@pytest.mark.parametrize(
    "target",
    [
        dimensions._flatten_decision,
        dimensions._share_dimension_groups,
        batched._decode_assessment_dimensions,
        scoped._dimension_relation_issue,
        scoped._source_entity_forms,
        scoped._protected_entities,
    ],
)
def test_new_identity_binds_instruction_helpers_and_relation_boundary(
    monkeypatch, target
):
    """New representation changes invalidate only their independent contract."""
    from litagent.eval import relation_scopes

    before = dimensions.dimensions_identity()
    support = scoped.support_validation_identity(scoped.SUPPORT_VALIDATION_REVISION_V2)
    response = batched.response_contract_identity(
        contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
    )
    original = dimensions.inspect.getsource

    def changed_source(function):
        source = original(function)
        return source + "\n# new boundary probe\n" if function is target else source

    with monkeypatch.context() as changed:
        changed.setattr(dimensions.inspect, "getsource", changed_source)
        assert dimensions.dimensions_identity() != before
        assert (
            scoped.support_validation_identity(scoped.SUPPORT_VALIDATION_REVISION_V2)
            == support
        )
        assert (
            batched.response_contract_identity(
                contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
            )
            == response
        )
    with monkeypatch.context() as changed:
        changed.setattr(relation_scopes, "relation_scope_identity", lambda: "f" * 64)
        assert dimensions.dimensions_identity() != before
    with monkeypatch.context() as changed:
        changed.setitem(scoped._RELATION_SCOPES, "edge_deployment", "changed scope")
        assert dimensions.dimensions_identity() != before
    with monkeypatch.context() as changed:
        changed.setattr(
            dimensions,
            "DIMENSIONS_INSTRUCTION",
            dimensions.DIMENSIONS_INSTRUCTION + "\nprobe",
        )
        assert dimensions.dimensions_identity() != before


def test_identical_group_sharing_reduces_size_without_changing_validation_space(
    dimensions_payload,
):
    """Sharing complete objects preserves both valid negatives and rejected shapes."""
    original, projected, _ = native_for(dimensions_payload)
    expanded = copy.deepcopy(projected)
    shared_schema = projected["json_schema"]["schema"]
    expanded_schema = expanded["json_schema"]["schema"]
    shared_names = set(expanded_schema["$defs"]) - set(
        original["json_schema"]["schema"]["$defs"]
    )
    assert shared_names
    for branch in schema_branches(expanded):
        for field in ("report_check", "evidence_check"):
            branch["properties"][field] = copy.deepcopy(
                check_group(expanded, branch, field)
            )
    for name in shared_names:
        expanded_schema["$defs"].pop(name)
    assert len(json.dumps(projected, ensure_ascii=False).encode()) < len(
        json.dumps(expanded, ensure_ascii=False).encode()
    )
    validators = [
        Draft202012Validator(schema) for schema in (shared_schema, expanded_schema)
    ]
    positive = nested_reply(complete_reply(dimensions_payload, original))
    negative = copy.deepcopy(positive)
    negative["decisions"][0]["report_check"]["source_faithful"] = False
    negative["decisions"][0]["evidence_check"].update(
        status="unsupported", reason_code="insufficient_support"
    )
    invalid = copy.deepcopy(positive)
    invalid["decisions"][0]["report_check"]["atomic"] = "true"
    for value, expected in ((positive, True), (negative, True), (invalid, False)):
        assert [validator.is_valid(value) for validator in validators] == [
            expected,
            expected,
        ]


@pytest.mark.parametrize(
    "payload",
    [None, [], 3, "text", {"writer_envelope": []}, {"writer_envelope": "text"}],
)
def test_inactive_decoder_keeps_original_malformed_input_error(payload):
    from litagent.eval import batched

    outcomes = []
    for decoder in (
        batched._DECODE_ASSESSMENT_RESPONSE_LEGACY,
        batched.decode_assessment_response,
    ):
        with pytest.raises(Exception) as caught:
            decoder("{}", payload)
        outcomes.append((type(caught.value), str(caught.value)))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("subclass_location", ["payload", "envelope"])
def test_inactive_decoder_does_not_change_old_dict_subclass_semantics(
    subclass_location,
):
    from litagent.eval import batched

    class LegacyDictionary(dict):
        """Represent an accepted historical Mapping implementation."""

    payload = (
        LegacyDictionary(writer_envelope={})
        if subclass_location == "payload"
        else {"writer_envelope": LegacyDictionary()}
    )
    outcomes = []
    for decoder in (
        batched._DECODE_ASSESSMENT_RESPONSE_LEGACY,
        batched.decode_assessment_response,
    ):
        with pytest.raises(Exception) as caught:
            decoder("{}", payload)
        outcomes.append((type(caught.value), str(caught.value)))
    assert outcomes[0] == outcomes[1]
