"""Keep native aspect permissions claim-owned without replacing scientific checks."""

import copy
import hashlib
import inspect
import json

import pytest
from jsonschema import Draft202012Validator
from test_claim_response_contract import permissions  # noqa: F401
from test_claim_response_contract import (
    complete_reply,
    response_envelope,
    unbound_native,
)
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_support_passages import selected

from litagent.capacity import CapacityConfig
from litagent.eval import batched, scoped
from litagent.native_wire import inline_local_schema
from litagent.safety.planned_budget import make_writer_envelope

V1_IDENTITY = "686e24fa9b9016c0174789d898dc88dad9bb6de6790abd72e77ba9402ed0c2b5"
LEGACY_SOURCE_HASHES = (
    "b2017978dfc5a84f64467ccf40e2321d84b0d16742fe6fff1621495bf1e83e58",
    "930e7d05a7a19251f812baddca3ecab971c178ac077b5abef6fc1e99b0bd9733",
    "f31cdcc4491dfc35cc555c0f9685386d92833bbf357e8dea3245a12d437d9c03",
    "501f3505c6656260fab1403a62f7280e3c120c5b390ad289ebb4302f8d7303d3",
    "99b35ff452f61ab29212c06d4fd6b3e11803c3eca1d8cee941b1c92f0e61f104",
    "1597395c42561f90e00c9ddf446baccc387e79dfd771e9dc47170b728e65c735",
)


def aspect_envelope():
    """Select v2 through the complete production envelope producer."""
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
        support_validation_revision="host-support-receipts-v2",
        evaluation_response_contract=batched.EVALUATION_RESPONSE_CONTRACT_V2,
    )


@pytest.fixture
def aspect_payload(permissions):
    """Retain citations while giving equal-source claims different aspect sets."""
    value = copy.deepcopy(permissions)
    value["writer_envelope"] = aspect_envelope()
    value["assertions"][1]["required_support_aspects"] = ["predicate"]
    value["schema"] = batched.assessment_schema(value)
    return value


def validator_for(payload, alias_wire=False):
    """Bind the actual native schema in the chosen final candidate namespace."""
    aliases = batched.batch_support_aliases(payload) if alias_wire else None
    native = batched.bind_assessment_schema(
        unbound_native(payload, aliases), payload, passage_aliases=aliases
    )
    schema = native["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema), native, aliases


def decision_branch(native, claim_id):
    """Find a claim's schema branch by its declared claim-id permission."""
    schema = native["json_schema"]["schema"]
    branches = [
        schema["$defs"][ref["$ref"].removeprefix("#/$defs/")]
        for ref in schema["properties"]["decisions"]["items"]["anyOf"]
    ]
    return next(b for b in branches if claim_id in b["properties"]["claim_id"]["enum"])


def test_v1_actual_callable_sources_and_default_identity_remain_exact():
    hashes = tuple(
        hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        for function in batched._RESPONSE_CONTRACT_V1_SOURCES
    )
    assert hashes == LEGACY_SOURCE_HASHES
    assert batched.response_contract_identity() == V1_IDENTITY
    assert (
        batched.response_contract_identity(
            contract=batched.EVALUATION_RESPONSE_CONTRACT
        )
        == V1_IDENTITY
    )
    assert (
        batched.response_contract_identity(
            contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
        )
        != V1_IDENTITY
    )


@pytest.mark.parametrize(
    "source_name", ["_bind_assessment_schema_v2", "_share_assessment_properties_v2"]
)
def test_v2_identity_binds_actual_new_binder_source(monkeypatch, source_name):
    original = inspect.getsource
    target = getattr(batched, source_name)
    before = batched.response_contract_identity(
        contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
    )

    def changed_source(function):
        source = original(function)
        if function is target:
            return source + "\n# independent source mutation probe\n"
        return source

    monkeypatch.setattr(batched.inspect, "getsource", changed_source)
    assert batched.response_contract_identity() == V1_IDENTITY
    assert (
        batched.response_contract_identity(
            contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
        )
        != before
    )


@pytest.mark.parametrize("alias_wire", [False, True])
@pytest.mark.parametrize("contract", [None, batched.EVALUATION_RESPONSE_CONTRACT])
def test_none_and_v1_dispatch_keep_the_original_native_result(
    permissions, contract, alias_wire
):
    payload = copy.deepcopy(permissions)
    payload["writer_envelope"] = response_envelope(selected=contract is not None)
    aliases = batched.batch_support_aliases(payload) if alias_wire else None
    native = unbound_native(payload, aliases)
    original = batched._BIND_ASSESSMENT_SCHEMA_V1(
        native, payload, passage_aliases=aliases
    )
    dispatched = batched.bind_assessment_schema(
        native, payload, passage_aliases=aliases
    )
    assert dispatched == original
    if contract is None:
        assert dispatched is native
        assert not any(
            k.startswith("evaluation_response_") for k in payload["writer_envelope"]
        )


@pytest.mark.parametrize("alias_wire", [False, True])
def test_same_sources_different_required_sets_cannot_share_permissions(
    aspect_payload, alias_wire
):
    before = copy.deepcopy(aspect_payload)
    validator, native, aliases = validator_for(aspect_payload, alias_wire)
    reply = complete_reply(aspect_payload, native, aliases)
    validator.validate(reply)
    first = decision_branch(native, "claim_0")
    second = decision_branch(native, "claim_1")
    assert first is not second
    assert first["properties"]["support_basis"] == second["properties"]["support_basis"]
    assert first["properties"]["aspect_checks"]["items"]["properties"]["aspect"][
        "enum"
    ] == ["predicate", "conditions"]
    assert second["properties"]["aspect_checks"]["items"]["properties"]["aspect"][
        "enum"
    ] == ["predicate"]
    reply["decisions"][1]["aspect_checks"].append(
        copy.deepcopy(reply["decisions"][0]["aspect_checks"][1])
    )
    assert list(validator.iter_errors(reply))
    assert aspect_payload == before


def test_equal_aspect_sets_share_only_schema_and_preserve_separate_decisions(
    aspect_payload,
):
    payload = copy.deepcopy(aspect_payload)
    payload["assertions"][1]["required_support_aspects"] = ["conditions", "predicate"]
    validator, native, aliases = validator_for(payload)
    first = decision_branch(native, "claim_0")
    assert first is decision_branch(native, "claim_1")
    reply = complete_reply(payload, native, aliases)
    validator.validate(reply)
    assert [row["claim_id"] for row in reply["decisions"]] == [
        claim["claim_id"] for claim in payload["assertions"]
    ]


@pytest.mark.parametrize("alias_wire", [False, True])
@pytest.mark.parametrize("location", ["support_basis", "aspect_checks"])
def test_v2_still_rejects_foreign_claim_support(aspect_payload, alias_wire, location):
    validator, native, aliases = validator_for(aspect_payload, alias_wire)
    reply = complete_reply(aspect_payload, native, aliases)
    validator.validate(reply)
    foreign = reply["decisions"][2]["support_basis"][0]["support_id"]
    if location == "support_basis":
        reply["decisions"][0][location] = [{"support_id": foreign}]
    else:
        reply["decisions"][0][location][0]["support_ids"] = [foreign]
    assert list(validator.iter_errors(reply))


@pytest.mark.parametrize("alias_wire", [False, True])
@pytest.mark.parametrize("status", ["unsupported", "unverifiable"])
def test_negative_verdicts_and_uncited_science_remain_expressible(
    aspect_payload, alias_wire, status
):
    validator, native, aliases = validator_for(aspect_payload, alias_wire)
    reply = complete_reply(aspect_payload, native, aliases)
    for index in (0, 3):
        row = reply["decisions"][index]
        row.update(
            status=status,
            reason_code=(
                "insufficient_support" if status == "unsupported" else "unverifiable"
            ),
        )
        for check in row["aspect_checks"]:
            check.update(supported=False, support_ids=[])
    validator.validate(reply)
    assert reply["decisions"][3]["support_basis"] == []
    assert [a["aspect"] for a in reply["decisions"][3]["aspect_checks"]] == [
        "predicate",
        "conditions",
    ]
    reply["decisions"][3]["aspect_checks"][0]["support_ids"] = [
        reply["decisions"][0]["support_basis"][0]["support_id"]
    ]
    assert list(validator.iter_errors(reply))


def test_empty_required_set_allows_only_empty_array_without_empty_enum(aspect_payload):
    validator, native, aliases = validator_for(aspect_payload)
    reply = complete_reply(aspect_payload, native, aliases)
    validator.validate(reply)
    assert reply["decisions"][4]["aspect_checks"] == []
    branch = decision_branch(native, "claim_4")
    assert branch["properties"]["aspect_checks"]["maxItems"] == 0

    def walk(node):
        if isinstance(node, dict):
            if "enum" in node:
                assert node["enum"]
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(native)
    reply["decisions"][4]["aspect_checks"] = [
        {
            "aspect": "predicate",
            "supported": False,
            "support_ids": [],
            "explanation": "No science.",
        }
    ]
    assert list(validator.iter_errors(reply))


@pytest.mark.parametrize(
    "required", [None, "predicate", ["unknown"], ["predicate", "predicate"], [True]]
)
def test_invalid_host_required_aspects_fail_before_native_binding(
    aspect_payload, required
):
    payload = copy.deepcopy(aspect_payload)
    payload["assertions"][0]["required_support_aspects"] = required
    with pytest.raises(ValueError, match="evaluation_response_claim_aspects_invalid"):
        batched.bind_assessment_schema(unbound_native(payload), payload)


@pytest.mark.parametrize(
    "fault", ["duplicate_claim", "unknown_alias", "swapped_alias", "duplicate_alias"]
)
def test_v2_preserves_claim_identity_and_alias_bijection_guards(aspect_payload, fault):
    payload = copy.deepcopy(aspect_payload)
    native = unbound_native(payload)
    aliases = batched.batch_support_aliases(payload)
    if fault == "duplicate_claim":
        payload["assertions"][1]["claim_id"] = payload["assertions"][0]["claim_id"]
    elif fault == "unknown_alias":
        aliases["p404"] = "unknown-passage"
    elif fault == "swapped_alias":
        keys = list(aliases)[:2]
        aliases[keys[0]], aliases[keys[1]] = aliases[keys[1]], aliases[keys[0]]
    else:
        aliases["p404"] = next(iter(aliases.values()))
    with pytest.raises(
        ValueError,
        match="evaluation_response_(claim_permissions|alias_identity)_invalid",
    ):
        batched.bind_assessment_schema(native, payload, passage_aliases=aliases)


@pytest.mark.parametrize(
    "fault",
    ["unknown", "identity", "missing_identity", "missing_marker", "parent", "round"],
)
def test_selector_rejects_forged_or_incomplete_new_pair(fault):
    envelope = aspect_envelope()
    if fault == "unknown":
        envelope["evaluation_response_contract"] = "host-bound-evaluation-v404"
    elif fault == "identity":
        envelope["evaluation_response_identity"] = V1_IDENTITY
    elif fault == "missing_identity":
        envelope.pop("evaluation_response_identity")
    elif fault == "missing_marker":
        envelope.pop("evaluation_response_contract")
    elif fault == "parent":
        envelope["support_validation_revision"] = None
    else:
        envelope.pop("classification_review_contract")
    with pytest.raises(ValueError, match="evaluation_response_contract_invalid"):
        batched.selected_response_contract(envelope)


# These exact claim texts and aspect-name projections come from the immutable 054
# responses. A controlled exact-text passage below isolates the contract defect;
# it is not a replacement source or a scientific reassessment of those responses.
LIVE054_ASPECT_CASES = (
    (
        "few-shot-claim_169",
        "性能解读的主要权衡是：更强的关系学习、子空间建模、多层级融合或不确定性估计可能改善特定瓶颈，但也带来更复杂的结构或优化。",
        ["predicate", "conditions"],
        ["predicate", "conditions", "causal_relation"],
        False,
    ),
    (
        "nerf-claim_32",
        "沿光线的体绘制函数天然可微，因此渲染像素颜色对 MLP 输出的颜色、密度以及网络参数都可通过该复合过程传递梯度。",
        ["predicate", "conditions", "causal_relation"],
        ["predicate", "conditions"],
        True,
    ),
)


@pytest.mark.parametrize(
    "case_id,text,required,actual,native_valid", LIVE054_ASPECT_CASES
)
def test_054_extra_and_missing_aspect_projections_do_not_become_valid_science(
    aspect_payload, case_id, text, required, actual, native_valid
):
    payload = copy.deepcopy(aspect_payload)
    payload["assertions"] = [payload["assertions"][0]]
    payload["assertions"][0].update(claim_text=text, required_support_aspects=required)
    payload["schema"] = batched.assessment_schema(payload)
    before = copy.deepcopy((text, required, actual))
    assert (
        scoped.required_support_aspects(
            text,
            applicability=True,
            support_validation_revision="host-support-receipts-v2",
        )
        == required
    )
    validator, native, aliases = validator_for(payload)
    reply = complete_reply(payload, native, aliases)
    prototype = reply["decisions"][0]["aspect_checks"][0]
    reply["decisions"][0]["aspect_checks"] = [
        {**prototype, "aspect": name} for name in actual
    ]
    assert validator.is_valid(reply) is native_valid, case_id
    legacy = copy.deepcopy(payload)
    legacy["writer_envelope"] = response_envelope()
    legacy_native = batched.bind_assessment_schema(unbound_native(legacy), legacy)
    Draft202012Validator(legacy_native["json_schema"]["schema"]).validate(reply)

    claim = {"claim_text": text, "evidence_ids": ["owned"]}
    ledger = {"owned": {"text": text, "supporting_text": text}}
    decision, passages = selected(
        claim, ledger, {"owned": "e1"}, source="original_support"
    )
    prototype = decision.aspect_checks[0].model_dump()
    decision = decision.model_copy(
        update={
            "aspect_checks": [
                scoped.AspectCheck(**{**prototype, "aspect": name}) for name in actual
            ]
        }
    )
    result = scoped.validate_support_basis(
        claim,
        decision,
        ledger,
        {"owned": "e1"},
        passages=passages,
        original_only=True,
        applicability=True,
        support_validation_revision="host-support-receipts-v2",
    )
    assert result["issue"] == "support_aspect_coverage_invalid"
    assert result["status"] == "unverified"
    assert (text, required, actual) == before


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "foreign_source", "wrong_fact"]
)
def test_native_valid_shape_cannot_override_existing_host_guards(aspect_payload, fault):
    text = "Model-A improves efficiency via distillation."
    payload = copy.deepcopy(aspect_payload)
    payload["assertions"] = [payload["assertions"][0]]
    payload["assertions"][0].update(
        claim_text=text, required_support_aspects=scoped.required_support_aspects(text)
    )
    payload["schema"] = batched.assessment_schema(payload)
    validator, native, native_aliases = validator_for(payload)
    reply = complete_reply(payload, native, native_aliases)
    if fault == "missing":
        reply["decisions"][0]["aspect_checks"].pop()
    elif fault == "duplicate":
        reply["decisions"][0]["aspect_checks"].append(
            copy.deepcopy(reply["decisions"][0]["aspect_checks"][0])
        )
    validator.validate(reply)
    source = text if fault != "wrong_fact" else "Model-A has quadratic attention cost."
    claim = {"claim_text": text, "evidence_ids": ["owned"]}
    ledger = {"owned": {"text": source, "supporting_text": source}}
    aliases = {"owned": "e1"}
    decision, passages = selected(claim, ledger, aliases)
    if fault == "missing":
        decision.aspect_checks.pop()
    elif fault == "duplicate":
        decision.aspect_checks.append(decision.aspect_checks[0].model_copy())
    elif fault == "foreign_source":
        claim["evidence_ids"] = ["different-owner"]
    result = scoped.validate_support_basis(
        claim, decision, ledger, aliases, passages=passages
    )
    assert result["status"] == "unverified"
    assert (
        result["issue"]
        == {
            "missing": "support_aspect_coverage_invalid",
            "duplicate": "support_aspect_coverage_invalid",
            "foreign_source": "support_basis_ownership_invalid",
            "wrong_fact": "support_relation_unverified",
        }[fault]
    )


def test_host_supported_control_remains_supported_without_relaxing_checks():
    text = "Model-A improves efficiency via distillation."
    claim = {"claim_text": text, "evidence_ids": ["owned"]}
    ledger = {"owned": {"text": text, "supporting_text": text}}
    aliases = {"owned": "e1"}
    decision, passages = selected(claim, ledger, aliases)
    result = scoped.validate_support_basis(
        claim, decision, ledger, aliases, passages=passages
    )
    assert result["status"] == "passed"
    assert result["spans"][0]["quote"] == text


def sharing_input(payload, count=5):
    """Use actual stage nodes to isolate the private representation transform."""
    value = unbound_native(payload)
    schema = value["json_schema"]["schema"]
    definitions = schema["$defs"]
    template = definitions["PassageDecision"]
    names = []
    for index in range(count):
        name = f"ClaimAspectPermissionGroup{index + 1}"
        branch = copy.deepcopy(template)
        branch["properties"]["claim_id"]["enum"] = [f"claim_{index}"]
        branch["properties"]["aspect_checks"]["items"] = copy.deepcopy(
            definitions["AspectCheck"]
        )
        definitions[name] = branch
        names.append(name)
    schema["properties"]["decisions"]["items"] = {
        "anyOf": [{"$ref": f"#/$defs/{name}"} for name in names]
    }
    batched._prune_response_definitions(schema)
    return value, names


def expanded_node(schema, node):
    """Resolve with the existing schema consumer, including annotation handling."""
    return inline_local_schema({"$defs": schema["$defs"], **node})


@pytest.mark.parametrize("alias_wire", [False, True])
def test_actual_v2_candidates_share_only_unchanged_public_nodes(
    aspect_payload, alias_wire
):
    aliases = batched.batch_support_aliases(aspect_payload) if alias_wire else None
    raw = unbound_native(aspect_payload, aliases)
    raw_before = copy.deepcopy(raw)
    bound = batched.bind_assessment_schema(raw, aspect_payload, passage_aliases=aliases)
    schema = bound["json_schema"]["schema"]
    pools = {
        name: node for name, node in schema["$defs"].items() if name.startswith("S")
    }
    assert pools
    original = raw["json_schema"]["schema"]["$defs"]
    for claim in aspect_payload["assertions"]:
        branch = decision_branch(bound, claim["claim_id"])
        assert branch["required"] == original["PassageDecision"]["required"]
        assert branch["additionalProperties"] is False
        for field, node in original["PassageDecision"]["properties"].items():
            if field not in {"claim_id", "support_basis", "aspect_checks"}:
                assert expanded_node(schema, branch["properties"][field]) == node
        checks = branch["properties"]["aspect_checks"]["items"]
        assert checks["required"] == original["AspectCheck"]["required"]
        assert checks["additionalProperties"] is False
        for field in ("supported", "explanation"):
            assert (
                expanded_node(schema, checks["properties"][field])
                == original["AspectCheck"]["properties"][field]
            )
        assert "$ref" not in branch["properties"]["claim_id"]
        assert "$ref" not in branch["properties"]["support_basis"]
        assert "$ref" not in checks["properties"]["aspect"]
        assert "$ref" not in checks["properties"]["support_ids"]
    reply = complete_reply(aspect_payload, bound, aliases)
    Draft202012Validator(schema).validate(reply)
    assert raw == raw_before


def test_sharing_expands_exactly_and_preserves_positive_negative_and_invalid_shapes(
    aspect_payload,
):
    native, names = sharing_input(aspect_payload)
    before = copy.deepcopy(native)
    shared = batched._share_assessment_properties_v2(native, names)
    old, new = [value["json_schema"]["schema"] for value in (native, shared)]
    assert inline_local_schema(old) == inline_local_schema(new)
    for name, node in old["$defs"].items():
        assert expanded_node(old, node) == expanded_node(new, new["$defs"][name])
    assert len(json.dumps(shared, ensure_ascii=False).encode()) < len(
        json.dumps(native, ensure_ascii=False).encode()
    )
    validators = [Draft202012Validator(schema) for schema in (old, new)]
    reply = complete_reply(aspect_payload, native)
    controls = [(reply, True)]
    negative = copy.deepcopy(reply)
    negative["decisions"][0].update(
        status="unsupported", reason_code="insufficient_support", source_faithful=False
    )
    for check in negative["decisions"][0]["aspect_checks"]:
        check.update(supported=False, support_ids=[])
    controls.append((negative, True))
    for field, value in (
        ("atomic", "true"),
        ("status", "invented"),
        ("explanation", ""),
    ):
        invalid = copy.deepcopy(reply)
        invalid["decisions"][0][field] = value
        controls.append((invalid, False))
    missing = copy.deepcopy(reply)
    missing["decisions"][0].pop("source_faithful")
    controls.append((missing, False))
    extra = copy.deepcopy(reply)
    extra["decisions"][0]["new_field"] = True
    controls.append((extra, False))
    for value, expected in controls:
        assert [validator.is_valid(value) for validator in validators] == [
            expected,
            expected,
        ]
    assert native == before
    assert batched._share_assessment_properties_v2(shared, names) == shared


def test_sharing_avoids_existing_definition_names(aspect_payload):
    native, names = sharing_input(aspect_payload)
    retained = {"type": "string", "enum": ["original-S0"]}
    native["json_schema"]["schema"]["$defs"]["S0"] = retained
    shared = batched._share_assessment_properties_v2(native, names)
    definitions = shared["json_schema"]["schema"]["$defs"]
    assert definitions["S0"] == retained
    assert any(name.startswith("S") and name != "S0" for name in definitions)
    assert inline_local_schema(native["json_schema"]["schema"]) == inline_local_schema(
        shared["json_schema"]["schema"]
    )


@pytest.mark.parametrize(
    "extension",
    [
        {"$id": "https://example.invalid/scoped"},
        {"$anchor": "scoped"},
        {"$dynamicAnchor": "scoped"},
        {"$dynamicRef": "#scoped"},
        {"$recursiveRef": "#"},
        {"$ref": "#/$defs/UnitDecision"},
        {"anyOf": [{"type": "string"}]},
        {"allOf": [{"type": "string"}]},
        {"oneOf": [{"type": "string"}]},
        {"not": {"const": "excluded"}},
    ],
)
def test_scoped_referenced_or_applicator_nodes_are_never_hoisted(
    aspect_payload, extension
):
    native, names = sharing_input(aspect_payload)
    for name in names:
        native["json_schema"]["schema"]["$defs"][name]["properties"][
            "explanation"
        ].update(extension)
    before = copy.deepcopy(native)
    shared = batched._share_assessment_properties_v2(native, names)
    for name in names:
        assert (
            shared["json_schema"]["schema"]["$defs"][name]["properties"]["explanation"]
            == before["json_schema"]["schema"]["$defs"][name]["properties"][
                "explanation"
            ]
        )
    assert native == before


def test_literal_annotation_objects_are_not_walked_as_schema(aspect_payload):
    native, names = sharing_input(aspect_payload)
    literal = {
        "properties": {
            "explanation": {"type": "string", "description": "literal" * 50}
        },
        "$ref": "this-is-data-not-a-schema-reference",
    }
    for name in names:
        native["json_schema"]["schema"]["$defs"][name]["default"] = copy.deepcopy(
            literal
        )
    shared = batched._share_assessment_properties_v2(native, names)
    for name in names:
        assert shared["json_schema"]["schema"]["$defs"][name]["default"] == literal
    assert inline_local_schema(native["json_schema"]["schema"]) == inline_local_schema(
        shared["json_schema"]["schema"]
    )


def test_no_repeated_or_unprofitable_nodes_do_not_grow_native(aspect_payload):
    native, names = sharing_input(aspect_payload, count=2)
    for index, name in enumerate(names):
        properties = native["json_schema"]["schema"]["$defs"][name]["properties"]
        for field, node in properties.items():
            if field not in {"claim_id", "support_basis", "aspect_checks"}:
                node["description"] = f"unique-{index}-{field}"
        properties["atomic"] = {"type": "boolean"}
        for field, node in properties["aspect_checks"]["items"]["properties"].items():
            node["description"] = f"unique-aspect-{index}-{field}"
    before = copy.deepcopy(native)
    assert batched._share_assessment_properties_v2(native, names) == before
    assert batched._share_assessment_properties_v2(native, []) is native
    assert native == before
