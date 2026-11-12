"""Keep explicit support receipts narrow, reversible and owned by real citations."""

import copy
import json

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_support_passages import selected

from litagent.capacity import CapacityConfig
from litagent.eval.batched import (
    assessment_schema,
    batch_alias_input,
    batch_prompt,
    batch_support_aliases,
    restore_batch_support_ids,
)
from litagent.eval.scoped import (
    SUPPORT_VALIDATION_INSTRUCTION,
    SUPPORT_VALIDATION_REVISION,
    WRITER_SUPPORT_RECEIPT_INSTRUCTION,
    Assessment,
    _protocol_numeric_facts,
    _support_revision,
    build_evaluation_request,
    partial_assessment,
    required_support_aspects,
    support_receipt_payload,
    support_validation_identity,
    validate_support_basis,
)
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import make_writer_envelope

REVISION = SUPPORT_VALIDATION_REVISION


def envelope(*, active=True):
    """Use the complete host-selected parent policy rather than a marker mock."""
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
        **({"support_validation_revision": REVISION} if active else {}),
    )


@pytest.fixture
def receipts(canonical):
    """Keep same-paper evidence records distinct under one report fragment."""
    value = copy.deepcopy(canonical)
    value["writer_envelope"] = envelope()
    value["evaluation_input_contract"] = "flat-evaluation-input-v1"
    value["cited_materials"]["e2"] = {"original_support_ref": "s2"}
    value["original_supports"]["s2"] = copy.deepcopy(value["original_supports"]["s1"])
    unit = value["source_units"][0]
    unit["citation_aliases"] = ["e1", "e2"]
    unit["source_fragments"][0].update(
        citation_aliases=["e1", "e2"],
        text="A method has a source-bound limitation. [E:e1] [E:e2]",
    )
    value["support_passages"] = {
        "passage:" + str(i).zfill(64): [alias, ref, 0, 29]
        for i, (alias, ref) in enumerate([("e1", "s1")] * 10 + [("e2", "s2")] * 10, 1)
    }
    value["assertions"][1]["evidence"] = [
        {"evidence_alias": "e2", "material_ref": "e2"}
    ]
    value["assertions"][2].update(
        kind="report_structure", evidence=[], required_support_aspects=[]
    )
    value["assertions"][3]["evidence"] = []
    value["schema"] = assessment_schema(value)
    return value


def checked(text, source, *, revision=REVISION):
    """Exercise the full host support validator, including its earlier guards."""
    ledger = {"a": {"text": source, "supporting_text": source}}
    claim = {"claim_text": text, "evidence_ids": ["a"]}
    decision, passages = selected(claim, ledger, {"a": "e1"})
    # Use the same public aspect producer as the real assessment payload.
    aspects = required_support_aspects(text, support_validation_revision=revision)
    decision.aspect_checks = [
        decision.aspect_checks[0].model_copy(update={"aspect": aspect})
        for aspect in aspects
    ]
    return validate_support_basis(
        claim,
        decision,
        ledger,
        {"a": "e1"},
        passages=passages,
        original_only=True,
        local_numeric_bindings=True,
        chinese_comparatives=True,
        protocol_numeric_scope=True,
        support_validation_revision=revision,
    )


def test_revision_identity_is_stable_and_binds_both_consumer_instructions():
    assert REVISION == "host-support-receipts-v1"
    assert len(support_validation_identity()) == 64
    assert support_validation_identity() == support_validation_identity()
    assert "allowed_support_ids" in SUPPORT_VALIDATION_INSTRUCTION
    assert "allowed_support_ids" not in WRITER_SUPPORT_RECEIPT_INSTRUCTION
    assert "percentage points" in WRITER_SUPPORT_RECEIPT_INSTRUCTION


@pytest.mark.parametrize("invalid", [False, True, 0, 1, "", [], ["revision"]])
def test_non_mapping_envelope_is_a_typed_host_failure(invalid):
    with pytest.raises(ValueError, match="writer_envelope_invalid"):
        _support_revision(invalid)
    with pytest.raises(ValueError, match="writer_envelope_invalid"):
        build_evaluation_request("Keep the input.", {"writer_envelope": invalid})


@pytest.mark.parametrize(
    "invalid",
    [
        {"support_validation_revision": None},
        {"support_validation_revision": None, "support_validation_identity": None},
        {"support_validation_revision": REVISION},
        {"support_validation_identity": support_validation_identity()},
        {"support_validation_revision": "unknown", "support_validation_identity": "x"},
        {"support_validation_revision": REVISION, "support_validation_identity": "x"},
    ],
)
def test_explicit_marker_cannot_be_dropped_or_replaced_with_none(invalid):
    with pytest.raises(ValueError, match="support_validation_"):
        _support_revision(invalid)
    assert _support_revision(None) is None
    assert _support_revision({}) is None


@pytest.mark.parametrize(
    "text",
    ["材料仅给出部分结果，故不能认定条件相同。", "故只能作有条件的比较。"],
)
def test_standalone_causal_connector_uses_one_new_aspect_plan(text):
    old = required_support_aspects(text)
    new = required_support_aspects(text, support_validation_revision=REVISION)
    assert "causal_relation" not in old
    assert new == [*old, "causal_relation"]


@pytest.mark.parametrize(
    "text", ["故障率为2%。", "事故分析属于当前范围。", "这是一项故有方法。"]
)
def test_words_containing_gu_are_not_causal_connectors(text):
    assert required_support_aspects(text, support_validation_revision=REVISION) == (
        required_support_aspects(text)
    )


@pytest.mark.parametrize(
    "text",
    [
        "摘要未给出1-shot/5-shot数值。",
        "当前来源没有报告1-shot和5-shot准确率。",
        "The supplied abstract does not report 1-shot or 5-shot accuracy.",
    ],
)
def test_source_bounded_absent_protocols_are_not_positive_results(text):
    assert _protocol_numeric_facts(text)
    assert _protocol_numeric_facts(text, support_validation_revision=REVISION) == []
    assert checked(text, "The supplied abstract describes the method.")["status"] == (
        "passed"
    )


@pytest.mark.parametrize("separator", ["，", ", ", "；", "; ", ". ", "但", " but "])
def test_a_later_positive_protocol_cannot_inherit_absence(separator):
    text = "摘要未给出1-shot准确率" + separator + "5-shot准确率为80%。"
    facts = _protocol_numeric_facts(text, support_validation_revision=REVISION)
    assert ("5", False, "protocol_shot", None) in facts
    assert any(fact[:2] == ("8E+1", True) or fact[:2] == ("80", True) for fact in facts)
    # A matching result at a different shot condition must not license this claim.
    assert checked(text, "1-shot accuracy is 80%.")["status"] == "unverified"
    assert checked(text, "5-shot accuracy is 80%.")["status"] == "passed"


@pytest.mark.parametrize(
    "text",
    [
        "摘要未给出1-shot准确率并报告5-shot准确率为80%。",
        "摘要未给出1-shot准确率并达到5-shot准确率80%。",
        "The supplied abstract does not report 1-shot accuracy and reports 5-shot accuracy of 80%.",
    ],
)
def test_independent_positive_predicate_breaks_negation_without_a_comma(text):
    facts = _protocol_numeric_facts(text, support_validation_revision=REVISION)
    assert ("5", False, "protocol_shot", None) in facts
    assert checked(text, "1-shot accuracy is 80%.")["status"] == "unverified"
    assert checked(text, "5-shot accuracy is 80%.")["status"] == "passed"


@pytest.mark.parametrize("conjunction", ["and", "or"])
@pytest.mark.parametrize("copula", ["is", "was", "are", "were"])
@pytest.mark.parametrize("first_noun", [" accuracy", ""])
def test_english_unpunctuated_copula_cannot_inherit_an_earlier_absence(
    conjunction, copula, first_noun
):
    text = (
        "Supplied text does not report 1-shot"
        + first_noun
        + f" {conjunction} 5-shot accuracy {copula} 90%."
    )
    facts = _protocol_numeric_facts(text, support_validation_revision=REVISION)
    assert ("5", False, "protocol_shot", None) in facts
    assert checked(text, "The 1-shot accuracy is 90%.")["status"] == "unverified"
    if first_noun:
        assert ("1", False, "protocol_shot", None) not in facts
        assert checked(text, "The 5-shot accuracy is 90%.")["status"] == "passed"


@pytest.mark.parametrize("conjunction", ["和", "及", "与"])
@pytest.mark.parametrize("copula", ["为", "是"])
@pytest.mark.parametrize("first_noun", ["准确率", ""])
def test_chinese_unpunctuated_copula_cannot_inherit_an_earlier_absence(
    conjunction, copula, first_noun
):
    text = f"来源未给出1-shot{first_noun}{conjunction}5-shot准确率{copula}90%。"
    facts = _protocol_numeric_facts(text, support_validation_revision=REVISION)
    assert ("5", False, "protocol_shot", None) in facts
    assert checked(text, "1-shot准确率是90%。")["status"] == "unverified"
    if first_noun:
        assert ("1", False, "protocol_shot", None) not in facts
        assert checked(text, "5-shot准确率是90%。")["status"] == "passed"


@pytest.mark.parametrize("value", ["70", "80", "90"])
@pytest.mark.parametrize(
    "template",
    [
        "Supplied text does not report 1-shot accuracy and 5-shot accuracy is {value}%.",
        "Supplied text does not report 1-shot accuracy and reports 5-shot accuracy of {value}%.",
        "Supplied text does not report 1-shot accuracy reports 5-shot accuracy is {value}%.",
        "Supplied text does not report 1-shot accuracy; 5-shot accuracy was {value}%.",
        "来源未给出1-shot准确率和5-shot准确率是{value}%。",
        "来源未给出1-shot准确率并报告5-shot准确率为{value}%。",
        "来源未给出1-shot准确率报告5-shot准确率是{value}%。",
        "来源未给出1-shot准确率；5-shot准确率为{value}%。",
    ],
)
def test_same_mixed_claim_pairs_correct_and_wrong_protocol_sources(template, value):
    text = template.format(value=value)
    wrong = checked(text, f"The 1-shot accuracy is {value}%.")
    correct = checked(text, f"The 5-shot accuracy is {value}%.")
    assert wrong["status"] == "unverified"
    assert correct["status"] == "passed"
    different = "80" if value != "80" else "90"
    assert (
        checked(text, f"The 5-shot accuracy is {different}%.")["status"] == "unverified"
    )


@pytest.mark.parametrize("positive", ["1", "5"])
@pytest.mark.parametrize("value", ["80", "90"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("order", ["absence_first", "positive_first"])
def test_only_the_actual_positive_protocol_and_value_may_support_a_mixed_claim(
    positive, value, language, order
):
    missing = "5" if positive == "1" else "1"
    if language == "en":
        absence = f"supplied text does not report {missing}-shot accuracy"
        result = f"{positive}-shot accuracy is {value}%"
        text = (
            f"{absence} and {result}."
            if order == "absence_first"
            else f"{result} and {absence}."
        )
    else:
        absence = f"来源未给出{missing}-shot准确率"
        result = f"{positive}-shot准确率为{value}%"
        text = (
            f"{absence}和{result}。"
            if order == "absence_first"
            else f"{result}和{absence}。"
        )
    assert checked(text, f"{positive}-shot accuracy is {value}%.")["status"] == "passed"
    assert (
        checked(text, f"{missing}-shot accuracy is {value}%.")["status"] == "unverified"
    )
    other_value = "80" if value == "90" else "90"
    assert (
        checked(text, f"{positive}-shot accuracy is {other_value}%.")["status"]
        == "unverified"
    )


@pytest.mark.parametrize(
    "text",
    [
        "未给出1-shot/5-shot准确率。",
        "We do not report 1-shot/5-shot accuracy.",
        "来源给出1-shot和5-shot准确率。",
        "Supplied text reports 1-shot and 5-shot accuracy.",
        "来源1-shot和5-shot准确率。",
        "来源1-shot和5-shot准确率未给出。",
    ],
)
def test_a_missing_source_bound_or_preceding_absence_predicate_grants_no_exemption(
    text,
):
    assert _protocol_numeric_facts(text, support_validation_revision=REVISION) == (
        _protocol_numeric_facts(text)
    )
    assert (
        checked(text, "The supplied text describes a method.")["status"] == "unverified"
    )


@pytest.mark.parametrize(
    "text,source",
    [
        (
            "EViT average speed is 43 FPS and peak speed is 50 FPS.",
            "EViT average speed is 50 FPS and peak speed is 43 FPS.",
        ),
        ("Latency decreases by 20%.", "Throughput increases by 20%."),
        ("Evaluation uses 5-way 1-shot tasks.", "Evaluation uses 5-way 5-shot tasks."),
    ],
)
def test_new_noun_phrase_scope_preserves_quantity_roles_metrics_and_conditions(
    text, source
):
    assert checked(text, source)["status"] == "unverified"


@pytest.mark.parametrize(
    "text",
    [
        "Supplied text does not report 1-shot and 5-shot is 90% in accuracy.",
        "来源未给出1-shot及5-shot为90%的准确率。",
        "来源未给出1-shot和5-shot=90%的准确率。",
    ],
)
def test_copula_before_the_first_result_noun_preserves_positive_conditions(text):
    facts = _protocol_numeric_facts(text, support_validation_revision=REVISION)
    assert ("5", False, "protocol_shot", None) in facts
    assert checked(text, "1-shot accuracy is 90%.")["status"] == "unverified"


@pytest.mark.parametrize(
    "text",
    [
        "Supplied text does not report 1-shot accuracy and 5-shot accuracy.",
        "来源未给出1-shot准确率和5-shot数值。",
    ],
)
def test_absence_scope_stops_at_its_first_result_noun_without_a_copula(text):
    facts = _protocol_numeric_facts(text, support_validation_revision=REVISION)
    assert ("5", False, "protocol_shot", None) in facts


def test_first_missing_noun_keeps_its_enumerated_protocols_without_a_result_value():
    for conjunction in ("/", "和", "及", "与"):
        text = f"来源未给出1-shot{conjunction}5-shot的准确率数值。"
        assert _protocol_numeric_facts(text, support_validation_revision=REVISION) == []


def test_a_supported_positive_condition_after_absence_is_still_checked_normally():
    text = "摘要未给出1-shot准确率，5-shot准确率为80%。"
    assert checked(text, "5-shot accuracy is 80%.")["status"] == "passed"
    assert checked(text, "1-shot accuracy is 80%.")["status"] == "unverified"


@pytest.mark.parametrize(
    "claim,source",
    [
        ("没有达到80%准确率。", "Accuracy is 70%."),
        (
            "来源未给出1-shot/5-shot数值；5-shot准确率提升2%。",
            "1-shot accuracy improves by 2%.",
        ),
        ("Accuracy improves by 2 percentage points.", "Accuracy improves by 2%."),
        ("Evaluation uses 5-shot tasks.", "Evaluation uses 1-shot tasks."),
        ("EViT reduces FLOPs by 30%.", "Evo-ViT reduces FLOPs by 30%."),
        (
            "miniImageNet accuracy improves by 2%.",
            "miniImagenNet accuracy improves by 2%.",
        ),
    ],
)
def test_negation_never_exempts_result_values_units_or_source_entities(claim, source):
    assert checked(claim, source)["status"] == "unverified"


def test_same_paper_records_and_uncited_science_have_exact_permissions(receipts):
    before = copy.deepcopy(receipts)
    projected = support_receipt_payload(receipts)
    expected = [
        list(receipts["support_passages"])[:10],
        list(receipts["support_passages"])[10:],
    ]
    assert projected["assertions"][0]["allowed_support_ids"] == expected[0]
    assert projected["assertions"][1]["allowed_support_ids"] == expected[1]
    assert projected["assertions"][2]["allowed_support_ids"] == []
    assert projected["assertions"][3]["allowed_support_ids"] == []
    assert projected["support_passages"] == before["support_passages"]
    assert projected["schema"] == before["schema"]
    assert receipts == before
    assert support_receipt_payload(projected) == projected


def test_same_paper_never_grants_another_evidence_records_support():
    text = "A comparison function is learned."
    ledger = {
        eid: {"paper_id": "same-paper", "text": text, "supporting_text": text}
        for eid in ("a", "b")
    }
    claim = {"claim_text": text, "evidence_ids": ["a"]}
    aliases = {"a": "e1", "b": "e2"}
    decision, passages = selected(claim, ledger, aliases)
    other = next(
        pid
        for pid, entry in passages.items()
        if entry["evidence_id"] == "b" and entry["source"] == "original_support"
    )
    decision.support_basis = [
        decision.support_basis[0].model_copy(update={"support_id": other})
    ]
    for aspect in decision.aspect_checks:
        aspect.support_ids = [other]
    checked = validate_support_basis(
        claim,
        decision,
        ledger,
        aliases,
        passages=passages,
        original_only=True,
        support_validation_revision=REVISION,
    )
    assert checked["issue"] == "support_basis_ownership_invalid"


def test_supplied_permission_lists_are_checked_not_silently_repaired(receipts):
    invalid = support_receipt_payload(receipts)
    invalid["assertions"][0]["allowed_support_ids"].append(
        invalid["assertions"][1]["allowed_support_ids"][0]
    )
    before = copy.deepcopy(invalid)
    with pytest.raises(ValueError, match="support_permission_invalid"):
        build_evaluation_request(batch_prompt(invalid), invalid)
    assert invalid == before


def test_permission_projection_and_native_response_ids_share_one_mapping(receipts):
    projected = batch_alias_input(receipts)
    columns = projected["assertion_columns"]
    rows = [dict(zip(columns, row, strict=True)) for row in projected["assertions"]]
    aliases = batch_support_aliases(receipts)
    assert rows[0]["allowed_support_ids"] == list(aliases)[:10]
    assert rows[1]["allowed_support_ids"] == list(aliases)[10:]
    assert rows[2]["allowed_support_ids"] == rows[3]["allowed_support_ids"] == []
    assert [aliases[pid] for pid in rows[0]["allowed_support_ids"]] == (
        support_receipt_payload(receipts)["assertions"][0]["allowed_support_ids"]
    )
    messages, native = build_evaluation_request(batch_prompt(receipts), receipts)
    wire = json.loads(messages[1]["content"])
    assert "assertion_columns" in wire
    assert wire["assertion_columns"] == columns
    assert SUPPORT_VALIDATION_INSTRUCTION in messages[0]["content"]
    assert wire["writer_envelope"]["support_validation_revision"] == REVISION
    assert "support_validation_identity" not in wire["writer_envelope"]
    assert wire["support_passages"] == projected["support_passages"]
    assert wire["original_supports"] == projected["original_supports"]
    definitions = native["json_schema"]["schema"]["$defs"]
    assert definitions["SupportSelection"]["properties"]["support_id"]["enum"] == list(
        aliases
    )
    assert strict_request_reservation(messages, 16384, native) > (
        strict_request_reservation(messages, 0, native) + 16000
    )


def test_canonical_fallback_retains_full_ids_and_same_native_output_shape(receipts):
    value = copy.deepcopy(receipts)
    value.pop("assessment_input_contract")
    messages, native = build_evaluation_request(batch_prompt(value), value)
    wire = json.loads(messages[1]["content"])
    assert "assertion_columns" not in wire
    assert (
        wire["assertions"][0]["allowed_support_ids"]
        == list(value["support_passages"])[:10]
    )
    assert native["json_schema"]["schema"]["$defs"]["SupportSelection"]["properties"][
        "support_id"
    ]["enum"] == list(value["support_passages"])
    response = Assessment(units=[], conflicts=[], decisions=[])
    assert restore_batch_support_ids(response, value) is response


def test_none_keeps_old_payload_prompt_and_flat_columns(receipts):
    value = copy.deepcopy(receipts)
    value["writer_envelope"] = envelope(active=False)
    before = copy.deepcopy(value)
    assert support_receipt_payload(value) is value
    messages, _ = build_evaluation_request(batch_prompt(value), value)
    wire = json.loads(messages[1]["content"])
    assert "allowed_support_ids" not in wire.get("assertion_columns", [])
    assert SUPPORT_VALIDATION_INSTRUCTION not in messages[0]["content"]
    assert "support_validation_revision" not in wire["writer_envelope"]
    assert "support_validation_identity" not in wire["writer_envelope"]
    assert value == before


def test_non_assessment_requests_receive_no_support_permission_payload():
    payload = {"writer_envelope": envelope(), "stage": "global", "assertions": []}
    before = copy.deepcopy(payload)
    messages, _ = build_evaluation_request("Check completeness.", payload)
    wire = json.loads(messages[1]["content"])
    assert SUPPORT_VALIDATION_INSTRUCTION not in messages[0]["content"]
    assert wire["assertions"] == []
    assert "support_validation_revision" not in wire["writer_envelope"]
    assert payload == before


@pytest.mark.parametrize("tamper", ["context", "inputs", "progress", "digest"])
def test_partial_replay_cannot_drop_or_rewrite_the_selected_support_revision(tamper):
    fields = {
        "support_validation_revision": REVISION,
        "support_validation_identity": support_validation_identity(),
    }
    progress = {**fields, "batches": []}
    inputs = {
        **fields,
        "claims": [],
        "ledger": {},
        "aliases": {},
        "support_passages": {},
    }
    context = {
        "_support_validation_revision": REVISION,
        "_host_validation_inputs": inputs,
    }
    assert partial_assessment(progress, context)["processed_assertions"] == 0
    if tamper == "context":
        context.pop("_support_validation_revision")
    elif tamper == "inputs":
        inputs.pop("support_validation_revision")
    elif tamper == "progress":
        progress.pop("support_validation_revision")
    else:
        inputs["support_validation_identity"] = "0" * 64
    with pytest.raises(
        ValueError, match="(?:partial_support_identity|support_validation_)"
    ):
        partial_assessment(progress, context)
