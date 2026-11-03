"""Enforce native capacity without changing scientific text or logical JSON fields."""

import copy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_responses import envelope as response_envelope
from test_native_responses import owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.safety.planned_budget import make_writer_envelope
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    body_section_schema,
    chapter_request_input,
    compile_document,
    native_layout_guidance,
    parse_response,
    provider_response_format,
    schema_for,
    summary_request_input,
)


@pytest.fixture
def scope():
    """Declare the 16-unit whole-body capacity used by the failed live case."""
    return make_writer_envelope(
        body_units=16,
        assertions=32,
        body_bytes=16000,
        source_identity="a" * 64,
        builder_identity="b" * 64,
    )


def sentence(case, text):
    """Keep source identity independent from the truth of authored text."""
    return {"text": text, "fact_card_refs": [case.by_eid["b"]]}


def paragraphs(case, sizes, prefix):
    """Build distinct conditions so capacity checks cannot remove duplicates."""
    return [
        {
            "sentences": [
                sentence(case, f"{prefix}: condition {index}, finding {offset}.")
                for offset in range(size)
            ]
        }
        for index, size in enumerate(sizes)
    ]


@pytest.mark.parametrize("maximum", range(1, 67))
def test_native_tuple_capacity_balances_paragraphs_without_n_squared_growth(
    case, maximum
):
    schema = body_section_schema("methods", case.aliases, maximum=maximum)
    original = copy.deepcopy(schema)
    native = provider_response_format("synthesis", schema)["json_schema"]["schema"]
    chapter = native["properties"]["methods"]
    capacities = [
        len(paragraph["properties"]["sentences"]["prefixItems"])
        for paragraph in chapter["prefixItems"]
    ]
    assert sum(capacities) == maximum
    assert max(capacities) <= 3 and max(capacities) - min(capacities) <= 1
    assert chapter["items"] is False
    value = {"methods": paragraphs(case, capacities, "Different conditions")}
    before = copy.deepcopy(value)
    Draft202012Validator(native).validate(value)
    assert parse_response(json.dumps(value), schema) == value
    assert value == before and schema == original
    guidance = native_layout_guidance(schema)
    assert "bounded array" in guidance and "no additional items" in guidance
    value["methods"][0]["sentences"].append(sentence(case, "Another condition."))
    assert list(Draft202012Validator(native).iter_errors(value))
    with pytest.raises(WriterError, match="writer_native_capacity_exceeded") as exc:
        parse_response(json.dumps(value), schema)
    assert exc.value.family == "output_scope"
    assert exc.value.diagnostics[0]["actual"] == capacities[0] + 1
    assert exc.value.diagnostics[0]["maximum"] == capacities[0]


def test_020_methods_shape_cannot_consume_reserved_later_chapter_units(case, scope):
    """Replay the observed 2 + 2 + six-times-two shape, not a repaired response."""
    prior = {
        "title": "Capacity regression",
        "introduction": paragraphs(case, [2], "Introduction"),
        "taxonomy": paragraphs(case, [2], "Taxonomy"),
    }
    candidate = {"methods": paragraphs(case, [2] * 6, "Methods")}
    before = copy.deepcopy(candidate)
    _, payload, schema = chapter_request_input(
        case.packet,
        case.aliases,
        "methods",
        prior,
        prior,
        RANGE_WRITING_PLAN,
        scope,
    )
    assert payload["writing_plan_progress"]["requested_section_maximum"] == 10
    Draft202012Validator(body_section_schema("methods", case.aliases)).validate(
        candidate
    )
    native = provider_response_format("synthesis", schema)["json_schema"]["schema"]
    assert list(Draft202012Validator(native).iter_errors(candidate))
    with pytest.raises(WriterError, match="writer_native_capacity_exceeded"):
        parse_response(json.dumps(candidate), schema)
    assert candidate == before


def test_complete_16_units_keep_conditions_and_citation_bindings(case, scope):
    sizes = ([2], [2], [3, 3], [3], [3])
    body = {"title": case.body["title"]}
    for section, paragraph_sizes in zip(BODY, sizes):
        part = {section: paragraphs(case, paragraph_sizes, section)}
        if section == BODY[0]:
            part["title"] = body["title"]
        _, _, schema = chapter_request_input(
            case.packet,
            case.aliases,
            section,
            body,
            body,
            RANGE_WRITING_PLAN,
            scope,
        )
        native = provider_response_format("synthesis", schema)["json_schema"]["schema"]
        Draft202012Validator(native).validate(part)
        body.update(parse_response(json.dumps(part), schema))
    before = copy.deepcopy(body)
    report, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, "en"
    )
    assert body == before
    assert sum(len(p["sentences"]) for key in BODY for p in body[key]) == 16
    assert len(bindings["units"]) == 18
    for key in BODY:
        for paragraph in body[key]:
            for unit in paragraph["sentences"]:
                assert unit["text"] in report
    assert report.count("[E:b]") == 18


def test_summary_native_capacity_uses_the_same_existing_body_dependencies(case, scope):
    units = {"u1": {"text": "Source bound finding."}}
    _, payload, schema = summary_request_input(case.packet, units, scope)
    assert payload["body_units"] == units
    native = provider_response_format("summary", schema)["json_schema"]["schema"]
    value = {
        key: [
            {
                "sentences": [
                    {"text": f"{key} finding {i}.", "body_unit_refs": ["u1"]}
                    for i in range(2)
                ]
            }
        ]
        for key in ("abstract", "conclusion")
    }
    Draft202012Validator(native).validate(value)
    parse_response(json.dumps(value), schema)
    value["abstract"][0]["sentences"].append(
        {"text": "Extra finding.", "body_unit_refs": ["u1"]}
    )
    with pytest.raises(WriterError, match="writer_native_capacity_exceeded"):
        parse_response(json.dumps(value), schema)


def test_native_nonempty_and_scientific_support_are_still_host_checks(case):
    schema = body_section_schema("methods", case.aliases, maximum=3)
    native = provider_response_format("synthesis", schema)["json_schema"]["schema"]
    Draft202012Validator(native).validate({"methods": []})
    with pytest.raises(WriterError, match="writer_schema_invalid"):
        parse_response('{"methods": []}', schema)
    unsupported = {"methods": paragraphs(case, [1], "Unverified scientific claim")}
    Draft202012Validator(native).validate(unsupported)
    assert parse_response(json.dumps(unsupported), schema) == unsupported


def test_envelope_omitted_preserves_legacy_schema_and_builders(case):
    _, _, schema = chapter_request_input(
        case.packet, case.aliases, "methods", {}, {}, RANGE_WRITING_PLAN
    )
    assert schema == body_section_schema("methods", case.aliases)
    assert "prefixItems" not in json.dumps(
        provider_response_format("synthesis", schema)
    )
    _, _, summary = summary_request_input(case.packet, {"u1": {}})
    assert summary == schema_for("summary", {"u1": {}})
    assert "prefixItems" not in json.dumps(provider_response_format("summary", summary))


def test_keyword_text_cannot_enable_new_projection_on_a_legacy_schema():
    schema = body_section_schema("methods", ["prefixItems"])
    native = provider_response_format("synthesis", schema)["json_schema"]["schema"]
    assert "$defs" not in native
    assert native["properties"]["methods"]["items"]["properties"]["sentences"]["items"][
        "properties"
    ]["fact_card_refs"]["items"]["enum"] == ["prefixItems"]


def test_native_slots_share_schema_without_losing_reference_enums(case):
    schema = body_section_schema("methods", case.aliases, maximum=16)
    native = provider_response_format("synthesis", schema)["json_schema"]["schema"]
    unit = native["$defs"]["grounded_unit"]
    assert unit["properties"]["fact_card_refs"]["items"]["enum"] == list(case.aliases)

    def expand(node):
        if isinstance(node, list):
            return [expand(child) for child in node]
        if isinstance(node, dict):
            if node == {"$ref": "#/$defs/grounded_unit"}:
                return copy.deepcopy(unit)
            return {key: expand(value) for key, value in node.items() if key != "$defs"}
        return node

    expanded = expand(native)
    assert len(json.dumps(native)) < len(json.dumps(expanded))
    for sizes in ([3, 3, 3, 3, 2, 2], [3, 3, 3, 3, 2, 3]):
        value = {"methods": paragraphs(case, sizes, "Conditions")}
        assert Draft202012Validator(native).is_valid(value) == Draft202012Validator(
            expanded
        ).is_valid(value)
    invalid = {"methods": paragraphs(case, [1], "Unknown source")}
    invalid["methods"][0]["sentences"][0]["fact_card_refs"] = ["unknown"]
    assert not Draft202012Validator(native).is_valid(invalid)
    assert not Draft202012Validator(expanded).is_valid(invalid)


@pytest.mark.asyncio
async def test_actual_sdk_sends_bounded_schema_and_consumes_original_values(
    case, scope, tmp_path
):
    sent = []
    part = {"methods": paragraphs(case, [2, 2, 2, 2], "Condition")}
    before = copy.deepcopy(part)
    system, payload, schema = chapter_request_input(
        case.packet, case.aliases, "methods", {}, {}, RANGE_WRITING_PLAN, scope
    )
    expected = provider_response_format("synthesis", schema)["json_schema"]["schema"]

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        assert body["text"]["format"]["schema"] == expected
        assert body["text"]["format"]["strict"] is True
        assert body["thinking"] == {"type": "disabled"}
        assert json.loads(body["input"][1]["content"])["sources"] == payload["sources"]
        Draft202012Validator(expected).validate(part)
        return httpx.Response(200, json=response_envelope(json.dumps(part)))

    owner, budget, _ = await owner_for(handler)
    session = WriterSession(
        owner,
        tmp_path / "native.json",
        "native-capacity",
        case.packet,
        case.aliases,
        provider_api="responses",
    )
    try:
        result = await session.call("body:methods", system, payload, schema)
        assert result == before and part == before
        assert len(sent) == 1 and budget.used == 120
    finally:
        await owner.close()
