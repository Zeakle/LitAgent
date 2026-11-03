"""Keep nonempty wire collections reversible across real writer schemas."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from litagent import writing_native as native
from litagent import writing_v2 as writer


def dumped(value):
    """Serialize synthetic authored values without changing their contents."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sentence(text="A supplied finding.", refs=("c1",), field="fact_card_refs"):
    """Make a structural fixture without claiming scientific support."""
    return {"text": text, field: list(refs)}


@pytest.mark.parametrize("maximum", range(1, 67))
def test_all_bounded_slots_roundtrip_at_the_declared_total_capacity(maximum):
    """Later heterogeneous slots retain their own capacity and canonical order."""
    schema = writer.body_section_schema("methods", ["c1", "c2"], maximum=maximum)
    before = copy.deepcopy(schema)
    paragraphs = []
    for pi, slot in enumerate(schema["properties"]["methods"]["prefixItems"]):
        capacity = len(slot["properties"]["sentences"]["prefixItems"])
        paragraphs.append(
            {
                "sentences": [
                    sentence(f'条件 α {pi}:{si} 保留 "quoted" 与 \\。', ("c2", "c1"))
                    for si in range(capacity)
                ]
            }
        )
    original = {"methods": paragraphs}
    wire = native.encode(original, schema)
    result = native.parse(dumped(wire), schema)
    assert result["canonical"] == original
    assert len(result["bindings"]["units"]) == maximum
    assert sum(len(p["sentences"]) for p in original["methods"]) == maximum
    assert schema == before
    assert result["bindings"]["contract"] == native.CONTRACT == "required-head-wire-v1"
    assert result["bindings"]["scientific_assessment"] == "not_verified"


def test_maximum_one_requires_head_and_permits_only_an_empty_tail():
    """The provider schema itself excludes the previously accepted empty chapter."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    prepared = native.prepare("synthesis", schema)
    validator = Draft202012Validator(prepared["native_schema"])
    wire = native.encode({"taxonomy": [{"sentences": [sentence()]}]}, schema)
    assert wire["taxonomy"]["tail"] == []
    assert wire["taxonomy"]["head"]["sentences"]["tail"] == []
    assert wire["taxonomy"]["head"]["sentences"]["head"]["fact_card_refs"]["tail"] == []
    validator.validate(wire)
    for value in (
        {"taxonomy": []},
        {"taxonomy": {"tail": []}},
        {"taxonomy": {"head": None, "tail": []}},
    ):
        assert not validator.is_valid(value)
    assert "empty array" in native.layout(prepared["native_schema"])


@pytest.mark.parametrize("summary_repair", [False, True])
def test_repairs_preserve_legal_empty_patches_and_reference_field(summary_repair):
    """Declining stays possible while existing patches retain their dependencies."""
    refs = ["u1", "u2"] if summary_repair else ["c1", "c2"]
    field = "body_unit_refs" if summary_repair else "fact_card_refs"
    schema = writer.schema_for(
        "repair", refs, ["target"], summary_repair=summary_repair
    )
    declined = {
        "outcome": "declined",
        "explanation": "Insufficient support.",
        "patches": [],
    }
    patched = {
        "outcome": "patched",
        "explanation": "Synthetic structure only.",
        "patches": [
            {"target_id": "target", "sentences": [sentence(refs=refs, field=field)]}
        ],
    }
    for value in (declined, patched):
        wire = native.encode(value, schema)
        assert isinstance(wire["patches"], list)
        assert native.parse(dumped(wire), schema)["canonical"] == value


def test_summary_and_reviewer_keep_their_real_canonical_consumers():
    """Summary roots preserve body dependencies and Reviewer can report no issues."""
    schema = writer.schema_for("summary", ["u1", "u2"])
    value = {
        section: [{"sentences": [sentence(section, ("u2", "u1"), "body_unit_refs")]}]
        for section in ("abstract", "conclusion")
    }
    parsed = native.parse(dumped(native.encode(value, schema)), schema)
    assert parsed["canonical"] == value
    assert {tuple(unit["canonical_path"]) for unit in parsed["bindings"]["units"]} == {
        ("abstract", 0, "sentences", 0),
        ("conclusion", 0, "sentences", 0),
    }
    schema = writer._closed(
        {
            "issues": writer._array(
                writer._closed({"unit_id": {"type": "string", "enum": ["u1"]}}),
                minimum=0,
            ),
            "explanation": {"type": "string"},
        }
    )
    value = {"issues": [], "explanation": "No synthetic issue."}
    assert native.encode(value, schema) == value
    assert native.parse(dumped(value), schema)["canonical"] == value


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "writer_schema_invalid"),
        ("budget" + "x" * 1998, "writer_schema_invalid"),
        ("   ", "writer_text_invalid"),
        ("A\nB", "writer_text_invalid"),
        ("Finding [E:pretend].", "writer_text_invalid"),
        ("# Heading", "writer_text_invalid"),
    ],
)
def test_native_structure_does_not_replace_text_policy(text, code):
    """A provider-valid free string still passes the complete host boundary."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    wire = native.encode({"taxonomy": [{"sentences": [sentence()]}]}, schema)
    wire["taxonomy"]["head"]["sentences"]["head"]["text"] = text
    Draft202012Validator(native.prepare("synthesis", schema)["native_schema"]).validate(
        wire
    )
    with pytest.raises(writer.WriterError) as failure:
        native.parse(dumped(wire), schema)
    assert failure.value.code == code
    assert failure.value.family == "protocol"


def test_duplicate_references_reach_host_without_being_deduplicated():
    """Transport cannot turn a duplicate reference sequence into valid content."""
    schema = writer.body_section_schema("taxonomy", ["c1", "c2"], maximum=1)
    wire = native.encode({"taxonomy": [{"sentences": [sentence()]}]}, schema)
    refs = wire["taxonomy"]["head"]["sentences"]["head"]["fact_card_refs"]
    refs["tail"] = ["c1"]
    Draft202012Validator(native.prepare("synthesis", schema)["native_schema"]).validate(
        wire
    )
    with pytest.raises(writer.WriterError, match="writer_schema_invalid"):
        native.parse(dumped(wire), schema)
    with pytest.raises(ValidationError):
        native.encode(
            {"taxonomy": [{"sentences": [sentence(refs=("c1", "c1"))]}]}, schema
        )


@pytest.mark.parametrize("minimum", [2, -1, True, 1.0])
def test_unsupported_minimum_is_a_stable_contract_error(minimum):
    """A malformed contract fails before either request formatting or decoding."""
    schema = writer._closed(
        {
            "taxonomy": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": minimum,
                "maxItems": 3,
            }
        }
    )
    with pytest.raises(writer.WriterError) as failure:
        native.prepare("synthesis", schema)
    assert failure.value.code == "writer_native_wire_contract_invalid"
    assert failure.value.diagnostics[0]["phase"] == "contract"
    assert failure.value.diagnostics[0]["schema_path"]


def test_unbounded_and_invalid_host_schemas_have_original_contract_paths():
    """Invalid schemas never leak a library error into the live formatter."""
    unbounded = writer._closed(
        {"taxonomy": {"type": "array", "items": {"type": "string"}, "minItems": 1}}
    )
    with pytest.raises(writer.WriterError) as failure:
        native.prepare("synthesis", unbounded)
    assert failure.value.diagnostics[0]["schema_path"] == ["properties", "taxonomy"]
    with pytest.raises(writer.WriterError) as failure:
        native.prepare("synthesis", writer._closed({"x": {"type": "bad-type"}}))
    assert failure.value.diagnostics[0]["schema_path"] == ["properties", "x", "type"]


def test_prepared_schemas_are_isolated_and_default_projection_does_not_reenter(
    monkeypatch,
):
    """An explicit adapter never changes the provider's default projection path."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    calls = []
    original = writer.provider_response_format

    def observed(stage, host_schema, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("wire_contract") is None
        return original(stage, host_schema, **kwargs)

    native._prepare.cache_clear()
    monkeypatch.setattr(writer, "provider_response_format", observed)
    first = native.prepare("synthesis", schema)
    first["native_schema"].clear()
    second = native.prepare("synthesis", schema)
    assert second["native_schema"] and calls == [{}]
    assert native.layout(second["native_schema"]).startswith("Native output type map")


@pytest.mark.parametrize(
    "bad_schema,path",
    [
        ({"type": "object"}, ["additionalProperties"]),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": True,
            },
            ["additionalProperties"],
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            ["required"],
        ),
        (
            writer._closed({"nested": {"type": "object"}}),
            ["properties", "nested", "additionalProperties"],
        ),
    ],
)
def test_open_or_optional_objects_are_rejected_before_value_mapping(bad_schema, path):
    """Every admitted object property must have an explicit required codec path."""
    with pytest.raises(writer.WriterError) as failure:
        native.prepare("synthesis", bad_schema)
    assert failure.value.code == "writer_native_wire_contract_invalid"
    assert failure.value.diagnostics[0]["schema_path"] == path
    with pytest.raises(writer.WriterError) as parsed:
        native.parse("{}", bad_schema)
    assert parsed.value.code == "writer_native_wire_contract_invalid"
    assert parsed.value.diagnostics[0]["schema_path"] == path


@pytest.mark.parametrize("entrypoint", ["formatter", "messages"])
@pytest.mark.parametrize(
    "bad_schema", [writer._closed({"x": {"type": "bad-type"}}), {"type": "object"}]
)
def test_new_wire_request_entrypoints_normalize_schema_errors(entrypoint, bad_schema):
    """Formatter and prompt production share the helper's checked boundary."""
    with pytest.raises(writer.WriterError) as failure:
        if entrypoint == "formatter":
            writer.provider_response_format(
                "synthesis", bad_schema, wire_contract=native.CONTRACT
            )
        else:
            writer.request_messages(
                "Write sources only.", {}, bad_schema, wire_contract=native.CONTRACT
            )
    assert failure.value.code == "writer_native_wire_contract_invalid"
    assert failure.value.diagnostics[0]["phase"] == "contract"
    assert failure.value.diagnostics[0]["schema_path"]


@pytest.mark.parametrize("entrypoint", ["formatter", "messages"])
@pytest.mark.parametrize(
    "unknown_tag", ["required-head-wire-v0", "required-head-wire-v2", ""]
)
def test_unknown_wire_tag_is_rejected_by_both_request_entrypoints(
    entrypoint, unknown_tag
):
    """An unrecognized transport cannot silently select the current codec."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    with pytest.raises(writer.WriterError, match="writer_native_wire_contract_invalid"):
        if entrypoint == "formatter":
            writer.provider_response_format(
                "synthesis", schema, wire_contract=unknown_tag
            )
        else:
            writer.request_messages(
                "Write sources only.", {}, schema, wire_contract=unknown_tag
            )
