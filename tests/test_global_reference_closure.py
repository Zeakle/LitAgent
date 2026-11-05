"""Close every global ID permission without altering scientific or legacy gates."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_global_permissions import actual_schema, global_payload  # noqa: F401
from test_native_responses import owner_for

from litagent.body_quality import ROLES, validate_analysis_decisions
from litagent.eval import batched, global_permissions
from litagent.eval.responsibilities import parse_global_response
from litagent.eval.results import contract_identity
from litagent.llm.routing import ModelBoundClient
from litagent.safety.budget import strict_request_reservation


@pytest.fixture
def reference_case(global_payload):
    """Provide two scientific and four non-scientific claims with real host owners."""
    base = copy.deepcopy(global_payload)
    base["coverage_requirements"] = [
        {
            "requirement_id": "question",
            "kind": "question",
            "text": "Explain mechanisms.",
        }
    ]
    base["unit_inventory"] = []
    value = batched.build_global_payload(base, base)
    return value


def reply(payload):
    """Create identity controls, not a scientific gold-standard assessment."""
    return {
        "units": [
            {
                "source_unit_id": f"v{i}",
                "complete": True,
                "explanation": "Identity-only control.",
            }
            for i in range(1, len(payload["source_units"]) + 1)
        ],
        "conflicts": [],
        "issues": [],
        "coverage": [
            {
                "requirement_id": r["requirement_id"],
                "answered": False,
                "claim_ids": [],
                "explanation": "No supported answer is asserted by this control.",
            }
            for r in payload["coverage_requirements"]
        ],
        "limitations": {
            "adequate": False,
            "claim_ids": [],
            "explanation": "Unverified limitations.",
        },
        "analyses": [
            {
                "section_key": k,
                "adequate": False,
                "claim_ids": [],
                "explanation": "Unverified analysis.",
            }
            for k in ROLES
        ],
    }


def validate(schema, value):
    """Resolve shared definitions against the complete actual provider schema."""
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema).is_valid(value)


@pytest.mark.parametrize(
    "region,field",
    [
        ("units", "source_unit_id"),
        ("conflicts", "claim_a"),
        ("conflicts", "claim_b"),
        ("coverage", "requirement_id"),
        ("coverage", "claim_ids"),
        ("limitations", "claim_ids"),
        ("analyses", "section_key"),
        ("analyses", "claim_ids"),
    ],
)
@pytest.mark.parametrize("bad", ["2", "a208", "claim_1", "v999"])
def test_every_identity_path_rejects_unknowns_at_actual_native_boundary(
    reference_case, region, field, bad
):
    _, _, schema = actual_schema(reference_case)
    value = reply(reference_case)
    if region == "conflicts":
        value["conflicts"] = [
            {"claim_a": "a1", "claim_b": "a4", "reason": "Identity control."}
        ]
    target = value[region] if region == "limitations" else value[region][0]
    target[field] = [bad] if field == "claim_ids" else bad
    assert not validate(schema, value)
    with pytest.raises(
        ValueError, match="identity_invalid|alias_invalid|coverage_invalid"
    ):
        result = parse_global_response(json.dumps(value), reference_case, {})
        if region == "analyses" and field == "section_key":
            validate_analysis_decisions(
                result.analyses,
                reference_case["assertions"],
                [],
                reference_case["report_structure"],
            )


@pytest.mark.parametrize("bad", ["2", "a208", "v2", "claim_1"])
def test_064_reference_fault_is_not_repaired_or_given_a_valid_score(
    reference_case, bad
):
    path = Path(__file__).parent / "fixtures/run064-global-reference.json"
    raw = path.read_bytes()
    fixture = json.loads(raw)
    assert fixture["scientific_gold_standard"] is False
    assert fixture["actual_id"] == "2"
    assert fixture["path"] == ["coverage", 0, "claim_ids", 1]
    value = reply(reference_case)
    value["coverage"][0].update(answered=True, claim_ids=["a1", bad])
    _, _, schema = actual_schema(reference_case)
    assert not validate(schema, value)
    with pytest.raises(ValueError, match="claim_alias_invalid"):
        parse_global_response(json.dumps(value), reference_case, {})
    assert path.read_bytes() == raw


def test_complete_reference_control_preserves_partial_negative_science(reference_case):
    _, _, schema = actual_schema(reference_case)
    value = reply(reference_case)
    value["coverage"][0].update(answered=True, claim_ids=["a1", "a4"])
    value["conflicts"] = [
        {
            "claim_a": "a1",
            "claim_b": "a4",
            "reason": "Contradictory scientific findings.",
        }
    ]
    assert validate(schema, value)
    result = parse_global_response(json.dumps(value), reference_case, {})
    assert result.conflicts[0].claim_a == reference_case["assertions"][0]["claim_id"]
    assert result.limitations.adequate is False
    assert not batched.valid_limitations(
        result, reference_case["assertions"], [], reference_case["report_structure"]
    )


@pytest.mark.parametrize("region", ["coverage", "conflicts"])
def test_existing_organization_ids_cannot_be_scientific_witnesses(
    reference_case, region
):
    value = reply(reference_case)
    if region == "coverage":
        value["coverage"][0].update(answered=True, claim_ids=["a2"])
    else:
        value["conflicts"] = [
            {
                "claim_a": "a1",
                "claim_b": "a2",
                "reason": "Invalid classification control.",
            }
        ]
    assert not validate(actual_schema(reference_case)[2], value)


def test_chapter_permissions_stay_strict_even_for_known_scientific_ids(reference_case):
    value = reply(reference_case)
    _, _, schema = actual_schema(reference_case)
    assert validate(schema, value)
    value["limitations"].update(adequate=True, claim_ids=["a1"])
    assert not validate(schema, value)
    value = reply(reference_case)
    value["analyses"][0].update(adequate=True, claim_ids=["a1"])
    assert not validate(schema, value)


def test_empty_science_and_requirement_directories_allow_only_negative_empty_results(
    reference_case,
):
    for claim in reference_case["assertions"]:
        claim["kind"] = "report_structure"
        claim["evidence"] = []
    reference_case["coverage_requirements"] = []
    reference_case["unit_inventory"] = []
    payload = batched.build_global_payload(reference_case, reference_case)
    _, _, schema = actual_schema(payload)
    value = reply(payload)
    assert validate(schema, value)
    value["conflicts"] = [
        {"claim_a": "a1", "claim_b": "a2", "reason": "No science exists."}
    ]
    assert not validate(schema, value)
    value = reply(payload)
    value["coverage"] = [
        {
            "requirement_id": "question",
            "answered": False,
            "claim_ids": [],
            "explanation": "Unknown requirement.",
        }
    ]
    assert not validate(schema, value)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_unit",
        "duplicate_claim",
        "duplicate_requirement",
        "bool_id",
        "unknown_kind",
        "changed_owner",
        "changed_chapter_catalog",
        "extra_structure_unit",
    ],
)
def test_bad_host_catalogs_fail_before_transmission(reference_case, fault):
    if fault == "duplicate_unit":
        reference_case["source_units"].append(
            copy.deepcopy(reference_case["source_units"][0])
        )
    elif fault == "duplicate_claim":
        reference_case["assertions"].append(
            copy.deepcopy(reference_case["assertions"][0])
        )
    elif fault == "duplicate_requirement":
        reference_case["coverage_requirements"] *= 2
    elif fault == "bool_id":
        reference_case["assertions"][0]["claim_id"] = True
    elif fault == "unknown_kind":
        reference_case["assertions"][0]["kind"] = "approved_science"
    elif fault == "changed_owner":
        reference_case["assertions"][0]["source_unit_id"] = "unknown"
    elif fault == "extra_structure_unit":
        reference_case["report_structure"].append(
            {"source_unit_id": "unknown", "section": "Methods"}
        )
    else:
        reference_case["analysis_claim_ids_by_section"]["methods"] = ["unknown"]
    with pytest.raises(ValueError, match="permission_schema_invalid"):
        global_permissions._reference_catalogs(reference_case)


def test_native_definitions_are_local_acyclic_and_deduplicate_exact_directories(
    reference_case,
):
    before = copy.deepcopy(reference_case)
    messages, native, schema = actual_schema(reference_case)
    assert reference_case == before
    assert global_permissions.REFERENCE_INSTRUCTION in messages[0]["content"]

    def visit(node, parents=()):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                assert ref.startswith("#/$defs/") and ref not in parents
                visit(schema["$defs"][ref.split("/")[-1]], (*parents, ref))
            for key, child in node.items():
                if key != "$defs":
                    visit(child, parents)
        elif isinstance(node, list):
            for child in node:
                visit(child, parents)

    visit(schema)
    definitions = [
        json.dumps(v, sort_keys=True) for v in schema["$defs"].values() if "enum" in v
    ]
    assert len(definitions) == len(set(definitions))
    assert strict_request_reservation(messages, 32768, native) < 120000
    # This is the exact pre-change result identity: only request permissions change.
    assert (
        contract_identity()
        == "a4273caa2f6f848615de3ea2a7dcff7ce6244f6e31408ceab84f82d4a1c65b2f"
    )


@pytest.mark.asyncio
async def test_actual_sdk_sends_the_measured_prompt_and_shared_native_schema(
    reference_case,
):
    messages, native, schema = actual_schema(reference_case)
    value = reply(reference_case)
    observed = []

    async def transport(request):
        wire = json.loads(request.content)
        observed.append(wire)
        assert wire["messages"] == messages
        assert wire["response_format"] == native
        assert validate(wire["response_format"]["json_schema"]["schema"], value)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "lite",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(value)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, budget, _ = await owner_for(transport)
    try:
        result = await ModelBoundClient(owner, "lite", "evaluation").chat(
            messages, max_tokens=32768, response_format=native, thinking_mode="disabled"
        )
        assert len(observed) == 1
        assert result.usage["usage_available"]
        assert budget.snapshot()["requests_sent"] == 1
        assert budget.used == 120
    finally:
        await owner.close()


def test_valid_chapter_witnesses_do_not_grant_other_chapters_permission(reference_case):
    """Preserve positive ownership controls as well as wrong-chapter negatives."""
    from litagent.review_structure import structure_sections

    titles = dict(structure_sections("academic_v2", "en"))
    reference_case["report_structure"][0]["section"] = titles["methods"]
    reference_case["report_structure"][1]["section"] = titles["open_problems"]
    reference_case["unit_inventory"] = []
    payload = batched.build_global_payload(reference_case, reference_case)
    value = reply(payload)
    for row in value["analyses"]:
        if row["section_key"] in {"methods", "open_problems"}:
            row.update(
                adequate=True,
                claim_ids=["a1" if row["section_key"] == "methods" else "a4"],
            )
    value["limitations"].update(adequate=True, claim_ids=["a4"])
    _, _, schema = actual_schema(payload)
    assert validate(schema, value)
    parsed = parse_global_response(json.dumps(value), payload, {})
    ids = [payload["assertions"][i]["claim_id"] for i in (0, 3)]
    assert batched.valid_limitations(
        parsed, payload["assertions"], ids, payload["report_structure"]
    )
    next(row for row in value["analyses"] if row["section_key"] == "methods")[
        "claim_ids"
    ] = ["a4"]
    assert not validate(schema, value)
    parsed = parse_global_response(json.dumps(value), payload, {})
    checks = validate_analysis_decisions(
        parsed.analyses, payload["assertions"], ids, payload["report_structure"]
    )
    method = next(row for row in checks["sections"] if row["section_key"] == "methods")
    assert method["adequate"] is False
    assert "analysis_wrong_section_claim" in method["host_validation"]["failure_codes"]


@pytest.mark.parametrize(
    "fault",
    [
        "conflict_type",
        "coverage_items",
        "limitations_type",
        "chapter_key",
        "chapter_count",
    ],
)
def test_changed_native_producer_fails_instead_of_weakening_permissions(
    reference_case, fault
):
    from test_global_permissions import historical_parent

    old = historical_parent(reference_case)
    native = actual_schema(old)[1]
    schema = native["json_schema"]["schema"]
    from litagent.eval.decision_output import _resolve

    if fault == "conflict_type":
        _resolve(schema, schema["properties"]["conflicts"]["items"])["properties"][
            "claim_a"
        ]["type"] = "integer"
    elif fault == "coverage_items":
        _resolve(schema, schema["properties"]["coverage"]["items"])["properties"][
            "claim_ids"
        ]["items"] = {"type": "string", "enum": ["undeclared"]}
    elif fault == "limitations_type":
        _resolve(schema, schema["properties"]["limitations"])["properties"][
            "claim_ids"
        ]["type"] = "string"
    elif fault == "chapter_key":
        schema["properties"]["analyses"]["items"]["anyOf"][0]["properties"][
            "section_key"
        ]["const"] = "unknown"
    else:
        schema["properties"]["analyses"]["items"]["anyOf"].pop()
    with pytest.raises(ValueError, match="permission_schema_invalid"):
        global_permissions.project_format(native, reference_case)


def test_real_064_old_owner_remains_readable_and_scientifically_unverified():
    from litagent.eval.results import EvaluationRunResult
    from litagent.safety.planned_accounting import _envelope
    from litagent.safety.planned_budget import validate_writer_envelope

    archive = Path(
        "artifacts/runs/frozen-writer-three-domain-20261105-064-few-shot-metric.json"
    )
    receipt = Path(
        "artifacts/acceptance/frozen-writer-three-domain-20261105-064/terminal-check.json"
    )
    if not archive.exists() or not receipt.exists():
        pytest.skip("Private real-run materials are not distributed")
    raw = archive.read_bytes()
    sha = json.loads(receipt.read_text())["authenticated_receipts"][0]["archives"][
        str(archive)
    ]
    assert hashlib.sha256(raw).hexdigest() == sha
    report = json.loads(raw)["report"]
    details = report["evaluation"]["faithfulness"]["details"]
    model = EvaluationRunResult.from_details(
        details, hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    assert model is not None
    assert model.metrics()["faithfulness"]["score"] is None
    assert model.metrics()["internal_consistency"]["score"] is None
    owner = details["stages"]["writer_envelope"]
    _envelope(owner)
    with pytest.raises(ValueError, match="assessment_decision"):
        validate_writer_envelope(owner)
    assert archive.read_bytes() == raw
