"""Keep target-local edit positions strict through the real writer transport."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_capacity_repair import OverflowInventoryGateway, run_capacity_case
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_support_receipts_pipeline import SELECTION, select_frozen
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_repair import (
    capacity_targets,
    edit_schema,
    normalize_edit_map,
    original_unit_directory,
    validate_edit_records,
)
from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped, support_validation_identity
from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest, strict_acceptance
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writing_flat import (
    CONTRACT,
    TRANSPORT_REPAIR_INSTRUCTION,
    parse,
    prepare,
    reference_set_identity,
    transport_identity,
)
from litagent.writing_v2 import (
    WriterError,
    compile_document,
    provider_response_format,
    request_messages,
    schema_for,
    writer_budget_plan,
)

FIXTURE = Path(__file__).with_name("fixtures") / "target_local_edit_042.json"
EXAMPLE = (
    "Example: r1 and r2 each returning one unit use [0] for both, never [0]/[1]. "
    "Two originals merged into r2's sole unit both use [0]."
)


@pytest.fixture
def structural_042():
    """Load sanitized positions, never private archives or scientific gold."""
    return json.loads(FIXTURE.read_text())


def contract_for(body, *, maximum=66):
    """Construct the unchanged schema with its static capacity-wide enum."""
    targets = capacity_targets(body, {}, maximum=maximum)
    schema = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=True
    )
    return targets, schema


def local_result(fixture):
    """Create a separate positive, never normalize the failed response."""
    response = copy.deepcopy(fixture["raw_response"])
    response["explanation"] = "Independent legal target-local test response."
    for record in response["edit_records"]:
        record["replacement_indexes"] = [0]
    return response


def parse_wire(response, schema):
    """Retain real flat bindings before invoking the unchanged map consumer."""
    return parse(
        json.dumps(response),
        schema,
        diagnostic_stage="repair",
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
    )


def test_original_042_structure_native_shape_passes_but_local_guard_rejects(
    structural_042,
):
    """Preserve 47 bad records and the first r2 failure without auto-renumbering."""
    fixture = structural_042
    targets, schema = contract_for(fixture["body"])
    response = fixture["raw_response"]
    before = copy.deepcopy(response)
    directory = original_unit_directory(targets)
    assert len(targets) == len(response["patches"]) == 38
    assert [len(ids) for ids in directory.values()] == fixture["target_original_counts"]
    assert sum(map(len, directory.values())) == len(response["edit_records"]) == 48
    assert schema["properties"]["edit_records"]["items"]["properties"][
        "replacement_indexes"
    ]["items"]["enum"] == list(range(66))
    provider = provider_response_format("repair", schema, wire_contract=CONTRACT)
    Draft202012Validator(provider["json_schema"]["schema"]).validate(response)
    parsed = parse_wire(response, schema)
    assert parsed["canonical"]["edit_records"] == response["edit_records"]
    assert parsed["bindings"]["scientific_assessment"] == "not_verified"
    owner = {uid: tid for tid, ids in directory.items() for uid in ids}
    wrong = [
        record
        for record in response["edit_records"]
        if any(index != 0 for index in record["replacement_indexes"])
    ]
    assert len(wrong) == 47
    assert owner[wrong[0]["unit_id"]] == "r2"
    assert wrong[0]["unit_id"] == "writer:introduction:1:0"
    assert wrong[0]["replacement_indexes"] == [1]
    with pytest.raises(WriterError, match="writer_capacity_edit_position_invalid"):
        normalize_edit_map(parsed["canonical"], targets)
    assert response == before


def test_two_targets_and_merged_originals_each_use_their_own_zero(structural_042):
    """Allow repeated zero across targets and many originals within one target."""
    targets, schema = contract_for(structural_042["body"])
    parsed = parse_wire(local_result(structural_042), schema)
    normalized = normalize_edit_map(parsed["canonical"], targets)
    validate_edit_records(normalized, targets)
    assert all(r["replacement_indexes"] == [0] for r in normalized["edit_records"])
    assert (
        len(
            [
                r
                for r in normalized["edit_records"]
                if r["unit_id"].startswith("writer:methods:0:")
            ]
        )
        == 2
    )
    merged = [
        r
        for r in normalized["edit_records"]
        if r["unit_id"].startswith("writer:methods:3:")
    ]
    assert len(merged) == 3
    assert all(record["action"] == "rewritten" for record in merged)
    assert normalized["removed_targets"] == []
    assert parsed["bindings"]["scientific_assessment"] == "not_verified"


def test_two_local_units_deletion_and_untouched_targets(structural_042):
    """Keep local order, explicit removal and untouched byte preservation distinct."""
    targets, schema = contract_for(structural_042["body"])
    directory = original_unit_directory(targets)
    raw = {
        "outcome": "patched",
        "explanation": "Only r11 is changed in this independent test response.",
        "patches": [
            {
                "target_id": "r11",
                "text": f"Distinct local replacement {i}.",
                "fact_card_refs": ["c1"],
            }
            for i in range(2)
        ],
        "edit_records": [
            {
                "unit_id": uid,
                "reason": "Preserve distinct positions.",
                "replacement_indexes": [i],
            }
            for i, uid in enumerate(directory["r11"])
        ],
    }
    normalized = normalize_edit_map(parse_wire(raw, schema)["canonical"], targets)
    assert [r["replacement_indexes"] for r in normalized["edit_records"][:2]] == [
        [0],
        [1],
    ]
    assert (
        next(p for p in normalized["patches"] if p["target_id"] == "r2")["sentences"]
        == targets["r2"]["original"]
    )
    raw["patches"] = []
    for record in raw["edit_records"]:
        record["replacement_indexes"] = []
    removed = normalize_edit_map(parse_wire(raw, schema)["canonical"], targets)
    assert removed["removed_targets"] == ["r11"]
    assert all(r["action"] == "removed" for r in removed["edit_records"][:2])


@pytest.mark.parametrize(
    "fault,code",
    [
        ("global_index", "writer_capacity_edit_position_invalid"),
        ("borrow_other_target_index", "writer_capacity_edit_position_invalid"),
        ("duplicate_index", "writer_capacity_edit_position_invalid"),
        ("missing_original", "writer_capacity_edit_coverage_invalid"),
        ("unmapped_new_unit", "writer_capacity_new_unit_unaccounted"),
    ],
)
def test_host_rejects_bad_relations_without_changing_indexes(
    structural_042, fault, code
):
    """Check ownership against the actual target, never a neighboring patch."""
    targets, schema = contract_for(structural_042["body"])
    raw = local_result(structural_042)
    if fault == "global_index":
        raw["edit_records"][1]["replacement_indexes"] = [1]
    elif fault == "borrow_other_target_index":
        raw["patches"].insert(
            2,
            {
                "target_id": "r2",
                "text": "Second local r2 unit.",
                "fact_card_refs": ["c1"],
            },
        )
        raw["edit_records"][0]["replacement_indexes"] = [1]
    elif fault == "duplicate_index":
        raw["edit_records"][0]["replacement_indexes"] = [0, 0]
    elif fault == "missing_original":
        raw["edit_records"].pop(0)
    else:
        raw["patches"].insert(
            1,
            {
                "target_id": "r1",
                "text": "Unaccounted new unit.",
                "fact_card_refs": ["c1"],
            },
        )
    parsed = parse_wire(raw, schema)
    before = copy.deepcopy(parsed["canonical"])
    with pytest.raises(WriterError, match=code):
        normalize_edit_map(parsed["canonical"], targets)
    assert parsed["canonical"] == before


@pytest.mark.parametrize("fault", ["unknown_original", "unknown_target", "bool_index"])
def test_native_parser_rejects_unknown_ids_and_bool_indexes(structural_042, fault):
    """Keep integer typing and frozen permissions ahead of map normalization."""
    _, schema = contract_for(structural_042["body"])
    raw = local_result(structural_042)
    if fault == "unknown_original":
        raw["edit_records"][0]["unit_id"] = "writer:methods:999:0"
    elif fault == "unknown_target":
        raw["patches"][0]["target_id"] = "unknown"
    else:
        raw["edit_records"][0]["replacement_indexes"] = [False]
    assert list(Draft202012Validator(prepare(schema)["wire_schema"]).iter_errors(raw))
    with pytest.raises(WriterError):
        parse_wire(raw, schema)


class TargetLocalGateway(FlatGateway):
    """Replace HTTP output only, preserving the actual SDK and writer consumers."""

    def __init__(self, case, response, **kwargs):
        """Store independent wire output at the mocked provider boundary."""
        super().__init__(case, **kwargs)
        self.response = copy.deepcopy(response)

    def answer(self, payload, native_schema):
        """Inject only capacity responses and delegate all other provider stages."""
        if payload.get("cause") == "capacity":
            return copy.deepcopy(self.response)
        return super().answer(payload, native_schema)


async def selected_session(case, tmp_path, gateway):
    """Lock the real complete v7 plan, including two shared repair opportunities."""
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **SELECTION,
    )
    assert plan["status"] == "admitted" and plan["repair_limit"] == 2
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    return owner, make_session(owner, case, tmp_path)


@pytest.mark.asyncio
async def test_actual_sdk_preserves_failure_then_uses_remaining_shared_attempt(
    case, tmp_path, structural_042
):
    """Exercise explicit primitives, not production retries after protocol failure.

    The second call is test-owned to check the remaining shared slot. The full
    runner test below separately proves production stops after invalid indexes.
    """
    case.body = copy.deepcopy(structural_042["body"])
    gateway = TargetLocalGateway(case, structural_042["raw_response"])
    owner, session = await selected_session(case, tmp_path, gateway)
    try:
        original = await session.initial_body()
        original_digest = digest(original)
        original_receipt = copy.deepcopy(session.reference_input(original))
        targets = capacity_targets(original, {}, maximum=66)
        with pytest.raises(WriterError, match="writer_capacity_edit_position_invalid"):
            await session.patch(original, targets, cause="capacity", claim_count=182)
        call = session.journal["calls"]["repair_0"]
        received = json.loads(call["raw"])
        assert (
            received["edit_records"] == structural_042["raw_response"]["edit_records"]
        )
        assert call["reference_canonical"]["edit_records"] == received["edit_records"]
        assert call["raw_sha256"] == hashlib.sha256(call["raw"].encode()).hexdigest()
        assert (
            original == structural_042["body"] and digest(original) == original_digest
        )
        assert session.reference_input(original) == original_receipt
        assert not session.journal.get("candidates")
        sent = gateway.sent[-1]
        prompt = sent["request"]["input"][0]["content"]
        assert prompt.count(EXAMPLE) == prompt.count(TRANSPORT_REPAIR_INSTRUCTION) == 1
        native_schema = {
            k: v for k, v in sent["request"]["text"]["format"].items() if k != "type"
        }["schema"]
        Draft202012Validator(native_schema).validate(received)
        assert (
            native_schema
            == provider_response_format(
                "repair", call["reference_schema"], wire_contract=CONTRACT
            )["json_schema"]["schema"]
        )
        assert call["wire_bindings"]["scientific_assessment"] == "not_verified"
        assert all(
            EXAMPLE not in row["request"]["input"][0]["content"]
            for row in gateway.sent[:-1]
        )
        gateway.response = local_result(structural_042)
        candidate = await session.patch(
            original, targets, cause="capacity", claim_count=182
        )
        assert (
            sum(
                len(paragraph["sentences"])
                for key, paragraphs in candidate.items()
                if key != "title"
                for paragraph in paragraphs
            )
            == 38
        )
        assert (
            original == structural_042["body"] and digest(original) == original_digest
        )
        assert session.reference_input(original) == original_receipt
        assert len(session.state["attempts"]) == 2
        count = len(gateway.sent)
        with pytest.raises(WriterError, match="repair_limit"):
            await session.patch(original, targets, cause="capacity", claim_count=182)
        assert len(gateway.sent) == count == 7
    finally:
        await owner.close()


class FullChainLocalGateway(OverflowInventoryGateway):
    """Inject only external edit indexes into the existing full runner fixture."""

    def __init__(self, case, *, global_indexes):
        """Choose a legal response or the deliberately invalid global-index fault."""
        super().__init__(case)
        self.global_indexes = global_indexes
        self.capacity_responses = []

    def answer(self, payload, native_schema):
        """Retain the established full workflow and alter only received positions."""
        result = super().answer(payload, native_schema)
        if payload.get("cause") != "capacity":
            return result
        if self.global_indexes:
            owners = {
                uid: ordinal
                for ordinal, target in enumerate(payload["targets"].values())
                for uid in target["original_unit_ids"]
            }
            for record in result["edit_records"]:
                record["replacement_indexes"] = [owners[record["unit_id"]]]
        self.capacity_responses.append(copy.deepcopy(result))
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("global_indexes", [False, True])
async def test_real_frozen_worker_report_and_automatic_gate_keep_local_positions(
    case, tmp_path, monkeypatch, global_indexes
):
    """Use the existing real consumer chain; mock only provider HTTP responses."""
    select_frozen(monkeypatch)
    gateway = FullChainLocalGateway(case, global_indexes=global_indexes)
    report, gateway = await run_capacity_case(
        case, tmp_path, monkeypatch, gateway=gateway
    )
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == len(gateway.capacity_responses) == 1
    capacity_calls = [
        row for row in gateway.sent if row["payload"].get("cause") == "capacity"
    ]
    assert len(capacity_calls) == 1
    sent = capacity_calls[0]
    assert sent["request"]["input"][0]["content"].count(EXAMPLE) == 1
    assert sent["request"]["thinking"] == {"type": "disabled"}
    assert any("report_layout" in row["payload"] for row in gateway.sent)
    assert all(
        EXAMPLE
        not in row["request"].get("input", row["request"].get("messages"))[0]["content"]
        for row in gateway.sent
        if row["payload"].get("cause") != "capacity"
    )
    if global_indexes:
        assert not report["delivery"]["publishable"]
        assert (
            report["metadata"]["writer_failure"]["code"]
            == "writer_capacity_edit_position_invalid"
        )
        assert attempts[0]["status"] == "failed"
        assert gateway.inventory_calls == 1
        assert not any("support_passages" in row["payload"] for row in gateway.sent)
        assert not any(
            item.get("committed")
            for item in report["metadata"]["evidence_rewrite_attempts"]
        )
    else:
        assert all(
            record["replacement_indexes"] in ([], [0])
            for record in gateway.capacity_responses[0]["edit_records"]
        )
        assert report["delivery"]["publishable"]
        details = report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["unsupported_count"] == details["unverified_count"] == 0
        assert gateway.inventory_calls == 2


@pytest.mark.parametrize(
    "policy", ["planned-budget-v5", "planned-budget-v6", "planned-budget-v7"]
)
def test_none_transport_keeps_plan_schema_and_prompt_unchanged(case, policy):
    """Keep inactive defaults absent and examples out of historical paths."""
    omitted = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy=policy
    )
    explicit = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy=policy,
        transport_revision=None,
    )
    assert omitted == explicit
    assert "transport_revision" not in omitted["envelope"]
    targets, schema = contract_for(case.body)
    payload = {"targets": targets, "writing_envelope": {}}
    plain = request_messages("Repair.", payload, schema, wire_contract=CONTRACT)
    assert EXAMPLE not in plain[0]["content"]
    selected = request_messages(
        "Repair.",
        {
            **payload,
            "writing_envelope": {"transport_revision": "host-lossless-transport-v1"},
        },
        schema,
        wire_contract=CONTRACT,
    )
    assert selected[0]["content"].count(EXAMPLE) == 1
    assert json.loads(plain[1]["content"]) == (
        json.loads(selected[1]["content"]) | {"writing_envelope": {}}
    )


@pytest.mark.parametrize(
    ("identity_field", "provenance_key", "expected_error"),
    [
        (
            "transport_identity",
            "source_transport_identity",
            "writer_envelope_transport_revision_invalid",
        ),
        (
            "support_validation_identity",
            "source_support_validation_identity",
            "writer_envelope_support_validation_invalid",
        ),
    ],
)
def test_old_transport_or_support_identity_is_independently_rejected(
    case, structural_042, identity_field, provenance_key, expected_error
):
    """Reject each stale selected identity while preserving reference identity."""
    provenance = structural_042["provenance"]
    assert (
        support_validation_identity()
        != provenance["source_support_validation_identity"]
    )
    assert reference_set_identity() == provenance["source_reference_identity"]
    assert transport_identity() != provenance["source_transport_identity"]
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **SELECTION,
    )
    assert validate_writer_envelope(plan["envelope"]) == plan["envelope"]
    envelope = copy.deepcopy(plan["envelope"])
    assert envelope["transport_identity"] == transport_identity()
    assert envelope["support_validation_identity"] == support_validation_identity()
    envelope[identity_field] = provenance[provenance_key]
    envelope["identity"] = digest(
        {key: value for key, value in envelope.items() if key != "identity"}
    )
    assert {key for key in envelope if envelope[key] != plan["envelope"][key]} == {
        identity_field,
        "identity",
    }
    with pytest.raises(ValueError) as error:
        validate_writer_envelope(envelope)
    assert str(error.value) == expected_error


@pytest.mark.asyncio
async def test_checkpoint_with_original_042_transport_receipt_cannot_resume(
    case, tmp_path, structural_042
):
    """Reject the pinned old receipt before another SDK request is sent."""
    gateway = TargetLocalGateway(case, {})
    owner, session = await selected_session(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        session.journal["calls"]["body:introduction"]["transport_identity"] = (
            structural_042["provenance"]["source_transport_identity"]
        )
        session.save()
        count = len(gateway.sent)
        with pytest.raises(
            WriterError, match="writer_reference_input_identity_invalid"
        ):
            restored = make_session(owner, case, tmp_path)
            await restored.summaries(body, "summary_initial")
        assert len(gateway.sent) == count == 5
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_legal_map_never_exempts_wrong_paper_from_scientific_evaluation(
    case, tmp_path
):
    """Run actual v3 extraction/assessment after a locally valid false owner edit."""
    targets = capacity_targets(case.body, {}, maximum=66)
    unit_id = original_unit_directory(targets)["r3"][0]
    response = {
        "outcome": "patched",
        "explanation": "Controlled unsupported paper-owner swap.",
        "patches": [
            {
                "target_id": "r3",
                "text": "Evo-ViT learns the relation-network comparison function.",
                "fact_card_refs": [case.by_eid["a"]],
            }
        ],
        "edit_records": [
            {
                "unit_id": unit_id,
                "reason": "Controlled scientific counterexample.",
                "replacement_indexes": [0],
            }
        ],
    }
    gateway = TargetLocalGateway(case, response, unsupported=True)
    owner, session = await selected_session(case, tmp_path, gateway)
    try:
        original = await session.initial_body()
        original_digest = digest(original)
        original_receipt = copy.deepcopy(session.reference_input(original))
        candidate = await session.patch(
            original, targets, cause="capacity", claim_count=5
        )
        assert "Evo-ViT" in candidate["methods"][0]["sentences"][0]["text"]
        summaries = await session.summaries(candidate, "summary_after_repair")
        survey, bindings = compile_document(
            candidate,
            summaries,
            case.cards,
            case.aliases,
            "en",
            envelope=session.envelope,
        )
        await session.call(
            "reviewer",
            "Review the complete candidate.",
            {"report_layout": []},
            {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 128,
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["issues", "explanation"],
                "additionalProperties": False,
            },
            stage="reviewer",
            max_tokens=4096,
        )
        details = await evaluate_scoped(
            llm=ModelBoundClient(owner, "lite", "evaluation"),
            survey=survey,
            context={
                "query": case.packet["query"],
                "quality_policy": "strict_auto_v2",
                "body_quality": "analysis_v1",
                "report_language": "en",
                "review_structure": "academic_v2",
                "writing_bindings": bindings,
                "scope_notice_contract": "scope-notice-v1",
                "fact_cards": case.cards,
                "evidence_snapshot_sha256": digest(case.cards),
                "evidence": case.ledger,
                "coverage_requirements": case.packet["requirements"],
                "structure_checks": {"status": "passed"},
                "method_checks": {"status": "not_applicable"},
                "language_checks": {"status": "passed"},
            },
            config=EvalConfig(),
        )
        assert details["coverage"]["complete"]
        assert details["unsupported_count"] > 0
        assert strict_acceptance(details)["status"] == "failed"
        assert original == case.body and digest(original) == original_digest
        assert session.reference_input(original) == original_receipt
    finally:
        await owner.close()
