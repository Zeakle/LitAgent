"""Exercise source-owned target derivation through actual frozen consumers."""

import copy
import json

import pytest
import test_structured_repair_pipeline as structured
from test_capacity_repair import run_capacity_case
from test_source_map_pipeline import SourceMapGateway, actual_start, products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writing_flat import _canonical_task_payload, parse
from litagent.writing_v2 import (
    WriterError,
    check_source_map_response,
    source_map_response_fields,
    writer_budget_plan,
)
from scripts import run_frozen_writing as campaign

CONTRACT = "capacity-source-map-v4"
FLAGS = {
    **structured.FLAGS,
    "support_validation_revision": "host-support-receipts-v2",
    "repair_mapping_contract": CONTRACT,
    "repair_guidance_wire_revision": "host-repair-task-view-v1",
}


@pytest.mark.parametrize(
    "change",
    [
        {"budget_policy": "planned-budget-v6"},
        {"support_validation_revision": None},
        {"support_validation_revision": "host-support-receipts-v1"},
        {"repair_guidance_contract": None},
        {"repair_guidance_wire_revision": None},
        {"repair_guidance_wire_revision": "host-unit-ref-guidance-v1"},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"classification_review": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
    ],
)
def test_source_owner_requires_every_current_selected_parent(case, change):
    """A partial selector cannot enable target derivation in a legacy scope."""
    settings = {"budget_policy": "planned-budget-v7", **FLAGS, **change}
    with pytest.raises(ValueError):
        writer_budget_plan(case.packet, case.aliases, methodology={}, **settings)


def test_source_owner_plan_and_fee_data_keep_exact_limits_and_old_default_pin(case):
    """New executable ownership does not increase any operational allowance."""
    from litagent.writing_flat import source_map_wire_identity

    assert source_map_wire_identity() == (
        "03a9471fc5e12065a4624b364dd1bedd5f8edface3df25cf8edca9c731a60929"
    )
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **FLAGS,
    )
    assert plan["status"] == "admitted" and plan["contract"] == "planned-budget-v7"
    assert plan["maximum_tokens"] == 4500000 and plan["maximum_requests"] == 64
    assert (
        plan["scope_bounds"]["shared_repairs"] == plan["stages"]["repair"]["count"] == 2
    )
    envelope = plan["envelope"]
    assert envelope["source_map_identity"] == source_map_wire_identity(
        contract=CONTRACT
    )
    assert validate_writer_envelope(envelope) == envelope
    assert accounting_envelope(envelope).max_body_units == 66
    historical = copy.deepcopy(envelope)
    historical["source_map_identity"] = "f" * 64
    historical["identity"] = digest(
        {key: value for key, value in historical.items() if key != "identity"}
    )
    assert accounting_envelope(historical).max_body_units == 66
    with pytest.raises(ValueError, match="writer_envelope_source_map_invalid"):
        validate_writer_envelope(historical)


def test_source_owner_cli_selection_is_explicit_and_preserves_old_defaults():
    """Frozen internal options require the same complete V4 selection."""
    assert campaign.repair_mapping_options({}) == {}
    manifest = {"writing_profile": campaign.ROUND_PROFILE, **FLAGS}
    assert campaign.repair_mapping_options(manifest) == {
        "repair_mapping_contract": CONTRACT
    }
    for field in (
        "support_validation_revision",
        "repair_guidance_contract",
        "repair_guidance_wire_revision",
    ):
        damaged = copy.deepcopy(manifest)
        damaged.pop(field)
        with pytest.raises(ValueError):
            campaign.repair_mapping_options(damaged)


def test_source_owner_actual_prepare_validate_and_manifest_strip_rejection(
    case, tmp_path, monkeypatch
):
    """Reuse portable inputs while retaining actual frozen plan and CLI consumers."""
    import test_structured_repair_identity as identity

    monkeypatch.setattr(
        identity,
        "SELECTION",
        {
            key: value
            for key, value in FLAGS.items()
            if key != "repair_guidance_contract"
        },
    )
    identity.test_real_prepare_read_bundle_validate_and_strip_resistance(
        case, tmp_path, monkeypatch
    )


class SourceOwnerGateway(SourceMapGateway):
    """Change only provider replies to the explicitly selected native shape."""

    def __init__(self, case, *, mode="valid", unknown=False, failing_candidates=0):
        super().__init__(
            case,
            mode="wrong_paper" if mode == "candidate" else mode,
            unknown=unknown,
        )
        self.owner_mode = mode
        self.failing_candidates = failing_candidates

    def answer(self, payload, native_schema):
        payload = _canonical_task_payload(payload)
        if payload.get("cause") == "capacity" and self.owner_mode == "candidate":
            self.mode = (
                "wrong_paper"
                if len(self.capacity_responses) < self.failing_candidates
                else "valid"
            )
        value = super().answer(payload, native_schema)
        if payload.get("cause") != "capacity":
            return value
        if (
            self.owner_mode == "candidate"
            and self.mode == "wrong_paper"
            and len(self.capacity_responses) > 1
        ):
            for patch in value["patches"]:
                if "Evo-ViT" in patch["text"]:
                    patch["text"] = "Evo-ViT uses the comparison function."
        for patch in value["patches"]:
            patch.pop("target_id")
        if self.owner_mode == "cross_owner":
            first, other = next(
                (a, b)
                for a in value["patches"]
                for b in value["patches"]
                if a["source_unit_ids"] != b["source_unit_ids"]
            )
            first["source_unit_ids"].append(other["source_unit_ids"][0])
        elif self.owner_mode == "unknown_source":
            value["patches"][0]["source_unit_ids"] = ["unknown-source"]
        elif self.owner_mode == "extra_target":
            value["patches"][0]["target_id"] = next(iter(payload["targets"]))
        self.capacity_responses[-1] = copy.deepcopy(value)
        return value


async def run_owner_case(
    case, tmp_path, monkeypatch, *, mode="valid", unknown=False, failing_candidates=0
):
    """Keep actual planning, Synthesis, Reviewer, evaluation and transaction guards."""
    if mode.startswith("split_merge"):
        unit = case.body["introduction"][0]["sentences"][0]
        unit.update(
            text="Relation networks learn a comparison function; on DeiT-S, EViT reduces computation.",
            fact_card_refs=[case.by_eid["b"], case.by_eid["a"]],
        )
    if mode == "missing_core":
        case.packet["requirements"][0][
            "text"
        ] = "Explain the EViT setting on DeiT-S and limitations."
    monkeypatch.setattr(structured, "FLAGS", FLAGS)
    structured.select_structured_policy(monkeypatch)
    gateway = SourceOwnerGateway(
        case, mode=mode, unknown=unknown, failing_candidates=failing_candidates
    )
    report, _ = await run_capacity_case(
        case, tmp_path, monkeypatch, gateway=gateway, allow_uncertain=unknown
    )
    return report, gateway


def assert_source_owner_receipt(call, envelope):
    """Reparse actual raw bytes and authenticate derived targets without spans."""
    raw = json.loads(call["raw"])
    assert all("target_id" not in patch for patch in raw["patches"])
    assert (
        call["repair_mapping_contract"]
        == envelope["repair_mapping_contract"]
        == CONTRACT
    )
    parsed = parse(
        call["raw"],
        call["reference_schema"],
        reference_contract=FLAGS["reference_contract"],
        transport_revision=FLAGS["transport_revision"],
    )
    check_source_map_response(envelope, call, parsed)
    directory = call["reference_schema"]["x-source-owner-directory"]
    directory_sha = digest(directory)
    assert (
        call["source_relation_bindings"]["source_owner_directory_sha256"]
        == directory_sha
    )
    assert call["source_relation_bindings"] == parsed["source_relation_bindings"]
    for relation, binding in zip(
        parsed["canonical"]["source_relations"],
        parsed["source_relation_bindings"]["relations"],
        strict=True,
    ):
        assert binding["target_id"] == {
            "derived": True,
            "value": relation["target_id"],
            "source_owner_directory_sha256": directory_sha,
        }
        assert "wire_path" not in binding["target_id"]
    for unit in parsed["bindings"]["units"]:
        assert "target_id_span" not in unit
        assert (
            unit["target_id_derivation"]["source_owner_directory_sha256"]
            == directory_sha
        )
    campaign._historical_call_binding(call, envelope)
    return parsed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "split_merge", "split_merge_interleaved"])
async def test_source_owner_reaches_full_evaluation_summary_refresh_and_commit(
    case, tmp_path, monkeypatch, mode
):
    """Source lineage alone cannot bypass any final scientific delivery gate."""
    report, gateway = await run_owner_case(case, tmp_path, monkeypatch, mode=mode)
    state, journal, ledger = products(tmp_path)
    assert report["delivery"]["publishable"] is True
    details = report["evaluation"]["faithfulness"]["details"]
    assert (
        details["coverage"]["complete"]
        and details["stages"]["global_check"] == "completed"
    )
    assert details["supported_count"] == details["factual_claim_count"] > 0
    assert details["unsupported_count"] == details["unverified_count"] == 0
    assert gateway.inventory_calls == 2 and len(gateway.capacity_responses) == 1
    assert len(sent_with(gateway, "body_units")) == 2
    assert len(state["state"]["attempts"]) == 1
    assert journal["candidates"][-1]["committed"] is True
    call = journal["calls"]["repair_0"]
    assert_source_owner_receipt(call, ledger["plan"]["envelope"])
    request = next(
        entry["request"]
        for entry in gateway.sent
        if entry["payload"].get("cause") == "capacity"
    )
    assert "x-source-owner-directory" not in json.dumps(request)
    assert "x-repair-mapping-contract" not in json.dumps(request)
    assert ledger["requests_sent"] == len(gateway.sent)
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
    assert ledger["usage_uncertain"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cross_owner", "unknown_source", "extra_target"])
async def test_invalid_source_owner_reply_is_retained_and_never_committed(
    case, tmp_path, monkeypatch, mode
):
    """Unknown, mixed and explicit target declarations fail at the actual consumer."""
    report, gateway = await run_owner_case(case, tmp_path, monkeypatch, mode=mode)
    _, journal, ledger = products(tmp_path)
    assert report["delivery"]["publishable"] is False
    assert journal["candidates"] == []
    assert 1 <= len(gateway.capacity_responses) <= 2
    call = journal["calls"]["repair_0"]
    assert (
        json.loads(call["raw"])["patches"] == gateway.capacity_responses[0]["patches"]
    )
    assert call["status"] != "completed"
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) <= 2
    assert ledger["requests_sent"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wrong_paper", "missing_core"])
async def test_valid_source_owner_map_does_not_commit_bad_science_or_missing_core(
    case, tmp_path, monkeypatch, mode
):
    """Correct provenance leaves source entailment and required conditions binding."""
    report, gateway = await run_owner_case(case, tmp_path, monkeypatch, mode=mode)
    _, journal, ledger = products(tmp_path)
    assert report["delivery"]["publishable"] is False
    assert journal["candidates"] and not any(
        row["committed"] for row in journal["candidates"]
    )
    candidate = journal["candidates"][0]
    details = candidate["evaluation"]["faithfulness"]["details"]
    assert (
        details["coverage"]["complete"]
        and details["stages"]["global_check"] == "completed"
    )
    if mode == "wrong_paper":
        assert details["unsupported_count"] > 0
    else:
        assert any(row["answered"] is False for row in details["core_coverage"])
    assert_source_owner_receipt(
        journal["calls"]["repair_0"], ledger["plan"]["envelope"]
    )
    assert len(gateway.capacity_responses) <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_source_owner_candidates_share_two_attempts_and_reassess_whole_report(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Only full candidate success can replace the retained original draft."""
    report, gateway = await run_owner_case(
        case,
        tmp_path,
        monkeypatch,
        mode="candidate",
        failing_candidates=failing_candidates,
    )
    _, journal, ledger = products(tmp_path)
    expected = min(2, failing_candidates + 1)
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    assert len(gateway.capacity_responses) == len(journal["candidates"]) == expected
    assert gateway.inventory_calls == expected + 1
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    assert journal["candidates"][-1]["committed"] is (failing_candidates < 2)
    if failing_candidates:
        assert journal["candidates"][0]["committed"] is False
        assert "Evo-ViT" not in report["survey"]
    for index in range(expected):
        assert_source_owner_receipt(
            journal["calls"][f"repair_{index}"], ledger["plan"]["envelope"]
        )
    assert ledger["requests_sent"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]


@pytest.mark.asyncio
async def test_unknown_source_owner_usage_keeps_hold_and_does_not_retry(
    case, tmp_path, monkeypatch
):
    """Unknown usage remains occupied despite the valid target-free native shape."""
    report, gateway = await run_owner_case(case, tmp_path, monkeypatch, unknown=True)
    _, journal, ledger = products(tmp_path)
    assert report["delivery"]["publishable"] is False
    assert len(gateway.capacity_responses) == 1
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert not journal["candidates"]
    assert ledger["usage_uncertain"] and ledger["stop_reason"] == "strict_usage_unknown"
    held = [
        entry
        for entry in ledger["entries"].values()
        if entry["sent"] and entry["status"] != "settled"
    ]
    assert len(held) == 1 and held[0]["actual_tokens"] is None


@pytest.mark.asyncio
async def test_source_owner_saved_receipt_and_directory_tampering_rejects_before_reuse(
    case, tmp_path, monkeypatch
):
    """Fresh raw/schema replay checks metadata without making another HTTP call."""
    _, gateway = await run_owner_case(case, tmp_path, monkeypatch)
    _, journal, ledger = products(tmp_path)
    call = journal["calls"]["repair_0"]
    envelope = ledger["plan"]["envelope"]
    parsed = assert_source_owner_receipt(call, envelope)
    sent = len(gateway.sent)
    request = next(
        entry for entry in gateway.sent if entry["payload"].get("cause") == "capacity"
    )
    start = actual_start(request)
    campaign._historical_writer_envelope(start, envelope, call=call)
    for key in (
        "repair_mapping_contract",
        "source_map_identity",
        "source_relation_bindings_sha256",
        "source_relations_sha256",
    ):
        damaged = copy.deepcopy(call)
        damaged[key] = "0" * 64
        with pytest.raises(WriterError, match="writer_source_map_receipt_invalid"):
            check_source_map_response(envelope, damaged, parsed)
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(call)
    directory = damaged["reference_schema"]["x-source-owner-directory"]
    first, second = list(directory)[:2]
    directory[first].append(directory[second][0])
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(call)
    guidance = damaged["guidance_wire_record"]
    targets = guidance["payload"]["targets"]
    first, second = list(targets)[:2]
    targets[first]["original_unit_ids"] = targets[second]["original_unit_ids"]
    guidance["payload_sha256"] = digest(guidance["payload"])
    guidance["identity"] = digest(
        {key: value for key, value in guidance.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="writer_source_owner_targets_invalid"):
        campaign._historical_writer_envelope(start, envelope, call=damaged)
    for key, value in (("derived", 1), ("value", "unrelated-target")):
        damaged = copy.deepcopy(call)
        damaged["source_relation_bindings"]["relations"][0]["target_id"][key] = value
        damaged["source_relation_bindings_sha256"] = digest(
            damaged["source_relation_bindings"]
        )
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(call)
    damaged["wire_bindings"]["units"][0]["target_id_derivation"][
        "source_relation_index"
    ] = False
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(parsed)
    damaged["source_relation_bindings"]["contract"] = "capacity-source-map-v3"
    with pytest.raises(WriterError, match="writer_source_map_binding_invalid"):
        source_map_response_fields(envelope, damaged)
    from litagent.writing_flat import source_map_wire_identity

    other = {
        **envelope,
        "repair_mapping_contract": "capacity-source-map-v3",
        "source_map_identity": source_map_wire_identity(),
    }
    with pytest.raises(WriterError, match="writer_source_map_binding_invalid"):
        source_map_response_fields(other, parsed)
    damaged = copy.deepcopy(call)
    swapped = damaged["reference_schema"]["x-source-owner-directory"]
    first, second = list(swapped)[:2]
    swapped[first], swapped[second] = swapped[second], swapped[first]
    resealed = parse(
        damaged["raw"],
        damaged["reference_schema"],
        reference_contract=FLAGS["reference_contract"],
        transport_revision=FLAGS["transport_revision"],
    )
    damaged.update(
        reference_canonical=copy.deepcopy(resealed["canonical"]),
        canonical_sha256=digest(resealed["canonical"]),
        received_canonical=copy.deepcopy(resealed["received_canonical"]),
        received_canonical_sha256=digest(resealed["received_canonical"]),
        wire_bindings=copy.deepcopy(resealed["bindings"]),
        source_relation_bindings=copy.deepcopy(resealed["source_relation_bindings"]),
        **source_map_response_fields(envelope, resealed),
    )
    # Rebuilding only parse objects cannot replace the actual authorized request.
    assert damaged["guidance_wire_record"] == call["guidance_wire_record"]
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    assert len(gateway.sent) == sent
