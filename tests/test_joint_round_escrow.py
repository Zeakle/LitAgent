"""Fund complete joint round domains through real owners and accounting readers."""

import copy
import json
from collections import Counter

import pytest
from test_initial_batch_reservation import _initial_batch_case
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.capacity_rounds import (
    choose_refinement,
    plan_round,
    request_proof,
    round_request_identity,
)
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval import batched, scoped
from litagent.quality import build_fact_cards, digest
from litagent.safety import planned_budget
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_v2 import BODY, SCOPE, source_packet, writer_budget_plan


def _base_plan(packet, aliases, **kwargs):
    """Produce a complete v7 source-bound plan before selecting the new domain."""
    return writer_budget_plan(
        packet,
        aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        classification_review=True,
        repair_feedback=True,
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
        support_validation_revision="host-support-receipts-v2",
        evaluation_response_contract=batched.EVALUATION_RESPONSE_CONTRACT,
        **kwargs,
    )


def _joint_plan(case, **kwargs):
    """Apply the real wrapper without introducing caller-controlled ceilings."""
    base = _base_plan(case.packet, case.aliases, **kwargs)
    assert base["status"] == "admitted"
    return base, planned_budget.with_joint_round_escrow(base)


def _domain(plan):
    """Read the authenticated descriptor used by both allocation and restore."""
    return plan["joint_round_escrow"]["assessment"]


def _settle_cap(owner, stage, key):
    """Charge the full protected ceiling rather than manufacture settled savings."""
    spec = owner.plan["stages"][stage]
    amount, output = spec["request_cap"], spec["output_tokens"]
    with request_scope(stage, key):
        owner.reserve(amount)
        owner.begin_request()
        owner.record({"prompt_tokens": amount - output, "completion_tokens": output})
    assert owner.entries[key]["actual_tokens"] == amount


def _settle_initial_max(owner):
    """Reach the actual first inventory boundary with every request at its cap."""
    for stage in ("body_initial", "body", "reviewer"):
        for index in range(owner.plan["stages"][stage]["count"]):
            _settle_cap(owner, stage, f"initial:{stage}:{index}")
    for stage in ("summary", "inventory"):
        _settle_cap(owner, stage, f"initial:{stage}")


def _amounts(total, count):
    """Distribute a fixed escrow while keeping every request fully reserved."""
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def _wire(stage, amount, marker, response_format=None):
    """Construct real serialized bytes with exactly the requested reservation."""
    output = 16384
    messages = [{"role": "user", "content": marker}]
    overhead = strict_request_reservation(messages, output, response_format)
    assert amount >= overhead
    messages[0]["content"] += "x" * (amount - overhead)
    proof = request_proof(stage, messages, output, response_format)
    assert proof["reserved_tokens"] == amount
    return stage, messages, response_format, proof


def _round_wires(owner, amounts, marker):
    """Keep maximum-usage accounting wires separate from the science fixture."""
    values = [
        _wire("assessment", amount, f"{marker}:assessment:{index}")
        for index, amount in enumerate(amounts)
    ]
    values.append(
        _wire("global", owner.plan["stages"]["global"]["request_cap"], marker)
    )
    return values


def _allocate(owner, values, *, assertions=128, report_sha="a" * 64):
    """Use the real complete round planner without stubbing any resource cost."""
    return plan_round(
        owner,
        assertions=assertions,
        batches=len(values) - 1,
        requests=[value[3] for value in values],
        report_sha256=report_sha,
    )


def _send_max(owner, value, base_key):
    """Check, reserve, send and settle one exact round wire at its full amount."""
    stage, messages, native, proof = value
    key = round_request_identity(owner, stage, base_key)
    amount = proof["reserved_tokens"]
    with request_scope(stage, key):
        assert owner.check_wire(messages, 16384, native) == amount
        owner.reserve(amount)
        owner.begin_request()
        usage = {"prompt_tokens": amount - 16384, "completion_tokens": 16384}
        owner.record(usage)
        settled = owner.snapshot()
        owner.record(usage)
        assert owner.snapshot() == settled
    assert owner.entries[key]["actual_tokens"] == amount
    return key


def _check_reader(snapshot, expected_plan):
    """Check exact immutable data without normalizing or replacing the owner."""
    before, expected = copy.deepcopy(snapshot), copy.deepcopy(expected_plan)
    result = validate_accounting_snapshot(snapshot, expected_plan=expected_plan)
    assert result == before and snapshot == before and expected_plan == expected
    assert result is not snapshot and result["plan"] is not snapshot["plan"]
    return result


@pytest.mark.parametrize("spent", [(0, 0), (12345, 2)])
@pytest.mark.parametrize("maximum_batches", [None, 12])
def test_joint_root_derives_all_three_escrows_without_changing_parent(
    case, spent, maximum_batches
):
    base, plan = _joint_plan(
        case,
        spent_tokens=spent[0],
        spent_requests=spent[1],
        capacity=CapacityConfig(max_assessment_batches=maximum_batches),
    )
    original = copy.deepcopy(base)
    assert planned_budget.with_joint_round_escrow(base) == plan
    assert base == original and "joint_round_escrow" not in base
    joint, domain = plan["joint_round_escrow"], _domain(plan)
    non = {name: spec for name, spec in base["stages"].items() if name != "assessment"}
    tokens = base["spent_tokens"] + sum(
        spec["reserved_tokens"] for spec in non.values()
    )
    count = base["spent_requests"] + sum(spec["count"] for spec in non.values())
    expected_count = min(maximum_batches or 64, (64 - count) // 3)
    expected_escrow = min(
        expected_count * 100000,
        (base["planned_tokens"] * 10 // 11 - tokens) // 3,
    )
    assert joint == {
        "contract": planned_budget.JOINT_ROUND_ESCROW_CONTRACT,
        "rule_identity": planned_budget.joint_round_escrow_identity(),
        "base_plan_identity": base["identity"],
        "rounds": 3,
        "assessment": {
            "max_requests": expected_count,
            "request_cap": 100000,
            "reserved_tokens": expected_escrow,
        },
    }
    assert domain["max_requests"] >= 8
    assert plan["stages"]["assessment"]["count"] == 3 * expected_count
    assert plan["stages"]["assessment"]["reserved_tokens"] == 3 * expected_escrow
    assert 3 * expected_escrow < 3 * expected_count * 100000
    assert {name: plan["stages"][name] for name in non} == non
    assert plan["planned_tokens"] == base["planned_tokens"] <= 4500000
    assert plan["envelope"] == base["envelope"]
    assert plan["repair_limit"] == base["repair_limit"] == 2
    assert plan["identity"] != base["identity"]
    assert PlannedBudget(plan).plan == plan


def test_joint_maximum_usage_funds_three_full_rounds_and_both_repairs(case, tmp_path):
    """Exhaust every reservation, including all fifteen assessment slots per round."""
    base, plan = _joint_plan(case)
    owner = PlannedBudget(plan, path=tmp_path / "joint-ledger.json")
    domain = _domain(plan)
    amounts = _amounts(domain["reserved_tokens"], domain["max_requests"])
    assert min(amounts) > 16384 and max(amounts) <= 100000
    _settle_initial_max(owner)
    for index in range(3):
        values = _round_wires(owner, amounts, f"full-max:{index}")
        allocation = _allocate(owner, values, report_sha=str(index + 1) * 64)
        assert allocation["round"]["remaining_repairs"] == 2 - index
        assert allocation["stages"]["assessment"]["reserved_tokens"] == 3 * sum(amounts)
        assert allocation["required_tokens"] == plan["required_tokens"]
        assert allocation["with_margin_tokens"] <= base["planned_tokens"]
        owner.check_stage_requests("assessment", amounts)
        for slot, value in enumerate(values):
            key = _send_max(owner, value, f"same-slot:{slot}")
            with pytest.raises(ValueError, match="no_replay"):
                round_request_identity(owner, value[0], f"same-slot:{slot}")
            assert (
                owner.entries[key]["reserved_tokens"]
                == owner.entries[key]["actual_tokens"]
            )
        snapshot = owner.snapshot()
        assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
        _check_reader(snapshot, plan)
        assert json.loads((tmp_path / "joint-ledger.json").read_text()) == snapshot
        if index < 2:
            for stage in ("repair", "summary", "inventory"):
                _settle_cap(owner, stage, f"candidate:{index}:{stage}")
    final = owner.snapshot()
    assert final["used_tokens"] == plan["required_tokens"]
    assert final["requests_sent"] == plan["planned_requests"] <= 64
    assert final["max_tokens"] == base["planned_tokens"]
    assert len(owner._stage_entries("assessment")) == 3 * domain["max_requests"]
    assert len(owner._stage_entries("repair")) == 2
    assert len(owner._stage_entries("global")) == 3
    assert all(
        entry["actual_tokens"] == entry["reserved_tokens"]
        for entry in owner.entries.values()
    )
    with request_scope("repair", "third-repair"):
        with pytest.raises(ValueError, match="exceeded"):
            owner.reserve(plan["stages"]["repair"]["request_cap"])


def _source_plan_for_initial_case(initial):
    """Rebuild the imported helper's exact synthetic source pool for the owner."""
    payload = initial.payload
    ledger = {}
    for material in payload["cited_materials"].values():
        source = payload["original_supports"][material["original_support_ref"]]
        eid = material["evidence_id"]
        ledger[eid] = {
            "evidence_id": eid,
            "paper_id": source["paper_id"],
            "paper_title": material["paper_title"],
            "chunk_key": "abstract",
            "text": "Controlled finding candidate; not scientific gold.",
            "supporting_text": source["text"],
        }
    ledger = dict(sorted(ledger.items()))
    selection = EvidenceSelection(
        candidate_count=len(ledger),
        selected_items=ledger,
        section_evidence_ids={section: list(ledger) for section in BODY},
        method_by_section={section: "lexical_fallback" for section in BODY},
        omitted_count=0,
        estimated_tokens=1000,
        writing_scope_policy=SCOPE,
    )
    cards = build_fact_cards(ledger)
    packet, aliases = source_packet(
        cards, selection, "Explain controlled limitations.", [], "zh-CN"
    )
    assert (
        sorted(source["source_quote"] for source in packet["sources"].values())
        == initial.source_texts
    )
    base = _base_plan(packet, aliases)
    assert base["status"] == "admitted"
    return base, planned_budget.with_joint_round_escrow(base)


def test_actual_complete_67_claim_nine_batches_enter_the_joint_owner():
    """Measure actual native wires, retaining every claim and complete own sources."""
    initial = _initial_batch_case()
    base, plan = _source_plan_for_initial_case(initial)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    payload = copy.deepcopy(initial.payload)
    payload["writer_envelope"] = copy.deepcopy(plan["envelope"])
    payload = batched.batch_payload(payload, payload["assertions"])
    before = copy.deepcopy(payload)
    batches = batched.plan_batches(payload, initial.config)
    assert [len(batch["assertions"]) for batch in batches] == [8] * 8 + [3]
    assert payload == before
    values = []
    reached_sources = {}
    for batch in batches:
        messages, native = scoped.build_evaluation_request(
            batched.batch_prompt(batch), batch
        )
        values.append(
            (
                "assessment",
                messages,
                native,
                request_proof("assessment", messages, 16384, native),
            )
        )
        wire = json.loads(messages[1]["content"])
        cited = {
            reference["evidence_alias"]
            for claim in batch["assertions"]
            for reference in claim["evidence"]
        }
        expected_sources = {
            payload["cited_materials"][alias]["original_support_ref"] for alias in cited
        }
        assert set(batch["original_supports"]) == expected_sources
        assert wire["original_supports"] == batch["original_supports"]
        for ref, source in batch["original_supports"].items():
            assert source == payload["original_supports"][ref]
            reached_sources[ref] = source
        schema = native["json_schema"]["schema"]
        claims_by_id = {claim["claim_id"]: claim for claim in batch["assertions"]}
        checked_ids = []
        for reference in schema["properties"]["decisions"]["items"]["anyOf"]:
            branch = schema["$defs"][reference["$ref"].rsplit("/", 1)[-1]]
            properties = branch["properties"]
            for claim_id in properties["claim_id"]["enum"]:
                own_citations = {
                    item["evidence_alias"]
                    for item in claims_by_id[claim_id]["evidence"]
                }
                allowed = {
                    pid
                    for pid, passage in wire["support_passages"].items()
                    if passage[0] in own_citations
                }
                assert (
                    set(
                        properties["support_basis"]["items"]["properties"][
                            "support_id"
                        ]["enum"]
                    )
                    == allowed
                )
                assert (
                    set(
                        properties["aspect_checks"]["items"]["properties"][
                            "support_ids"
                        ]["items"]["enum"]
                    )
                    == allowed
                )
                checked_ids.append(claim_id)
        assert Counter(checked_ids) == Counter(claims_by_id.keys())
    assert reached_sources == payload["original_supports"]
    assert (
        sorted(source["text"] for source in reached_sources.values())
        == initial.source_texts
    )
    assert Counter(
        claim["claim_id"] for batch in batches for claim in batch["assertions"]
    ) == Counter(claim["claim_id"] for claim in payload["assertions"])
    amounts = [value[3]["reserved_tokens"] for value in values]
    assert len(amounts) == 9 > base["envelope"]["assessment_batches"] == 8
    assert max(amounts) <= _domain(plan)["request_cap"]
    assert sum(amounts) <= _domain(plan)["reserved_tokens"]
    values.append(_wire("global", plan["stages"]["global"]["request_cap"], "67-global"))
    allocation = _allocate(owner, values, assertions=67)
    assert allocation["round"]["batches"] == 9
    assert (
        allocation["stages"]["assessment"]["reserved_tokens"]
        == sum(amounts) + 2 * _domain(plan)["reserved_tokens"]
    )
    for index, value in enumerate(values):
        _send_max(owner, value, f"actual67:{index}")
    assert owner.plan["planned_tokens"] == base["planned_tokens"]
    assert len(owner._stage_entries("assessment")) == 9
    _check_reader(owner.snapshot(), plan)


@pytest.mark.parametrize("fault", ["count", "total", "request_cap"])
def test_unfundable_current_round_preserves_all_future_escrows(case, fault):
    """Reject beyond the complete remaining path even with every old use at max."""
    _, plan = _joint_plan(case)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    domain = _domain(plan)
    non = {name: spec for name, spec in plan["stages"].items() if name != "assessment"}
    non_count = plan["spent_requests"] + sum(spec["count"] for spec in non.values())
    non_tokens = plan["spent_tokens"] + sum(
        spec["reserved_tokens"] for spec in non.values()
    )
    if fault == "count":
        count = 64 - non_count - 2 * domain["max_requests"] + 1
        amounts = _amounts(domain["reserved_tokens"], count)
    elif fault == "total":
        total = (
            plan["planned_tokens"] * 10 // 11
            - non_tokens
            - 2 * domain["reserved_tokens"]
            + 1
        )
        amounts = _amounts(total, domain["max_requests"])
    else:
        amounts = [100001]
    values = _round_wires(owner, amounts, fault)
    before = owner.snapshot()
    with pytest.raises(ValueError):
        _allocate(owner, values, assertions=len(amounts) * 16)
    assert owner.snapshot() == before
    assert owner.allocation is None and owner.revisions == []
    assert not owner._stage_entries("assessment")


def test_joint_refinement_can_target_below_initial_capacity_without_sending_old(case):
    _, plan = _joint_plan(case)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    allocation = choose_refinement(
        owner, assertions=67, batches=9, report_sha256="b" * 64
    )
    assert allocation["round"]["mode"] == "refine"
    assert allocation["round"]["assertions"] <= 67 < 128
    assert allocation["round"]["status"] == "superseded_unverified"
    assert (
        allocation["stages"]["assessment"]["reserved_tokens"]
        == 2 * _domain(plan)["reserved_tokens"]
    )
    assert allocation["stages"]["repair"]["count"] == 2
    assert not owner._stage_entries("assessment")
    with request_scope("assessment", "unsent-original"):
        with pytest.raises(ValueError, match="not_authorized"):
            owner.check_wire([{"role": "user", "content": "original"}], 16384)
    assert owner.plan["envelope"]["model_assertions"] == 128
    _check_reader(owner.snapshot(), plan)


def test_aggregate_refinement_keeps_counts_but_cannot_fund_an_unchecked_candidate(case):
    """A count-preserving target grants neither old slots nor candidate wires."""
    _, plan = _joint_plan(case)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    domain = _domain(plan)
    non_tokens = plan["spent_tokens"] + sum(
        spec["reserved_tokens"]
        for name, spec in plan["stages"].items()
        if name != "assessment"
    )
    old_total = (
        plan["planned_tokens"] * 10 // 11
        - non_tokens
        - 2 * domain["reserved_tokens"]
        + 1
    )
    old_amounts = _amounts(old_total, 9)
    assert 9 <= domain["max_requests"]
    assert max(old_amounts) <= domain["request_cap"]
    assert sum(old_amounts) > domain["reserved_tokens"]
    old_wires = _round_wires(owner, old_amounts, "aggregate-original")
    original = owner.snapshot()
    with pytest.raises(ValueError, match="unaffordable"):
        _allocate(owner, old_wires, assertions=67)
    assert owner.snapshot() == original
    target = choose_refinement(owner, assertions=67, batches=9, report_sha256="b" * 64)
    assert target["round"]["assertions"] == 67
    assert target["round"]["batches"] == 9
    assert target["round"]["mode"] == "refine"
    assert target["round"]["status"] == "superseded_unverified"
    assert target["round"]["requests"] == []
    assert not owner._stage_entries("assessment")
    with request_scope("assessment", "old-aggregate-wire"):
        with pytest.raises(ValueError, match="not_authorized"):
            owner.check_wire(old_wires[0][1], 16384)
    for stage in ("repair", "summary", "inventory"):
        _settle_cap(owner, stage, f"aggregate-candidate:{stage}")
    # Even a target with unchanged A/B grants no candidate spending authority.
    # The actual writer progress gate belongs to the separate repair pipeline.
    with request_scope("assessment", "unchecked-candidate"):
        with pytest.raises(ValueError, match="not_authorized"):
            owner.check_wire(old_wires[0][1], 16384)
    invalid_amounts = _amounts(domain["reserved_tokens"], 9)
    invalid_amounts[0] = domain["request_cap"] + 1
    invalid_wires = _round_wires(owner, invalid_amounts, "aggregate-candidate-invalid")
    before_invalid = owner.snapshot()
    with pytest.raises(ValueError):
        _allocate(owner, invalid_wires, assertions=67, report_sha="c" * 64)
    assert owner.snapshot() == before_invalid
    assert owner.allocation["round"]["mode"] == "refine"
    assert not owner._stage_entries("assessment")
    valid_wires = _round_wires(
        owner, _amounts(domain["reserved_tokens"], 9), "aggregate-candidate-valid"
    )
    candidate = _allocate(owner, valid_wires, assertions=67, report_sha="c" * 64)
    assert candidate["round"]["mode"] == "assess"
    assert candidate["round"]["remaining_repairs"] == 1
    assert candidate["round"]["batches"] == 9
    assert candidate["stages"]["assessment"]["reserved_tokens"] == (
        2 * domain["reserved_tokens"]
    )
    _send_max(owner, valid_wires[0], "preflighted-candidate")
    _check_reader(owner.snapshot(), plan)


@pytest.mark.parametrize("sent", [False, True])
def test_unknown_or_crashed_round_keeps_hold_and_forbids_replay(case, sent):
    _, plan = _joint_plan(case)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    values = _round_wires(
        owner, _amounts(_domain(plan)["reserved_tokens"], 9), "unknown"
    )
    _allocate(owner, values)
    stage, messages, native, proof = values[0]
    key = round_request_identity(owner, stage, "unknown")
    with request_scope(stage, key):
        owner.check_wire(messages, 16384, native)
        owner.reserve(proof["reserved_tokens"])
        if sent:
            owner.begin_request()
            owner.charge_unknown_usage(proof["reserved_tokens"])
    snapshot = owner.snapshot()
    restored = CostBudget.from_snapshot(snapshot)
    held = restored.snapshot()["entries"][key]
    assert held["status"] == "interrupted_no_replay"
    assert held["reserved_tokens"] == proof["reserved_tokens"]
    assert held["actual_tokens"] is None
    assert restored.snapshot()["usage_uncertain"] is True
    assert restored.stop_reason == "strict_usage_unknown"
    with pytest.raises(ValueError, match="no_replay"):
        round_request_identity(restored, stage, "unknown")
    with pytest.raises(ValueError, match="unsafe"):
        _allocate(restored, values, report_sha="c" * 64)
    _check_reader(restored.snapshot(), plan)


def test_pure_reader_uses_pinned_data_without_current_rule_or_owner(case, monkeypatch):
    _, plan = _joint_plan(case)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    values = _round_wires(
        owner, _amounts(_domain(plan)["reserved_tokens"], 9), "reader"
    )
    _allocate(owner, values)
    _send_max(owner, values[0], "reader-first")
    snapshot = owner.snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Historical accounting cannot execute current owners or rule identity"
        )

    monkeypatch.setattr(planned_budget, "joint_round_escrow_identity", forbidden)
    monkeypatch.setattr(PlannedBudget, "__init__", forbidden)
    monkeypatch.setattr(CostBudget, "from_snapshot", forbidden)
    _check_reader(snapshot, plan)


@pytest.mark.parametrize(
    "fault",
    [
        "contract",
        "rule",
        "parent",
        "rounds",
        "count",
        "cap",
        "total",
        "extra",
        "strip",
        "allowance",
    ],
)
def test_live_joint_plan_rejects_forgery_even_after_self_rehash(case, fault):
    _, plan = _joint_plan(case)
    damaged = copy.deepcopy(plan)
    descriptor = damaged["joint_round_escrow"]
    if fault == "contract":
        descriptor["contract"] = "unknown-joint-rule"
    elif fault == "rule":
        descriptor["rule_identity"] = "f" * 64
    elif fault == "parent":
        descriptor["base_plan_identity"] = "f" * 64
    elif fault == "rounds":
        descriptor["rounds"] = 2
    elif fault == "count":
        descriptor["assessment"]["max_requests"] += 1
    elif fault == "cap":
        descriptor["assessment"]["request_cap"] += 1
    elif fault == "total":
        descriptor["assessment"]["reserved_tokens"] += 1
    elif fault == "extra":
        descriptor["unexpected"] = True
    elif fault == "strip":
        damaged.pop("joint_round_escrow")
    else:
        damaged["planned_tokens"] += 10000
    damaged["identity"] = digest(
        {key: value for key, value in damaged.items() if key != "identity"}
    )
    with pytest.raises(ValueError):
        PlannedBudget(damaged)
    assert "joint_round_escrow" in plan


@pytest.mark.parametrize("fault", ["count", "total", "parent", "stage_total", "usage"])
def test_pure_reader_rejects_rehashed_joint_arithmetic_and_usage(case, fault):
    _, plan = _joint_plan(case)
    owner = PlannedBudget(plan)
    _settle_initial_max(owner)
    snapshot = owner.snapshot()
    damaged = copy.deepcopy(snapshot)
    if fault == "usage":
        damaged["used_tokens"] -= 1
    elif fault == "stage_total":
        damaged["plan"]["stages"]["assessment"]["reserved_tokens"] += 1
    elif fault == "parent":
        damaged["plan"]["joint_round_escrow"]["base_plan_identity"] = "f" * 64
    else:
        field = "max_requests" if fault == "count" else "reserved_tokens"
        damaged["plan"]["joint_round_escrow"]["assessment"][field] += 1
    damaged["plan"]["identity"] = digest(
        {key: value for key, value in damaged["plan"].items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="planned_accounting"):
        validate_accounting_snapshot(damaged, expected_plan=damaged["plan"])
    assert owner.snapshot() == snapshot


def test_none_and_rejected_base_plans_are_not_upgraded(case):
    base = _base_plan(case.packet, case.aliases)
    frozen = copy.deepcopy(base)
    owner = PlannedBudget(base)
    assert owner.plan == frozen and "joint_round_escrow" not in owner.plan
    assert (
        writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
        )["contract"]
        == "planned-budget-v7"
    )
    _check_reader(owner.snapshot(), frozen)
    rejected = _base_plan(case.packet, case.aliases, spent_tokens=4500001)
    assert rejected["status"] == "preflight_rejected"
    assert planned_budget.with_joint_round_escrow(rejected) == rejected
    assert "joint_round_escrow" not in rejected
    with pytest.raises(ValueError, match="preflight_rejected"):
        PlannedBudget(planned_budget.with_joint_round_escrow(rejected))
