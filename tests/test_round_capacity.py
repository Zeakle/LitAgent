"""Exercise exact round escrows, explicit edits and unchanged old contracts."""

import copy

import pytest
from test_adaptive_capacity import settle, settle_initial
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_repair import capacity_targets, normalize_edit_map
from litagent.capacity_rounds import choose_refinement, plan_round, request_proof
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_v2 import WriterError, writer_budget_plan


def make_owner(case, path=None):
    """Build the actual writer plan with the new opt-in identity."""
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    assert plan["status"] == "admitted"
    return PlannedBudget(plan, path=path)


def wires(count, *, text="source"):
    """Keep deterministic full requests so amount and identity can be checked."""
    values = []
    for i in range(count + 1):
        stage = "global" if i == count else "assessment"
        messages = [{"role": "user", "content": text + str(i)}]
        values.append((stage, messages, request_proof(stage, messages, 16384, None)))
    return values


def send(owner, stage, messages, amount, key):
    """Use the exact check/reserve/send/settle path, including maximum usage."""
    with request_scope(stage, key):
        owner.check_wire(messages, 16384)
        owner.reserve(amount)
        owner.begin_request()
        owner.record({"prompt_tokens": amount - 16384, "completion_tokens": 16384})


def test_round_slots_are_exact_and_both_full_repairs_survive(case, tmp_path):
    owner = make_owner(case, tmp_path / "ledger.json")
    settle_initial(owner)
    for round_index in range(3):
        current = wires(9, text=f"round:{round_index}")
        allocation = plan_round(
            owner,
            assertions=144,
            batches=9,
            requests=[v[2] for v in current],
            report_sha256=str(round_index) * 64,
        )
        assert allocation["round"]["remaining_repairs"] == 2 - round_index
        assert allocation["with_margin_tokens"] <= owner.plan["planned_tokens"]
        with request_scope("assessment", f"wrong:{round_index}"):
            with pytest.raises(ValueError, match="wire_identity"):
                owner.check_wire([{"role": "user", "content": "changed"}], 16384)
        for i, (stage, messages, proof) in enumerate(current):
            send(
                owner,
                stage,
                messages,
                proof["reserved_tokens"],
                f"round:{round_index}:{i}",
            )
        restored = CostBudget.from_snapshot(owner.snapshot())
        assert restored.snapshot() == owner.snapshot()
        if round_index < 2:
            for stage in ("repair", "summary", "inventory"):
                settle(
                    owner,
                    stage,
                    f"candidate:{round_index}:{stage}",
                    owner.plan["stages"][stage]["request_cap"],
                )
    assert owner.snapshot()["used_tokens"] <= owner.plan["planned_tokens"]
    assert len(owner._stage_entries("repair")) == 2
    assert len(owner._stage_entries("global")) == 3


def test_refinement_skips_only_unsent_old_assessment_and_locks_goal(case):
    owner = make_owner(case)
    settle_initial(owner)
    allocation = choose_refinement(
        owner, assertions=184, batches=12, report_sha256="a" * 64
    )
    assert allocation["round"]["status"] == "superseded_unverified"
    assert allocation["round"]["assertions"] < 184
    assert allocation["round"]["batches"] < 12
    assert allocation["stages"]["inventory"]["count"] == 3
    assert allocation["stages"]["repair"]["count"] == 2
    with request_scope("assessment", "cannot_assess_old_draft"):
        with pytest.raises(ValueError, match="not_authorized"):
            owner.check_wire([{"role": "user", "content": "old"}], 16384)
    assert owner.plan["envelope"]["model_assertions"] == 128
    assert CostBudget.from_snapshot(owner.snapshot()).snapshot() == owner.snapshot()


@pytest.mark.parametrize(
    "fault", ["unknown", "reused_inventory", "tamper", "unchecked_wire"]
)
def test_round_budget_rejects_unsafe_changes(case, fault):
    owner = make_owner(case)
    settle_initial(owner)
    current = wires(8)
    arguments = dict(
        assertions=128,
        batches=8,
        requests=[v[2] for v in current],
        report_sha256="a" * 64,
    )
    if fault == "unknown":
        with request_scope("repair", "unknown"):
            owner.reserve(20000)
            owner.begin_request()
            owner.charge_unknown_usage(20000)
        with pytest.raises(ValueError, match="unsafe"):
            plan_round(owner, **arguments)
        assert owner.entries["unknown"]["status"] == "interrupted_no_replay"
        return
    plan_round(owner, **arguments)
    if fault == "reused_inventory":
        with pytest.raises(ValueError, match="identity_conflict"):
            plan_round(owner, **arguments)
    elif fault == "tamper":
        snapshot = owner.snapshot()
        snapshot["capacity_allocation"] = copy.deepcopy(snapshot["capacity_allocation"])
        snapshot["capacity_allocation"]["round"]["requests"][0]["reserved_tokens"] -= 1
        with pytest.raises(ValueError, match="allocation_invalid"):
            CostBudget.from_snapshot(snapshot)
    else:
        with request_scope("assessment", "unchecked"):
            with pytest.raises(ValueError, match="wire_identity"):
                owner.reserve(current[0][2]["reserved_tokens"])


def test_mapping_merges_without_redundant_actions_and_preserves_others(case):
    body = copy.deepcopy(case.body)
    body["methods"][0]["sentences"].append(
        copy.deepcopy(body["methods"][0]["sentences"][0])
    )
    targets = capacity_targets(body, {})
    tid = next(k for k, t in targets.items() if t["path"][0] == "methods")
    result = normalize_edit_map(
        {
            "outcome": "patched",
            "explanation": "Merge repeated finding.",
            "patches": [
                {
                    "target_id": tid,
                    "sentences": [copy.deepcopy(body["methods"][0]["sentences"][0])],
                }
            ],
            "edit_records": [
                {
                    "unit_id": f"writer:methods:0:{i}",
                    "replacement_indexes": [0],
                    "reason": "Shared merged finding.",
                }
                for i in range(2)
            ],
        },
        targets,
    )
    assert len(result["edit_records"]) == sum(
        len(t["original"]) for t in targets.values()
    )
    assert result["removed_targets"] == []
    for patch in result["patches"]:
        if patch["target_id"] != tid:
            assert patch["sentences"] == targets[patch["target_id"]]["original"]


@pytest.mark.parametrize("fault", ["omit", "outside", "bool", "orphan", "duplicate"])
def test_mapping_cannot_silently_drop_or_invent_units(case, fault):
    targets = capacity_targets(case.body, {})
    tid = next(iter(targets))
    t = targets[tid]
    uid = "writer:" + ":".join(map(str, t["path"]))
    result = {
        "outcome": "patched",
        "explanation": "Edit.",
        "patches": [{"target_id": tid, "sentences": copy.deepcopy(t["original"])}],
        "edit_records": [
            {"unit_id": uid, "replacement_indexes": [0], "reason": "Preserve."}
        ],
    }
    if fault == "omit":
        result["edit_records"] = []
    if fault == "outside":
        result["edit_records"][0]["replacement_indexes"] = [5]
    if fault == "bool":
        result["edit_records"][0]["replacement_indexes"] = [False]
    if fault == "orphan":
        result["patches"][0]["sentences"].append(copy.deepcopy(t["original"][0]))
    if fault == "duplicate":
        result["edit_records"] *= 2
    with pytest.raises(WriterError, match="writer_capacity"):
        normalize_edit_map(result, targets)


def test_mapping_explicit_deletion_has_no_implicit_patch(case):
    targets = capacity_targets(case.body, {})
    tid = next(iter(targets))
    t = targets[tid]
    result = normalize_edit_map(
        {
            "outcome": "patched",
            "explanation": "Omit redundant background.",
            "patches": [],
            "edit_records": [
                {
                    "unit_id": "writer:" + ":".join(map(str, t["path"])),
                    "replacement_indexes": [],
                    "reason": "Redundant background.",
                }
            ],
        },
        targets,
    )
    assert result["removed_targets"] == [tid]
    assert result["patches"][0]["sentences"] == []


def test_allocation_storage_failure_never_authorizes_requests(case, monkeypatch):
    owner = make_owner(case)
    settle_initial(owner)

    def fail():
        raise OSError("offline disk failure")

    monkeypatch.setattr(owner, "_persist", fail)
    with pytest.raises(OSError):
        choose_refinement(owner, assertions=184, batches=12, report_sha256="a" * 64)
    assert owner.revisions == [] and owner.allocation is None
    assert owner.stop_reason == "capacity_allocation_persist_failed"


def test_round_receipt_chain_rejects_parent_and_wire_drift(case):
    from litagent.safety.planned_budget import _digest

    owner = make_owner(case)
    settle_initial(owner)
    current = wires(9)
    plan_round(
        owner,
        assertions=144,
        batches=9,
        requests=[v[2] for v in current],
        report_sha256="a" * 64,
    )
    stage, messages, proof = current[0]
    send(owner, stage, messages, proof["reserved_tokens"], "sent")
    snapshot = owner.snapshot()
    snapshot["entries"]["sent"]["round_wire_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="receipt_invalid"):
        CostBudget.from_snapshot(snapshot)
    snapshot = owner.snapshot()
    event = snapshot["capacity_revisions"][0]
    event["parent"] = "f" * 64
    event["identity"] = _digest({k: v for k, v in event.items() if k != "identity"})
    with pytest.raises(ValueError, match="chain_invalid"):
        CostBudget.from_snapshot(snapshot)


def test_current_exact_amounts_do_not_replace_future_full_bounds(case):
    owner = make_owner(case)
    settle_initial(owner)
    current = wires(12)
    allocation = plan_round(
        owner,
        assertions=184,
        batches=12,
        requests=[v[2] for v in current],
        report_sha256="a" * 64,
    )
    spec = allocation["stages"]["assessment"]
    assert (
        spec["reserved_tokens"]
        == sum(v[2]["reserved_tokens"] for v in current[:-1])
        + 24 * owner.plan["stages"]["assessment"]["request_cap"]
    )
    assert allocation["stages"]["inventory"]["count"] == 3


def test_operational_limit_produces_funded_goal_before_any_edit(case):
    from litagent.capacity import CapacityConfig
    from litagent.capacity_evaluation import refine_rejected_round

    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        capacity=CapacityConfig(max_assertions=140),
    )
    owner = PlannedBudget(plan)
    settle_initial(owner)
    progress = {
        "inventory_assertions": [
            {"source_fragment_id": str(i), "section": "methods"} for i in range(144)
        ]
    }
    with pytest.raises(ValueError, match="evaluation_capacity_refinement_required"):
        refine_rejected_round(
            owner, progress, "evaluation_assertion_capacity_exceeded", "a" * 64
        )
    assert len(progress["inventory_assertions"]) == 144
    assert progress["capacity"]["target_assertions"] == 128
    assert progress["capacity"]["round"]["status"] == "superseded_unverified"
    assert owner._stage_entries("repair") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("declined", [False, True])
async def test_mapping_split_and_decline_use_real_schema_sdk(case, tmp_path, declined):
    from test_flat_writer_pipeline import FlatGateway
    from test_native_responses import owner_for
    from test_native_wire_session import make_session

    class SplitGateway(FlatGateway):
        def answer(self, payload, native_schema):
            tid, target = next(iter(payload["targets"].items()))
            section, pi, si = target["path"]
            unit = payload["current_body"][section][pi]["sentences"][si]
            return {
                "outcome": "declined" if declined else "patched",
                "explanation": "Keep all premises in two explicit units.",
                "patches": (
                    []
                    if declined
                    else [
                        {"target_id": tid, **unit},
                        {
                            "target_id": tid,
                            "text": "This transition organizes the comparison.",
                            "fact_card_refs": [],
                        },
                    ]
                ),
                "edit_records": [
                    {
                        "unit_id": f"writer:{section}:{pi}:{si}",
                        "replacement_indexes": [] if declined else [0, 1],
                        "reason": "Split the transition from the supported finding.",
                    }
                ],
            }

    gateway = SplitGateway(case)
    owner, budget, _ = await owner_for(gateway)
    budget.activate_plan(
        writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
        )
    )
    session = make_session(owner, case, tmp_path)
    targets = capacity_targets(case.body, {}, maximum=66)
    try:
        if declined:
            with pytest.raises(WriterError, match="writer_decline_with_patches"):
                await session.patch(
                    case.body, targets, cause="capacity", claim_count=184
                )
        else:
            body = await session.patch(
                case.body, targets, cause="capacity", claim_count=184
            )
            assert len(body["introduction"][0]["sentences"]) == 2
            assert body["methods"] == case.body["methods"]
            assert (
                session.state["attempts"][0]["edit_records"][0]["action"] == "rewritten"
            )
        assert len(gateway.sent) == len(session.state["attempts"]) == 1
    finally:
        await owner.close()


def test_v7_evaluation_wire_omits_only_host_ledger(case):
    import json

    from litagent.eval.scoped import build_evaluation_request

    plan = make_owner(case).plan
    payload = {
        "writer_envelope": plan["envelope"],
        "scientific_source": "Full unchanged source, including conditions and values.",
        "assertions": [{"id": "one", "text": "A scientific candidate."}],
    }
    before = copy.deepcopy(payload)
    messages, _ = build_evaluation_request("Evaluate.", payload)
    actual = json.loads(messages[1]["content"])
    assert actual["scientific_source"] == payload["scientific_source"]
    assert actual["assertions"] == payload["assertions"] and payload == before
    assert set(actual["writer_envelope"]) == {
        "claim_json_bytes",
        "claim_max_json_bytes",
        "diagnostic_json_bytes",
    }
    old = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v6"
    )
    payload["writer_envelope"] = old["envelope"]
    messages, _ = build_evaluation_request("Evaluate.", payload)
    assert json.loads(messages[1]["content"])["writer_envelope"] == old["envelope"]
