"""Keep fresh round evaluations distinct while denying every same-round replay."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
import test_semantic_conflict_pipeline as pipeline
from test_adaptive_capacity import settle, settle_initial
from test_evaluation_stability_pipeline import read_inventory_catalog_at_provider
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_round_capacity import make_owner, send, wires
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_rounds import plan_round, round_request_identity
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import PlannedBudget, _digest, request_scope


def allocate(owner):
    """Authorize the same complete assessment/global wires for a new round."""
    current = wires(1)
    value = plan_round(
        owner,
        assertions=8,
        batches=1,
        requests=[proof for _, _, proof in current],
        report_sha256="a" * 64,
    )
    return value, current


def advance_inventory(owner, index):
    """Settle the genuine boundaries required before another full allocation."""
    for stage in ("repair", "summary", "inventory"):
        settle(owner, stage, f"round:{index}:{stage}", 1000)


@pytest.mark.parametrize("stage", ["assessment", "global"])
def test_round_key_accepts_settled_legacy_content_only_in_new_complete_round(
    case, stage
):
    """Three valid rounds may independently assess identical content."""
    owner = make_owner(case)
    settle_initial(owner)
    keys, allocations = [], []
    for index in range(3):
        allocation, current = allocate(owner)
        allocations.append(allocation["identity"])
        for actual_stage, messages, proof in current:
            base = "legacy:" + actual_stage
            key = (
                base
                if index == 0
                else round_request_identity(owner, actual_stage, base)
            )
            if actual_stage == stage:
                keys.append(key)
            send(owner, actual_stage, messages, proof["reserved_tokens"], key)
            with pytest.raises(ValueError, match="planned_request_no_replay"):
                round_request_identity(owner, actual_stage, base)
        if index < 2:
            advance_inventory(owner, index)
    assert len(set(keys)) == len(set(allocations)) == 3
    assert owner.entries["legacy:" + stage]["round_number"] == 1
    assert [owner.entries[key]["round_number"] for key in keys] == [1, 2, 3]
    assert CostBudget.from_snapshot(owner.snapshot()).snapshot() == owner.snapshot()


@pytest.mark.parametrize(
    "status,sent",
    [
        ("settled", True),
        ("reserved_not_sent", False),
        ("started_no_replay", True),
        ("interrupted_no_replay", True),
    ],
)
@pytest.mark.parametrize("round_number", [1, None, True, 2])
def test_legacy_same_round_or_unproved_receipt_never_gets_fresh_key(
    case, status, sent, round_number
):
    """A new key encoding cannot evade old paid or uncertain identities."""
    owner = make_owner(case)
    settle_initial(owner)
    allocate(owner)
    owner.entries["legacy"] = {
        "stage": "assessment",
        "status": status,
        "sent": sent,
        "round_number": round_number,
    }
    with pytest.raises(ValueError, match="planned_request_no_replay"):
        round_request_identity(owner, "assessment", "legacy")


@pytest.mark.parametrize(
    "fault", ["missing", "identity", "revision", "event", "refine", "round_bool"]
)
def test_unverified_or_refinement_allocation_never_grants_request_identity(case, fault):
    """Only the current host allocation can provide a dispatch namespace."""
    owner = make_owner(case)
    settle_initial(owner)
    allocate(owner)
    if fault == "missing":
        owner.allocation = None
    elif fault == "identity":
        owner.allocation["identity"] = "f" * 64
    elif fault == "revision":
        owner.revisions = []
    elif fault == "event":
        owner.revisions[-1]["identity"] = "f" * 64
    else:
        if fault == "refine":
            owner.allocation["round"]["mode"] = "refine"
            owner.allocation["round"]["status"] = "superseded_unverified"
        else:
            owner.allocation["round"]["number"] = True
        owner.allocation["identity"] = _digest(
            {k: v for k, v in owner.allocation.items() if k != "identity"}
        )
        event = owner.revisions[-1]
        event["allocation"] = copy.deepcopy(owner.allocation)
        event["identity"] = _digest({k: v for k, v in event.items() if k != "identity"})
    with pytest.raises(ValueError, match="capacity_round_request_not_authorized"):
        round_request_identity(owner, "assessment", "base")


@pytest.mark.parametrize(
    "contract",
    [None, *[f"planned-budget-v{i}" for i in range(2, 7)]],
)
@pytest.mark.parametrize("stage", ["inventory", "repair", "assessment", "global"])
def test_unselected_and_older_stage_keys_remain_exact(contract, stage):
    """Legacy callers keep the original identity without requiring allocation."""
    owner = None if contract is None else SimpleNamespace(plan={"contract": contract})
    assert round_request_identity(owner, stage, "original-key") == "original-key"


@pytest.mark.parametrize("stage", ["body", "summary", "inventory", "repair"])
def test_other_v7_stages_do_not_require_or_acquire_round_scope(case, stage):
    """Only assessment/global slots receive the new identity boundary."""
    assert round_request_identity(make_owner(case), stage, "original") == "original"


def test_selected_round_identity_rejects_duck_owner(case):
    """Copied allocation-shaped attributes do not establish budget ownership."""
    owner = make_owner(case)
    settle_initial(owner)
    allocate(owner)
    fake = SimpleNamespace(**owner.__dict__)
    with pytest.raises(ValueError, match="capacity_round_request_not_authorized"):
        round_request_identity(fake, "assessment", "legacy")


@pytest.mark.parametrize("base", [None, 1, [], {}, ""])
def test_selected_round_identity_does_not_repair_invalid_base_scope(case, base):
    """Hashing cannot turn a previously invalid request owner into a valid one."""
    owner = make_owner(case)
    settle_initial(owner)
    allocate(owner)
    with pytest.raises(ValueError, match="planned_request_scope_missing"):
        round_request_identity(owner, "assessment", base)


@pytest.mark.asyncio
async def test_third_round_same_wire_reaches_sdk_transport_once_and_replay_never_sends(
    case, tmp_path
):
    """Use real SDK reservation and send markers with only HTTP replaced."""
    calls = []

    def transport(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-round",
                "object": "chat.completion",
                "created": 0,
                "model": "pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            },
        )

    owner = make_owner(case, tmp_path / "ledger.json")
    settle_initial(owner)
    for index in range(2):
        _, current = allocate(owner)
        for stage, messages, proof in current:
            key = f"round:{index}:{stage}" if index == 0 else "legacy:" + stage
            send(owner, stage, messages, proof["reserved_tokens"], key)
        advance_inventory(owner, index)
    _, current = allocate(owner)
    stage, messages, _ = current[0]
    base = "legacy:assessment"
    key = round_request_identity(owner, stage, base)
    client, _, _ = await owner_for(transport)
    client._cost_budget = owner
    try:
        with request_scope(stage, key):
            await client.chat(messages, max_tokens=16384, transport_max_attempts=1)
        assert len(calls) == 1
        assert owner.entries[key]["round_number"] == 3
        assert owner.entries[key]["status"] == "settled"
        with pytest.raises(ValueError, match="planned_request_no_replay"):
            round_request_identity(owner, stage, base)
        assert len(calls) == 1
        with pytest.raises(ValueError, match="capacity_round_request_not_authorized"):
            with request_scope(
                "assessment", round_request_identity(owner, stage, "other")
            ):
                owner.check_wire(messages, 16384)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", [False, True])
async def test_restored_uncertain_round_never_reaches_transport(case, tmp_path, sent):
    """Crash recovery retains both reserved and sent unknown requests."""
    calls = []
    owner = make_owner(case, tmp_path / "unknown.json")
    settle_initial(owner)
    _, current = allocate(owner)
    stage, messages, proof = current[0]
    base = "legacy:assessment"
    key = round_request_identity(owner, stage, base)
    with request_scope(stage, key):
        owner.check_wire(messages, 16384)
        owner.reserve(proof["reserved_tokens"])
        if sent:
            owner.begin_request()
    restored = PlannedBudget(owner.plan, path=tmp_path / "unknown.json")
    client, _, _ = await owner_for(lambda request: calls.append(request))
    client._cost_budget = restored
    try:
        with pytest.raises(ValueError, match="planned_request_no_replay"):
            with request_scope(stage, round_request_identity(restored, stage, base)):
                await client.chat(messages, max_tokens=16384)
        assert calls == []
        assert restored.stop_reason == "strict_usage_unknown"
        assert restored.snapshot()["requests_sent"] == owner.snapshot()["requests_sent"]
    finally:
        await client.close()


class RepeatedRoundGateway(FlatGateway):
    """Change only a late extracted claim while keeping earlier batches identical."""

    def __init__(self, case):
        super().__init__(case)
        self.inventories = 0
        original = self.evaluator.chat

        async def chat(messages, **options):
            response = await original(messages, **options)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                self.inventories += 1
                fragment = next(reversed(value["fragments"].values()))
                fragment["assertions"][0][
                    "claim_text"
                ] += f" Extraction-only variant {self.inventories}."
            if "cited_materials" in payload:
                claims = {c["claim_id"]: c for c in payload["assertions"]}
                for decision in value["decisions"]:
                    if (
                        "Extraction-only variant"
                        in claims[decision["claim_id"]]["claim_text"]
                    ):
                        decision["atomic"] = False
                        decision["source_faithful"] = False
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        read_inventory_catalog_at_provider(self)


@pytest.mark.asyncio
async def test_real_scoped_evaluation_reassesses_unchanged_batches_in_three_rounds(
    case, tmp_path, monkeypatch
):
    """Exercise actual call/global preflight scopes through the frozen pipeline."""
    monkeypatch.setattr(pipeline, "ConflictGateway", RepeatedRoundGateway)
    report, gateway, ledger = await pipeline.run_conflict_case(
        case, tmp_path, monkeypatch
    )
    assert gateway.inventories == 3
    assert len(ledger["capacity_revisions"]) == 3
    assessments = [
        entry for entry in ledger["entries"].values() if entry["stage"] == "assessment"
    ]
    assert {entry["round_number"] for entry in assessments} == {1, 2, 3}
    assert all(entry["status"] == "settled" for entry in assessments)
    first_proofs = [
        event["allocation"]["round"]["requests"][0]
        for event in ledger["capacity_revisions"]
    ]
    assert first_proofs[0] == first_proofs[1] == first_proofs[2]
    assert (
        len(
            [
                entry
                for entry in ledger["entries"].values()
                if entry["stage"] == "global"
            ]
        )
        == 3
    )
    assert (
        report["evaluation"]["faithfulness"]["details"]["stages"]["global_check"]
        == "completed"
    )
    assert report["delivery"]["publishable"] is False
    assert ledger["requests_sent"] == len(gateway.sent)


def test_changed_wire_still_fails_exact_current_slot_guard(case):
    """A round key changes ownership only; it cannot authorize other content."""
    owner = make_owner(case)
    settle_initial(owner)
    allocate(owner)
    key = round_request_identity(owner, "assessment", "legacy")
    with request_scope("assessment", key):
        with pytest.raises(ValueError, match="capacity_round_wire_identity_conflict"):
            owner.check_wire([{"role": "user", "content": "forged"}], 16384)
    assert not any(entry["stage"] == "assessment" for entry in owner.entries.values())
