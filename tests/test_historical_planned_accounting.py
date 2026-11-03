"""Validate immutable historical costs without restoring a scientific runtime."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_adaptive_budget import certified_bounds
from test_adaptive_capacity import settle, settle_initial
from test_planned_budget import plan as static_plan
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.safety.budget import CostBudget
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import (
    PlannedBudget,
    adaptive_writer_plan,
    request_scope,
)
from litagent.writing_v2 import writer_budget_plan

FIXTURE = Path(__file__).parent / "fixtures/run040-cost-snapshot.json"
FIXTURE_SHA256 = "d4e7010521323f3ac80f3b1480d915b74ccb9619aaf2700f4b4e13611568aad8"


@pytest.fixture(scope="module")
def original040():
    """Use original costs and token breakdowns, never model or HTTP contents."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def checked(value, *, expected_plan=None):
    """Exercise the reader against a separately held frozen expected plan."""
    expected = copy.deepcopy(value["plan"] if expected_plan is None else expected_plan)
    before, expected_before = copy.deepcopy(value), copy.deepcopy(expected)
    result = validate_accounting_snapshot(value, expected_plan=expected)
    assert result == before and value == before and expected == expected_before
    assert result is not value and result["plan"] is not value["plan"]
    assert result["entries"] is not value["entries"]
    return result


def rejected(value, *, expected_plan=None):
    """Require typed failure without normalizing invalid historical evidence."""
    expected = copy.deepcopy(value["plan"] if expected_plan is None else expected_plan)
    before, expected_before = copy.deepcopy(value), copy.deepcopy(expected)
    with pytest.raises(ValueError, match="planned_accounting"):
        validate_accounting_snapshot(value, expected_plan=expected)
    assert value == before and expected == expected_before


def rehash_plan(value):
    """Update local checksums while leaving semantic arithmetic to the reader."""
    plan = value["plan"]
    plan["envelope"]["identity"] = digest(
        {k: v for k, v in plan["envelope"].items() if k != "identity"}
    )
    plan["identity"] = digest({k: v for k, v in plan.items() if k != "identity"})


def rehash_revisions(value):
    """Rebind all revision hashes so a bad payload cannot fail only its checksum."""
    parent = value["plan"]["identity"]
    for revision in value["capacity_revisions"]:
        revision["parent"] = parent
        allocation = revision["allocation"]
        allocation["envelope"]["identity"] = digest(
            {k: v for k, v in allocation["envelope"].items() if k != "identity"}
        )
        allocation["identity"] = digest(
            {k: v for k, v in allocation.items() if k != "identity"}
        )
        revision["identity"] = digest(
            {k: v for k, v in revision.items() if k != "identity"}
        )
        parent = revision["identity"]
    value["capacity_allocation"] = copy.deepcopy(
        value["capacity_revisions"][-1]["allocation"]
    )


def produced_plan(case, policy):
    """Use the actual historical producer instead of hand-building cost plans."""
    if policy == "planned-budget-v2":
        return static_plan()
    if policy == "planned-budget-v3":
        return adaptive_writer_plan(
            identity="controlled-historical-costs", stage_bounds=certified_bounds()
        )
    return writer_budget_plan(
        case.packet, case.aliases, budget_policy=policy, methodology={}
    )


def test_original040_cost_fixture_preserves_all22_settled_usage_records(original040):
    assert original040["science_gold_standard"] is False
    assert original040["provenance"]["cost_metadata_only"] is True
    snapshot = original040["snapshot"]
    encoded = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == original040["provenance"]["snapshot_canonical_sha256"]
    )
    assert (
        snapshot["plan"]["identity"]
        == original040["provenance"]["frozen_manifest_plan_identity"]
    )
    records = original040["usage_records"]
    assert len(records) == len(snapshot["entries"]) == 22
    assert {r["request_id"] for r in records} == set(snapshot["entries"])
    for record in records:
        entry = snapshot["entries"][record["request_id"]]
        assert entry["sent"] is True and entry["status"] == "settled"
        usage = {k: record[k] for k in ("prompt_tokens", "completion_tokens")}
        assert sum(usage.values()) == record["total_tokens"] == entry["actual_tokens"]
        assert digest(usage) == entry["usage_sha256"]
    assert sum(r["total_tokens"] for r in records) == snapshot["used_tokens"] == 344347
    assert snapshot["call_count"] == snapshot["requests_sent"] == 22
    assert snapshot["usage_uncertain"] is False


def test_fixture_does_not_export_private_model_or_http_records(original040):
    forbidden = {
        "operation_id",
        "trace_id",
        "messages",
        "api_key",
        "secret",
        "headers",
        "base_url",
        "provider_receipt",
        "request_body_base64",
        "response_body_base64",
    }

    def scan(value):
        """Inspect keys without confusing token counts with credentials."""
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    scan(original040)
    assert "/Users/" not in json.dumps(original040, ensure_ascii=False)


def test_original040_pinned_costs_are_readable_without_runtime_resume(original040):
    snapshot = copy.deepcopy(original040["snapshot"])
    frozen = copy.deepcopy(snapshot)
    result = checked(snapshot)
    assert result["requests_sent"] == len(result["entries"]) == 22
    assert result["used_tokens"] == 344347
    assert result["usage_uncertain"] is False
    result["entries"].clear()
    assert snapshot == frozen
    # The newer repair transport rejects this old runtime before support checks.
    with pytest.raises(ValueError, match="writer_envelope_transport_revision_invalid"):
        CostBudget.from_snapshot(snapshot)
    assert snapshot == frozen


def test_reader_cannot_construct_or_restore_a_budget_owner(original040, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical cost inspection cannot create a runtime owner")

    monkeypatch.setattr(CostBudget, "__init__", forbidden)
    monkeypatch.setattr(CostBudget, "from_snapshot", forbidden)
    monkeypatch.setattr(PlannedBudget, "__init__", forbidden)
    result = checked(copy.deepcopy(original040["snapshot"]))
    assert type(result) is dict and not hasattr(result, "reserve")


@pytest.mark.parametrize("policy", [f"planned-budget-v{i}" for i in range(2, 8)])
def test_known_schema2_generations_keep_their_original_data_contract(case, policy):
    plan = produced_plan(case, policy)
    assert plan["status"] == "admitted"
    snapshot = PlannedBudget(plan).snapshot()
    result = checked(snapshot)
    assert result["strict_policy"] == policy and result["used_tokens"] == 0


@pytest.mark.parametrize("policy", [f"planned-budget-v{i}" for i in range(2, 6)])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("round_number", False),
        ("round_wire_sha256", "forged"),
        ("inventory_recovery", {"arbitrary": "no-pre-v7-recovery"}),
    ],
)
def test_old_cost_contracts_reject_nonexistent_round_or_recovery_metadata(
    case, policy, field, value
):
    plan = produced_plan(case, policy)
    owner = PlannedBudget(plan)
    with request_scope("body", "old-cost-receipt"):
        owner.reserve(20000)
        owner.begin_request()
        owner.record({"prompt_tokens": 12, "completion_tokens": 3})
    snapshot = owner.snapshot()
    checked(snapshot, expected_plan=plan)
    snapshot["entries"]["old-cost-receipt"][field] = value
    rejected(snapshot, expected_plan=plan)


@pytest.mark.parametrize("tamper", [False, True])
def test_v6_actual_reallocation_math_survives_without_a_runtime_owner(case, tamper):
    plan = produced_plan(case, "planned-budget-v6")
    owner = PlannedBudget(plan)
    settle_initial(owner)
    owner.expand_capacity(144, 9)
    for index in range(9):
        settle(owner, "assessment", f"assessment:{index}", 1000)
    for stage, key in (
        ("global", "global:0"),
        ("repair", "repair:0"),
        ("summary", "summary:1"),
        ("inventory", "inventory:1"),
    ):
        settle(owner, stage, key, 1000)
    owner.expand_capacity(160, 10)
    snapshot = owner.snapshot()
    checked(snapshot, expected_plan=plan)
    assert snapshot["used_tokens"] == 21000
    assert len(snapshot["capacity_revisions"]) == 2
    assert snapshot["capacity_allocation"]["stages"]["assessment"]["count"] == 29
    if tamper:
        snapshot["capacity_revisions"][1]["allocation"]["stages"]["assessment"][
            "reserved_tokens"
        ] += 1
        rehash_revisions(snapshot)
        rejected(snapshot, expected_plan=plan)


@pytest.mark.parametrize(
    ("status", "sent", "uncertain", "reason"),
    [
        ("started_no_replay", True, False, None),
        ("interrupted_no_replay", True, True, "strict_usage_unknown"),
        ("interrupted_no_replay", False, True, "strict_usage_unknown"),
        ("reserved_not_sent", False, False, None),
    ],
)
def test_pending_cost_states_are_preserved_without_normalization(
    original040, status, sent, uncertain, reason
):
    snapshot = copy.deepcopy(original040["snapshot"])
    entry = next(reversed(snapshot["entries"].values()))
    snapshot["used_tokens"] -= entry["actual_tokens"]
    snapshot["call_count"] -= 1
    snapshot["requests_sent"] -= int(not sent)
    snapshot["usage_uncertain"], snapshot["stop_reason"] = uncertain, reason
    entry.update(status=status, sent=sent, actual_tokens=None, usage_sha256=None)
    result = checked(snapshot)
    actual = next(reversed(result["entries"].values()))
    assert actual["status"] == status and actual["sent"] is sent
    assert result["used_tokens"] == 329412 and result["call_count"] == 21
    assert result["usage_uncertain"] is uncertain


@pytest.mark.parametrize(
    ("status", "sent", "uncertain"),
    [
        ("reserved_not_sent", True, False),
        ("started_no_replay", False, False),
        ("interrupted_no_replay", True, False),
    ],
)
def test_pending_state_flags_cannot_claim_an_impossible_send_history(
    original040, status, sent, uncertain
):
    snapshot = copy.deepcopy(original040["snapshot"])
    entry = next(reversed(snapshot["entries"].values()))
    snapshot["used_tokens"] -= entry["actual_tokens"]
    snapshot["call_count"] -= 1
    snapshot["requests_sent"] -= int(not sent)
    snapshot["usage_uncertain"] = uncertain
    entry.update(status=status, sent=sent, actual_tokens=None, usage_sha256=None)
    rejected(snapshot)


def test_recorded_budget_bound_violation_keeps_the_known_service_charge():
    plan = static_plan()
    owner = PlannedBudget(plan)
    with request_scope("body", "known-charge-above-reservation"):
        owner.reserve(40000)
        owner.begin_request()
        with pytest.raises(ValueError, match="planned_usage_bound_violation"):
            owner.record({"prompt_tokens": 40000, "completion_tokens": 1})
    snapshot = owner.snapshot()
    result = checked(snapshot, expected_plan=plan)
    assert result["used_tokens"] == 40001
    assert result["stop_reason"] == "planned_usage_bound_violation"
    assert result["call_count"] == result["requests_sent"] == 1


def test_untransmitted_future_global_slot_is_not_invented_as_spend(original040):
    snapshot = copy.deepcopy(original040["snapshot"])
    slot = snapshot["capacity_allocation"]["round"]["requests"][-1]
    assert slot["stage"] == "global"
    snapshot["entries"]["controlled-future-global-reservation"] = {
        "stage": "global",
        "status": "reserved_not_sent",
        "reserved_tokens": slot["reserved_tokens"],
        "actual_tokens": None,
        "usage_sha256": None,
        "sent": False,
        "round_number": 2,
        "round_wire_sha256": slot["wire_sha256"],
    }
    result = checked(snapshot)
    assert result["used_tokens"] == 344347
    assert result["requests_sent"] == result["call_count"] == 22
    assert result["entries"]["controlled-future-global-reservation"]["sent"] is False


@pytest.mark.parametrize("revisions", [0, 1])
def test_original_revision_entry_prefixes_are_valid_cost_observations(
    original040, revisions
):
    snapshot = copy.deepcopy(original040["snapshot"])
    entries = snapshot["capacity_revisions"][revisions]["entries"]
    snapshot["entries"] = copy.deepcopy(entries)
    snapshot["capacity_revisions"] = snapshot["capacity_revisions"][:revisions]
    snapshot["capacity_allocation"] = (
        copy.deepcopy(snapshot["capacity_revisions"][-1]["allocation"])
        if revisions
        else None
    )
    snapshot["used_tokens"] = sum(e["actual_tokens"] for e in entries.values())
    snapshot["call_count"] = snapshot["requests_sent"] = len(entries)
    checked(snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 3),
        ("max_tokens", True),
        ("used_tokens", False),
        ("used_tokens", -1),
        ("used_tokens", 344347.0),
        ("call_count", True),
        ("requests_sent", False),
        ("usage_uncertain", "false"),
        ("stop_reason", "ignore_old_failures"),
        ("entries", []),
        ("closed_stages", ["unknown"]),
        ("closed_stages", ["body", "body"]),
        ("capacity_allocation", None),
        ("capacity_revisions", {}),
        ("unrecognized_scope", True),
    ],
)
def test_snapshot_fields_reject_coercion_unknown_scope_and_missing_allocations(
    original040, field, value
):
    snapshot = copy.deepcopy(original040["snapshot"])
    snapshot[field] = value
    rejected(snapshot)


@pytest.mark.parametrize(
    "fault", ["used_total", "settled_count", "sent_count", "drop_sent_entry", "order"]
)
def test_totals_and_revision_entry_order_cannot_be_rewritten(original040, fault):
    snapshot = copy.deepcopy(original040["snapshot"])
    if fault == "used_total":
        snapshot["used_tokens"] -= 1
    elif fault == "settled_count":
        snapshot["call_count"] -= 1
    elif fault == "sent_count":
        snapshot["requests_sent"] -= 1
    elif fault == "drop_sent_entry":
        snapshot["entries"].pop(next(iter(snapshot["entries"])))
    else:
        entries = list(snapshot["entries"].items())
        entries[:2] = reversed(entries[:2])
        snapshot["entries"] = dict(entries)
    rejected(snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "unknown"),
        ("status", "paid"),
        ("sent", 1),
        ("sent", False),
        ("actual_tokens", True),
        ("actual_tokens", -1),
        ("actual_tokens", None),
        ("reserved_tokens", True),
        ("reserved_tokens", 0),
        ("reserved_tokens", 200000),
        ("usage_sha256", None),
        ("usage_sha256", "not-a-digest"),
        ("wire_sha256", "z" * 64),
        ("extra_untrusted_field", True),
    ],
)
def test_entry_shapes_states_and_numeric_bounds_are_strict(original040, field, value):
    snapshot = copy.deepcopy(original040["snapshot"])
    next(iter(snapshot["entries"].values()))[field] = value
    rejected(snapshot)


@pytest.mark.parametrize(
    "fault",
    [
        "plan_digest",
        "envelope_digest",
        "contract",
        "stage_count",
        "output_cap",
        "total",
        "allowance",
        "maximum",
        "maximum_requests",
        "context",
        "missing_evaluation",
        "repair_limit",
        "body_units",
        "request_cap",
        "support_revision",
        "support_identity",
        "ancestor",
        "boolean_stage",
    ],
)
def test_rehashed_plans_still_require_the_complete_frozen_path(original040, fault):
    snapshot = copy.deepcopy(original040["snapshot"])
    plan = snapshot["plan"]
    if fault == "plan_digest":
        plan["identity"] = "0" * 64
    elif fault == "envelope_digest":
        plan["envelope"]["identity"] = "0" * 64
    else:
        if fault == "contract":
            plan["contract"] = "planned-budget-v99"
        elif fault == "stage_count":
            plan["stages"]["body"]["count"] = 3
        elif fault == "output_cap":
            plan["stages"]["reviewer"]["output_tokens"] = 4095
        elif fault == "total":
            plan["required_tokens"] -= 1
        elif fault == "allowance":
            plan["planned_tokens"] -= 10000
        elif fault == "maximum":
            plan["maximum_tokens"] = 5000000
        elif fault == "maximum_requests":
            plan["maximum_requests"] = 65
        elif fault == "context":
            plan["context_margin"] = 0.2
        elif fault == "missing_evaluation":
            plan["initial_evaluation_reserved"] = False
        elif fault == "repair_limit":
            plan["repair_limit"] = 3
        elif fault == "body_units":
            plan["envelope"]["body_max_units"] = 67
        elif fault == "request_cap":
            plan["envelope"]["request_caps"]["assessment"] = 99999
        elif fault == "support_revision":
            plan["envelope"]["support_validation_revision"] = "unknown"
        elif fault == "support_identity":
            plan["envelope"]["support_validation_identity"] = True
        elif fault == "ancestor":
            plan["envelope"].pop("transport_revision")
        elif fault == "boolean_stage":
            plan["stages"]["reviewer"]["count"] = True
        rehash_plan(snapshot)
    rejected(snapshot)


def test_a_rehashed_but_different_expected_plan_does_not_match_original040(original040):
    snapshot = copy.deepcopy(original040["snapshot"])
    other = copy.deepcopy(snapshot)
    other["plan"]["input_identity"] = "a" * 64
    rehash_plan(other)
    rejected(snapshot, expected_plan=other["plan"])


@pytest.mark.parametrize(
    "stage", ["body", "summary", "inventory", "assessment", "global", "repair"]
)
def test_balanced_rehashed_plan_still_cannot_drop_a_complete_path_stage(
    original040, stage
):
    snapshot = copy.deepcopy(original040["snapshot"])
    plan = snapshot["plan"]
    spec = plan["stages"][stage]
    spec["count"] -= 1
    spec["reserved_tokens"] = spec["count"] * spec["request_cap"]
    plan["required_tokens"] = sum(s["reserved_tokens"] for s in plan["stages"].values())
    plan["planned_requests"] = sum(s["count"] for s in plan["stages"].values())
    plan["planned_tokens"] = max(
        500000, ((plan["required_tokens"] * 11 + 99999) // 100000) * 10000
    )
    snapshot.update(
        entries={},
        used_tokens=0,
        call_count=0,
        requests_sent=0,
        max_tokens=plan["planned_tokens"],
        capacity_revisions=[],
        capacity_allocation=None,
    )
    rehash_plan(snapshot)
    rejected(snapshot)


@pytest.mark.parametrize(
    "fault",
    [
        "required_total",
        "margin",
        "request_count",
        "stage_reserve",
        "offset",
        "round_number",
        "remaining_repairs",
        "consumed_wire",
        "future_output",
        "saved_known_entry",
        "saved_entry_prefix",
        "envelope_cap",
    ],
)
def test_locally_rehashed_capacity_revisions_cannot_rewrite_their_execution(
    original040, fault
):
    snapshot = copy.deepcopy(original040["snapshot"])
    revision = snapshot["capacity_revisions"][1]
    allocation = revision["allocation"]
    if fault == "required_total":
        allocation["required_tokens"] -= 1
    elif fault == "margin":
        allocation["with_margin_tokens"] -= 10000
    elif fault == "request_count":
        allocation["planned_requests"] += 1
    elif fault == "stage_reserve":
        allocation["stages"]["global"]["reserved_tokens"] -= 1
    elif fault == "offset":
        allocation["round"]["offsets"]["assessment"] -= 1
    elif fault == "round_number":
        allocation["round"]["number"] = 3
    elif fault == "remaining_repairs":
        allocation["round"]["remaining_repairs"] = 2
    elif fault == "consumed_wire":
        revision["arguments"]["requests"][0]["wire_sha256"] = "a" * 64
        allocation["round"]["requests"][0]["wire_sha256"] = "a" * 64
    elif fault == "future_output":
        revision["arguments"]["requests"][-1]["output_tokens"] = 16383
        allocation["round"]["requests"][-1]["output_tokens"] = 16383
    elif fault == "saved_known_entry":
        next(iter(revision["entries"].values()))["actual_tokens"] += 1
    elif fault == "saved_entry_prefix":
        revision["entries"].pop(next(iter(revision["entries"])))
    else:
        allocation["envelope"]["request_caps"]["global"] -= 1
    rehash_revisions(snapshot)
    rejected(snapshot)


@pytest.mark.parametrize("fault", ["parent", "allocation_mirror", "missing_revision"])
def test_capacity_parent_chain_and_last_allocation_are_exact(original040, fault):
    snapshot = copy.deepcopy(original040["snapshot"])
    if fault == "parent":
        revision = snapshot["capacity_revisions"][1]
        revision["parent"] = "a" * 64
        revision["identity"] = digest(
            {k: v for k, v in revision.items() if k != "identity"}
        )
    elif fault == "allocation_mirror":
        snapshot["capacity_allocation"]["round"]["number"] = 1
    else:
        snapshot["capacity_revisions"].pop(0)
    rejected(snapshot)


@pytest.mark.parametrize(
    "field",
    ["parent_allocation", "source_report_sha256", "contract", "feedback_sha256"],
)
def test_inventory_recovery_keeps_its_original_parent_and_purpose(original040, field):
    snapshot = copy.deepcopy(original040["snapshot"])
    key, entry = next(
        (key, e) for key, e in snapshot["entries"].items() if "inventory_recovery" in e
    )
    entry["inventory_recovery"][field] = "a" * 64 if field != "contract" else "unknown"
    if field == "feedback_sha256":
        entry["inventory_recovery"][field] = "invalid"
    for revision in snapshot["capacity_revisions"]:
        if key in revision["entries"]:
            revision["entries"][key] = copy.deepcopy(entry)
    rehash_revisions(snapshot)
    rejected(snapshot)
