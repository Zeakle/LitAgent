"""Measure partition feasibility; these requests never authorize a global verdict."""

import copy
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest
from test_global_status_codes import global_request, vit_replay

from litagent.eval.ragas_global_wire import unpack_global_input
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import nested_json_size


@lru_cache(maxsize=1)
def frozen_case():
    """Keep the local immutable replay independent of provider transport."""
    return vit_replay()


def inventory(case, scenario):
    """Create dimension-only counterexamples, never scientific acceptance fixtures."""
    claims = copy.deepcopy(
        case[0]["evaluation"]["faithfulness"]["details"]["assertions"]
    )
    if scenario == "recorded":
        return claims
    selected = claims[:3] if scenario == "three_long" else claims
    for index, claim in enumerate(selected):
        # Each value remains within the current producer's individual byte limit.
        # Unique values rule out an apparent solution based on text deduplication.
        prefix = f"{index}:"
        claim["claim_text"] = prefix + "x" * (2394 - len(prefix))
    return claims


def request_for_units(case, claims, units):
    """Meter the production constructor on whole units, without a new wire format."""
    selected = {unit.unit_id for unit in units}
    local = tuple(case[:6]) + (units, case[7])
    canonical, messages, schema = global_request(
        local, [claim for claim in claims if claim["source_unit_id"] in selected]
    )
    assert unpack_global_input(json.loads(messages[1]["content"])) == canonical
    return {
        "reserved_tokens": strict_request_reservation(
            messages, case[3].eval.max_tokens, schema
        ),
        "payload": canonical,
    }


def partition_measurement(case, claims):
    """Preserve every unit and claim while measuring stable, whole-unit partitions."""
    limit = case[4]["context_reservation_limit"]
    batches, pending = [], []
    for unit in case[6]:
        candidate = request_for_units(case, claims, [*pending, unit])
        if candidate["reserved_tokens"] <= limit:
            pending.append(unit)
            continue
        if pending:
            batches.append(request_for_units(case, claims, pending))
        pending = [unit]
        single = request_for_units(case, claims, pending)
        if single["reserved_tokens"] > limit:
            return {"status": "unit_cannot_fit", "unit_id": unit.unit_id}
    if pending:
        batches.append(request_for_units(case, claims, pending))
    expected = Counter(claim["claim_id"] for claim in claims)
    actual = Counter(
        claim["claim_id"]
        for batch in batches
        for claim in batch["payload"]["assertions"]
    )
    assert actual == expected
    assert [
        unit["source_unit_id"]
        for batch in batches
        for unit in batch["payload"]["source_units"]
    ] == [unit.unit_id for unit in case[6]]
    return {
        "status": "partition_fits",
        "batches": batches,
        "global_semantics_verified": False,
    }


def resource_projection(plan, batch_reservations):
    """Show a proposal's costs, including every reserved complete reevaluation."""
    global_stages = {
        name: spec
        for name, spec in plan["stages"].items()
        if name.split("/")[-1] == "global"
    }
    rounds = sum(spec["count"] for spec in global_stages.values())
    base = sum(spec["reserved_tokens"] for spec in plan["stages"].values())
    replaced = sum(spec["reserved_tokens"] for spec in global_stages.values())
    # These are proposed wire costs, not valid budget revisions or paid usage.
    projected = base - replaced + rounds * sum(batch_reservations)
    requests = plan["planned_requests"] + rounds * (len(batch_reservations) - 1)
    with_margin = max(500000, math.ceil(projected * 11 / 100000) * 10000)
    return {
        "evaluation_rounds": rounds,
        "projected_tokens_with_margin": with_margin,
        "projected_requests": requests,
        "fits_locked_tokens": with_margin <= plan["planned_tokens"],
        "fits_hard_tokens": with_margin <= plan["maximum_tokens"],
        "fits_hard_requests": requests <= plan["maximum_requests"],
        "includes_whole_report_reducer": False,
        "includes_additional_nli_batches": False,
        "authorizes_execution": False,
    }


@lru_cache(maxsize=3)
def scenario_measurement(scenario):
    """Keep exact request measurements separate from a semantic design claim."""
    case = frozen_case()
    claims = inventory(case, scenario)
    full = request_for_units(case, claims, case[6])
    measured = partition_measurement(case, claims)
    assert measured["status"] == "partition_fits"
    batches = measured["batches"]
    costs = [batch["reserved_tokens"] for batch in batches]
    return {
        "scenario": scenario,
        "claim_count": len(claims),
        "claim_text_json_bytes": nested_json_size([c["claim_text"] for c in claims]),
        "largest_claim_json_bytes": max(
            nested_json_size(c["claim_text"]) for c in claims
        ),
        "full_request_reservation": full["reserved_tokens"],
        "context_limit": case[4]["context_reservation_limit"],
        "partition_reservations": costs,
        "partition_claim_counts": [len(b["payload"]["assertions"]) for b in batches],
        "partition_unit_counts": [len(b["payload"]["source_units"]) for b in batches],
        "all_claims_and_units_retained": True,
        "global_semantics_verified": False,
        "resource_projection": resource_projection(case[4], costs),
    }


@pytest.mark.parametrize("scenario", ["recorded", "three_long", "all_long"])
def test_whole_unit_partition_retains_complete_inventory(scenario):
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

    statements = [c["claim_text"] for c in inventory(frozen_case(), scenario)]
    assert (
        StatementGeneratorOutput.model_validate(
            {"statements": statements}, strict=True
        ).statements
        == statements
    )
    measured = scenario_measurement(scenario)
    assert measured["claim_count"] == 505
    assert sum(measured["partition_claim_counts"]) == 505
    assert max(measured["partition_reservations"]) <= measured["context_limit"]
    assert measured["largest_claim_json_bytes"] <= 2400
    assert measured["global_semantics_verified"] is False
    assert measured["resource_projection"]["authorizes_execution"] is False


def test_single_call_pass_does_not_prove_current_producer_capacity():
    recorded = scenario_measurement("recorded")
    longer = scenario_measurement("three_long")
    assert recorded["full_request_reservation"] == 207077
    assert longer["full_request_reservation"] > longer["context_limit"]
    assert len(longer["partition_reservations"]) > 1


def test_obsolete_aggregate_limit_cannot_be_reinstated_for_the_real_inventory():
    measured = scenario_measurement("recorded")
    assert measured["claim_text_json_bytes"] == 40447
    assert (
        measured["claim_text_json_bytes"]
        > frozen_case()[4]["envelope"]["claim_json_bytes"]
    )
    assert measured["full_request_reservation"] < measured["context_limit"]


def test_global_subsets_cannot_certify_cross_partition_conflicts():
    case = frozen_case()
    claims = inventory(case, "three_long")
    batches = partition_measurement(case, claims)["batches"]
    left, right = [b["payload"]["assertions"][0]["claim_id"] for b in batches[:2]]
    # A contradiction across these two claims has no permitted complete witness
    # in either request. Merging local positive judgments cannot establish it.
    assert not any(
        {left, right} <= {c["claim_id"] for c in b["payload"]["assertions"]}
        for b in batches
    )


def test_all_reserved_reevaluations_are_charged_in_design_projection():
    plan = frozen_case()[4]
    cap = plan["context_reservation_limit"]
    two = resource_projection(plan, [cap, cap])
    three = resource_projection(plan, [cap, cap, cap])
    assert two["evaluation_rounds"] == three["evaluation_rounds"] == 7
    assert two["projected_requests"] == plan["planned_requests"] + 7
    assert three["projected_requests"] == plan["planned_requests"] + 14
    assert two["fits_hard_tokens"] is True
    assert three["fits_hard_tokens"] is False
    assert two["fits_locked_tokens"] is False


def test_full_individual_capacity_exposes_the_missing_complete_flow_bound():
    measured = scenario_measurement("all_long")
    projection = measured["resource_projection"]
    assert measured["largest_claim_json_bytes"] <= 2400
    assert not (projection["fits_hard_tokens"] and projection["fits_hard_requests"])


if __name__ == "__main__":
    destination = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/global-capacity-design"
    )
    destination.mkdir(parents=True, exist_ok=True)
    case = frozen_case()
    result = {
        "purpose": "offline transport feasibility, not a global semantic execution",
        "paid_requests": 0,
        "plan_identity": case[4]["identity"],
        "source_report_sha256": case[0]["evaluation"]["faithfulness"]["details"][
            "source_report_sha256"
        ],
        "locked_tokens": case[4]["planned_tokens"],
        "hard_tokens": case[4]["maximum_tokens"],
        "hard_requests": case[4]["maximum_requests"],
        "scenarios": [
            scenario_measurement(name)
            for name in ("recorded", "three_long", "all_long")
        ],
        "at_request_cap": [
            resource_projection(case[4], [case[4]["context_reservation_limit"]] * count)
            for count in (1, 2, 3)
        ],
    }
    (destination / "measurements.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
