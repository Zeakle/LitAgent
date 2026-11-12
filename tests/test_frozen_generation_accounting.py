"""Bind billing corrections to original responses without refunding old holds."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_frozen_generation_jev import trial
from test_frozen_generation_parent import child, finish

from litagent.benchmark import citation_audit
from litagent.benchmark.citation_audit import CitationAuditUsage
from scripts.run_frozen_generation import (
    build_accounting_correction,
    prepare,
    run_trial,
    validate_accounting_correction,
)


async def failed_parent(tmp_path, monkeypatch):
    """Reproduce the historical lost-usage branch with real producer contracts."""
    monkeypatch.setattr(
        citation_audit,
        "_failed_response_usage",
        lambda payload: CitationAuditUsage(
            input_tokens=None,
            output_tokens=None,
            call_count=1,
            missing_usage_calls=1,
            usage_complete=False,
        ),
    )
    root, cfg, history, grant, state, artifact, *_ = await trial(
        tmp_path, monkeypatch, "probability"
    )
    assert state["costs"]["jev"]["usage_complete"] is False
    assert state["cumulative_after"]["jev"]["charged"] == 200000
    return root, cfg, history, grant, state, artifact


def save_correction(parent):
    """Append a separate artifact instead of modifying any original receipt."""
    value = build_accounting_correction(parent)
    path = parent / "accounting-correction.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.asyncio
async def test_explicit_correction_preserves_hold_and_allows_new_identity_dry_run(
    tmp_path, monkeypatch
):
    parent, cfg, history, grant, state, _ = await failed_parent(tmp_path, monkeypatch)
    old = (parent / "checkpoint.json").read_bytes()
    root = child(parent, "corrected-child")
    with pytest.raises(ValueError):
        prepare(
            root,
            cfg,
            "corrected-next",
            history,
            grant,
            parent_root=parent,
            quality_policy="strict_auto_v2",
        )
    correction, record = save_correction(parent)
    assert record["actual_usage"] == {
        "input_tokens": 100,
        "output_tokens": 5,
        "call_count": 1,
    }
    assert record["retained_budget_charge"] == {"tokens": 200000, "requests": 1}
    plan = prepare(
        root,
        cfg,
        "corrected-next",
        history,
        grant,
        parent_root=parent,
        quality_policy="strict_auto_v2",
        accounting_correction=correction,
    )
    assert plan["cumulative_before"]["jev"] == {
        "charged": 200000,
        "requests": 1,
        "usage_complete": True,
    }
    assert (
        plan["rounds_before"] == 2 and plan["remaining_rounds_after_reservation"] == 15
    )
    assert (await run_trial(root, cfg, plan))["paid_calls"] == 0
    assert not (root / "checkpoint.json").exists()
    assert (parent / "checkpoint.json").read_bytes() == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "low_usage",
        "refund",
        "wrong_parent",
        "raw_response",
        "missing_response",
        "missing_start",
        "duplicate_operation",
        "source_change",
    ],
)
async def test_correction_cannot_cover_unknown_forged_or_changed_material(
    tmp_path, monkeypatch, mutation
):
    parent, cfg, history, grant, state, artifact = await failed_parent(
        tmp_path, monkeypatch
    )
    correction, record = save_correction(parent)
    if mutation == "low_usage":
        record["actual_usage"]["input_tokens"] = 0
    elif mutation == "refund":
        record["retained_budget_charge"]["tokens"] = 100
    elif mutation == "wrong_parent":
        record["source_root"] = str(parent.parent / "foreign")
    elif mutation in {"raw_response", "missing_response", "duplicate_operation"}:
        path = Path(state["extraction_receipt"]["path"])
        receipt = json.loads(path.read_text())
        batch = receipt["jev"]["batches"][0]
        if mutation == "raw_response":
            batch["response"]["usage"]["input_tokens"] += 1
        elif mutation == "missing_response":
            batch["response"] = None
        else:
            receipt["jev"]["batches"].append(copy.deepcopy(batch))
        path.write_text(json.dumps(receipt))
    elif mutation == "missing_start":
        path = Path("artifacts/runs/new-jev.json")
        artifact["events"] = [
            e
            for e in artifact["events"]
            if not (
                e["event"] == "subspan.start"
                and e["data"].get("name") == "citation_audit.batch"
            )
        ]
        path.write_text(json.dumps(artifact))
    else:
        path = parent / "input-snapshot.json"
        inp = json.loads(path.read_text())
        inp["question"] = "Changed query"
        path.write_text(json.dumps(inp))
    correction.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate_accounting_correction(correction, parent)


@pytest.mark.asyncio
async def test_bound_correction_is_inherited_by_the_next_parent_chain(
    tmp_path, monkeypatch
):
    parent, cfg, history, grant, *_ = await failed_parent(tmp_path, monkeypatch)
    correction, _ = save_correction(parent)
    root = child(parent, "child-one")
    plan = prepare(
        root,
        cfg,
        "next-known",
        history,
        grant,
        parent_root=parent,
        quality_policy="strict_auto_v2",
        accounting_correction=correction,
    )
    finish(root, plan, operation="next-known-writing")
    third = child(root, "child-two")
    plan2 = prepare(
        third,
        cfg,
        "third-known",
        history,
        grant,
        parent_root=root,
        quality_policy="strict_auto_v2",
    )
    assert plan2["cumulative_before"]["jev"]["charged"] == 200000
    assert plan2["cumulative_before"]["survey"]["charged"] == 110
    assert plan2["rounds_before"] == 3
    assert str(parent.resolve()) in plan2["accounting_corrections"]


@pytest.mark.asyncio
async def test_response_without_usage_cannot_be_recovered_by_a_known_flag(
    tmp_path, monkeypatch
):
    root, *rest = await trial(tmp_path, monkeypatch, "usage")
    with pytest.raises(ValueError):
        build_accounting_correction(root)


@pytest.mark.asyncio
async def test_missing_operation_end_cannot_be_covered_by_receipt_totals(
    tmp_path, monkeypatch
):
    parent, _, _, _, _, archive = await failed_parent(tmp_path, monkeypatch)
    archive["events"] = [
        e
        for e in archive["events"]
        if not (
            e["event"] == "subspan.end" and "batch_id" in e["data"].get("output", {})
        )
    ]
    Path("artifacts/runs/new-jev.json").write_text(json.dumps(archive))
    with pytest.raises(ValueError, match="response_binding_invalid"):
        build_accounting_correction(parent)
