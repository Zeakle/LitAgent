"""Keep the three-case paid boundary and restart behavior deterministic offline."""

import copy
import json
from pathlib import Path

import pytest

from litagent.report_audit import atomic_json
from scripts import run_frozen_writing as campaign


@pytest.mark.asyncio
async def test_campaign_restart_holds_unsettled_case_without_any_request(
    tmp_path, monkeypatch
):
    """A process interruption cannot clear costs or resend the uncertain case."""
    root = tmp_path / "suite"
    root.mkdir()
    manifest = {
        "identity": "frozen",
        "source_commit": "commit",
        "history": {},
        "limits": dict(campaign.LIMITS),
    }
    monkeypatch.setattr(campaign, "validate", lambda *_: manifest)
    monkeypatch.chdir(tmp_path)
    state = {
        "identity": "frozen",
        "status": "running",
        "cases": [
            {
                "case_id": "few-shot-metric",
                "run_id": "suite-few-shot-metric",
                "status": "started_no_replay",
            }
        ],
    }
    atomic_json(root / "state.json", state)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must not create a new transport")

    monkeypatch.setattr(campaign, "LitAgent", forbidden)
    result = await campaign.execute(root, None)
    assert result["status"] == "interrupted"
    assert result["cases"][0]["unknown_hold"] == 500000
    assert result["cases"][0]["request_hold"] == 64
    assert not result["cases"][0]["usage_complete"]
    assert await campaign.execute(root, None) == result
    public = json.loads(
        (tmp_path / "benchmarks/survey/results/suite/benchmark.json").read_text()
    )
    assert public["planned"] == 3 and public["not_run"] == 2 and public["passed"] == 0
    assert [r["case_id"] for r in public["cases"]] == [c for c, _ in campaign.CASES]
    assert all(
        r["run_id"] is None and r["status"] == "not_run" for r in public["cases"][1:]
    )
    assert (
        public["new_usage"]["known_tokens"] == 0
        and public["new_usage"]["request_hold"] == 64
    )
    assert public["judge"] == public["jev"] == "disabled"
    assert (
        ">None<"
        not in (tmp_path / "benchmarks/survey/results/suite/benchmark.html").read_text()
    )


def test_campaign_rejects_limits_runtime_and_case_drift_before_transport(
    tmp_path, monkeypatch
):
    """Recomputing a manifest digest does not authorize a wider campaign."""
    from litagent.config import load_config
    from litagent.contracts import build_config_summary

    cfg = load_config()
    monkeypatch.setattr(campaign, "check_models", lambda _c: None)
    monkeypatch.setattr(campaign, "source_paths", lambda: [])
    root = tmp_path / "suite"
    root.mkdir()
    original = {
        "identity": "pending",
        "version": campaign.VERSION,
        "contract": campaign.CONTRACT,
        "limits": dict(campaign.LIMITS),
        "runtime": campaign.runtime_identity(),
        "options": campaign.frozen_options().model_dump(mode="json"),
        "config": build_config_summary(cfg, "strict_auto_v2"),
        "source_hashes": {},
        "cases": [
            {"case_id": c, "bundle": {"source_run_id": r}} for c, r in campaign.CASES
        ],
    }
    for mutation in ("limit", "runtime", "order", "strategy"):
        value = copy.deepcopy(original)
        if mutation == "limit":
            value["limits"]["cases"] = 18
        if mutation == "runtime":
            value["runtime"]["python"] = "different"
        if mutation == "order":
            value["cases"].reverse()
        if mutation == "strategy":
            value["options"]["generation_strategy"] = "body_summary_v1"
        value["identity"] = campaign.digest(
            {k: v for k, v in value.items() if k != "identity"}
        )
        atomic_json(root / "manifest.json", value)
        with pytest.raises(ValueError, match="identity_conflict"):
            campaign.validate(root, cfg)


def test_campaign_source_inventory_includes_new_consumers_and_tests():
    """Source freezing includes files added before the first implementation commit."""
    paths = {str(p) for p in campaign.source_paths()}
    assert {
        "src/litagent/writing_v2.py",
        "src/litagent/frozen_writing.py",
        "scripts/run_frozen_writing.py",
        "tests/test_writer_v2.py",
        "tests/test_writer_v2_campaign.py",
    } <= paths
    assert campaign.LIMITS["formal_cases"] == 0 and len(campaign.CASES) == 3


def accounting_parent(tmp_path, name, history, tokens):
    """Create a terminal receipt with independently reconcilable SDK events."""
    root = tmp_path / name
    root.mkdir()
    manifest = {"history": dict(history)}
    manifest["identity"] = campaign.digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "state.json",
        {
            "identity": manifest["identity"],
            "status": "protocol_completed",
            "known_tokens": tokens,
            "requests": 1,
            "usage_complete": True,
            "unknown_hold": 0,
        },
    )
    atomic_json(
        tmp_path / "artifacts/runs" / (name + ".json"),
        {
            "run_id": name,
            "events": [
                {"event": "llm.start", "data": {"operation_id": name}},
                {
                    "event": "llm.complete",
                    "data": {
                        "operation_id": name,
                        "usage_available": True,
                        "prompt_tokens": tokens - 1,
                        "completion_tokens": 1,
                        "total_tokens": tokens,
                    },
                },
            ],
        },
    )
    return root


def test_accounting_prefix_keeps_both_paid_observations_and_old_unknown(
    tmp_path, monkeypatch
):
    """Protocol attempts add costs, without becoming new initial Survey rounds."""
    monkeypatch.chdir(tmp_path)
    history = {
        "model_requests": 723,
        "model_known_tokens": 3801740,
        "model_unknown_conservative_tokens": 1030414,
        "initial_generation_rounds": 35,
    }
    p1 = accounting_parent(tmp_path, "first", history, 12477)
    after_first = {**history, "model_requests": 724, "model_known_tokens": 3814217}
    p2 = accounting_parent(tmp_path, "protocol", after_first, 14578)
    result, receipts = campaign.accounting_history(history, [p1, p2])
    assert result == {**history, "model_requests": 725, "model_known_tokens": 3828795}
    assert [r["known_tokens"] for r in receipts] == [12477, 14578]
    assert history["model_requests"] == 723
    assert campaign.accounting_history(history, [p1, p2]) == (result, receipts)
    with pytest.raises(ValueError, match="duplicate"):
        campaign.accounting_history(history, [p1, p1])
    with pytest.raises(ValueError, match="prefix_invalid"):
        campaign.accounting_history(history, [p2, p1])


@pytest.mark.parametrize("fault", ["count", "unknown", "operation", "tokens"])
def test_accounting_cannot_hide_unknown_or_forge_completed_usage(
    tmp_path, monkeypatch, fault
):
    """A receipt needs terminal, complete usage reconciled with the original events."""
    monkeypatch.chdir(tmp_path)
    history = {"model_requests": 1, "model_known_tokens": 100}
    root = accounting_parent(tmp_path, "parent", history, 30)
    if fault in {"count", "unknown"}:
        path = root / "state.json"
        state = json.loads(path.read_text())
        state.update(
            {"known_tokens": 29} if fault == "count" else {"usage_complete": False}
        )
        atomic_json(path, state)
    else:
        path = tmp_path / "artifacts/runs/parent.json"
        archive = json.loads(path.read_text())
        end = archive["events"][-1]["data"]
        end.update(
            {"operation_id": "wrong"} if fault == "operation" else {"total_tokens": 31}
        )
        atomic_json(path, archive)
    with pytest.raises(ValueError, match="accounting_.*(invalid|unknown)"):
        campaign.accounting_history(history, [root])
