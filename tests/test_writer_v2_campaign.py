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
    assert (
        public["new_usage"]["known_tokens"] == 0
        and public["new_usage"]["request_hold"] == 64
    )
    assert public["judge"] == public["jev"] == "disabled"


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
