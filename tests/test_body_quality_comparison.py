"""Check bounded paired scheduling and cumulative no-replay accounting offline."""

import copy
import json
from pathlib import Path

import pytest
from test_strict_automatic import frozen

from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.quality import digest as evidence_digest
from litagent.report_audit import atomic_json
from scripts import run_body_quality_comparison as comparison
from scripts.run_frozen_generation import file_sha, settle


@pytest.fixture(autouse=True)
def isolated_exports(tmp_path, monkeypatch):
    """Never expose mock benchmark rows in the live workbench archive."""
    monkeypatch.setattr(comparison, "BENCHMARK_ROOT", tmp_path / "benchmarks")


def prepared(tmp_path, *, formal_enabled=True):
    """Freeze complete local inputs and historical accounting for offline trials."""
    previous = tmp_path / "previous"
    base = frozen()
    archive = tmp_path / "source.json"
    atomic_json(archive, {"run_id": "old", "report": base["report"]})
    selection = EvidenceSelection(
        1,
        base["ledger"],
        {
            k: ["a"]
            for k in (
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            )
        },
        {
            k: "lexical_fallback"
            for k in (
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            )
        },
        0,
        50,
    )
    for domain in comparison.CASE_IDS:
        path = previous / domain
        path.mkdir(parents=True)
        atomic_json(
            path / "input-snapshot.json",
            {
                "question": "EViT speed",
                "evidence_selection": selection.to_dict(),
                "fact_cards": base["report"]["metadata"]["fact_cards"],
                "fact_cards_sha256": evidence_digest(
                    base["report"]["metadata"]["fact_cards"]
                ),
                "source_artifact": str(archive),
                "source_artifact_sha256": file_sha(archive),
            },
        )
    history = {
        "total_rounds": 16,
        "cumulative_after": {
            s: {
                "charged": 200000 if s == "jev" else 1000,
                "requests": 2,
                "usage_complete": s != "jev",
            }
            for s in comparison.ZERO
        },
    }
    atomic_json(previous / "vit" / "checkpoint.json", history)
    root = tmp_path / "body-paired-test"
    cfg = load_config()
    manifest = comparison.prepare(root, cfg, previous, formal_enabled=formal_enabled)
    return root, cfg, manifest


def trial_result(plan, *, technical=False, passed=False):
    """Return a fully accounted transport fixture without paying a service."""
    costs = {
        s: {
            "tokens": 10 if s == "survey" else 0,
            "requests": 1 if s == "survey" else 0,
            "usage_complete": not technical if s == "survey" else True,
        }
        for s in comparison.ZERO
    }
    return {
        "status": "completed" if passed else "blocked",
        "reason": [],
        "budget": {},
        "costs": costs,
        "cumulative_after": settle(plan, costs),
        "report": {
            "metadata": {
                "automatic_acceptance": {
                    "publishable": passed,
                    "gates": {
                        "evaluation": "passed" if passed else "failed",
                        "judge": "disabled",
                        "extraction_jev": "disabled",
                    },
                }
            },
            "evaluation": {
                "faithfulness": {
                    "details": {
                        "coverage": {"complete": True},
                        "stages": {"global_check": "completed"},
                    }
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_dry_run_and_scientific_failures_keep_all_six_and_formal_denominator(
    tmp_path,
):
    """Exercise dry run and scientific failures keep all six and formal denominator."""
    root, cfg, manifest = prepared(tmp_path)
    calls = []

    async def trial(slot, config, plan, **kwargs):
        """Simulate a single external execution boundary with durable reservations."""
        calls.append(plan)
        pending = json.loads((root / "comparison.json").read_text())
        assert (
            pending["budgets"]["survey"]["charged"]
            == plan["cumulative_before"]["survey"]["charged"] + 500000
        )
        assert (
            plan["verification"]["judge"] is False
            and plan["verification"]["jev"] is False
        )
        assert plan["evaluation_body_quality"] == "analysis_v1"
        return trial_result(plan)

    assert (await comparison.run(root, cfg))["paid_calls"] == 0
    assert not (root / "comparison.json").exists()
    result = await comparison.run(root, cfg, live=True, trial=trial)
    assert len(calls) == 6 and len(result["observations"]) == 6
    assert result["budgets"]["survey"]["charged"] == 60
    assert result["stop_reasons"] == ["candidate_quality_goal_not_met"]
    assert manifest["historical"]["services"]["jev"]["charged"] == 200000
    shown = comparison.presentation(root, manifest, result)
    assert (
        shown["counts"]["pilot"]["planned"] == 6
        and shown["counts"]["pilot"]["not_run"] == 0
    )
    assert (
        shown["counts"]["formal"]["planned"] == 15
        and shown["counts"]["formal"]["not_run"] == 15
    )
    await comparison.run(root, cfg, live=True, trial=trial)
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_unknown_usage_stops_next_slot_and_retains_full_allowance(tmp_path):
    """Exercise unknown usage stops next slot and retains full allowance."""
    root, cfg, _ = prepared(tmp_path)
    calls = []

    async def trial(slot, config, plan, **kwargs):
        """Simulate a single external execution boundary with durable reservations."""
        calls.append(plan)
        return trial_result(plan, technical=True)

    result = await comparison.run(root, cfg, live=True, trial=trial)
    assert len(calls) == 1 and result["budgets"]["survey"]["charged"] == 500000
    assert (
        result["status"] == "stopped" and "usage_unknown" in result["stop_reasons"][0]
    )


@pytest.mark.asyncio
async def test_cancelled_pending_request_cannot_be_replayed_or_refunded(tmp_path):
    """Exercise cancelled pending request cannot be replayed or refunded."""
    import asyncio

    root, cfg, _ = prepared(tmp_path)
    calls = []

    async def trial(*args, **kwargs):
        """Simulate a single external execution boundary with durable reservations."""
        calls.append(1)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await comparison.run(root, cfg, live=True, trial=trial)
    result = await comparison.run(root, cfg, live=True, trial=trial)
    assert len(calls) == 1 and result["budgets"]["survey"]["charged"] == 500000
    assert result["stop_reasons"] == ["body_comparison_interrupted_no_replay"]


@pytest.mark.asyncio
async def test_input_mutation_stops_before_any_transport(tmp_path):
    """Exercise input mutation stops before any transport."""
    root, cfg, manifest = prepared(tmp_path)
    Path(manifest["cases"][0]["root"], "input-snapshot.json").write_text("{}")
    with pytest.raises(ValueError, match="identity_changed"):
        await comparison.run(root, cfg, live=True)


@pytest.mark.asyncio
async def test_three_candidates_pass_before_formal_is_dispatched(tmp_path, monkeypatch):
    """Exercise three candidates pass before formal is dispatched."""
    root, cfg, _ = prepared(tmp_path)
    formal = []

    async def trial(slot, config, plan, **kwargs):
        """Simulate a single external execution boundary with durable reservations."""
        return trial_result(plan, passed=plan["body_quality"] == "analysis_v1")

    class Formal:
        def __init__(self, *args, **kwargs):
            """Require three passed candidate observations before formal execution."""
            candidates = [
                r
                for r in kwargs["pilot_state"]["observations"]
                if r["generation_variant"] == "candidate"
            ]
            assert len(candidates) == 3 and all(
                r["acceptance"]["publishable"] for r in candidates
            )

        async def run(self, **kwargs):
            """Capture the formal dispatch limit without running a paid Survey."""
            formal.append(kwargs)
            return {"status": "completed", "observations": [], "budgets": {}}

    monkeypatch.setattr(comparison, "FormalBodyRunner", Formal)
    result = await comparison.run(root, cfg, live=True, trial=trial)
    assert result["formal_eligible"] and formal == [{"max_new_observations": 15}]


@pytest.mark.asyncio
async def test_formal_only_ledger_has_no_pilot_or_disabled_service_reservations(
    tmp_path,
):
    """Use the actual formal scheduler and persist only the selected service budget."""
    from unittest.mock import AsyncMock

    from test_strict_automatic import suite

    from litagent.benchmark.quality_suite import summarize
    from litagent.workbench import RunOptions, VerificationOptions

    template = suite(tmp_path / "template", executor=None)
    state = {
        "observations": [
            {"generation_variant": "candidate", "acceptance": {"publishable": True}}
            for _ in range(3)
        ]
    }
    runner = comparison.FormalBodyRunner(
        template.config,
        template.dataset,
        template.profile,
        root=tmp_path / "formal",
        run_archive=template.archive,
        suite_id="offline-formal",
        run_options=RunOptions(
            quality_policy="strict_auto_v2",
            body_quality="analysis_v1",
            verification=VerificationOptions(),
        ),
        pilot_manifest={"binding": "offline"},
        pilot_state=state,
    )
    initial = runner.initial()
    assert summarize(initial)["pilot"] == {
        "planned": 0,
        "run": 0,
        "complete_verification": 0,
        "automatic_passed": 0,
        "pending": 0,
        "terminal": 0,
        "blocked": 0,
        "not_run": 0,
        "pass_rate_planned": None,
        "pass_rate_completed": None,
    }
    assert initial["budgets"]["survey"]["aggregate"] == 7500000
    assert all(initial["budgets"][s]["aggregate"] == 0 for s in ("judge", "jev"))

    async def executor(case, config, run_id):
        persisted = json.loads(runner.state_path.read_text())
        assert persisted["budgets"]["survey"]["charged"] == 500000
        assert all(persisted["budgets"][s]["charged"] == 0 for s in ("judge", "jev"))
        return {
            "status": "blocked",
            "acceptance": {"publishable": False},
            "costs": {
                s: {"tokens": 0, "requests": 0, "usage_complete": True}
                for s in comparison.ZERO
            },
        }

    runner.executor = executor
    runner.service.initialize = AsyncMock()
    runner.service.shutdown = AsyncMock()
    result = await runner.run(max_new_observations=1)
    assert len(result["observations"]) == 1
    assert result["counts"]["pilot"]["planned"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("technical", [False, True])
async def test_new_suite_preserves_previous_comparison_and_unknown_reservations(
    tmp_path, technical
):
    """Carry prior paid outcomes forward while keeping the new bounded ledger separate."""
    root, cfg, previous = prepared(tmp_path)

    async def trial(slot, config, plan, **kwargs):
        return trial_result(plan, technical=technical)

    state = await comparison.run(root, cfg, live=True, trial=trial)
    frozen_prior = (root / "comparison.json").read_bytes()
    manifest = comparison.prepare(
        tmp_path / "next-suite", cfg, tmp_path / "previous", prior_comparison=root
    )
    assert manifest["historical"]["rounds"] == 16 + len(state["observations"])
    assert manifest["historical"]["services"]["survey"]["charged"] == (
        previous["historical"]["services"]["survey"]["charged"]
        + state["budgets"]["survey"]["charged"]
    )
    assert manifest["historical"]["services"]["jev"]["charged"] == 200000
    assert manifest["historical"]["services"]["jev"]["usage_complete"] is False
    assert (
        manifest["historical"]["services"]["survey"]["usage_complete"] is not technical
    )
    assert manifest["limits"]["survey_tokens"] == 3000000
    assert (root / "comparison.json").read_bytes() == frozen_prior
    assert not (tmp_path / "next-suite" / "comparison.json").exists()
    assert manifest["hashes"][str((root / "comparison.json").resolve())] == file_sha(
        root / "comparison.json"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["undercharged", "pending"])
async def test_invalid_previous_comparison_cannot_prepare_a_new_paid_suite(
    tmp_path, corruption
):
    """Reject active work or missing known costs before creating transport state."""
    root, cfg, _ = prepared(tmp_path)

    async def trial(slot, config, plan, **kwargs):
        return trial_result(plan)

    state = await comparison.run(root, cfg, live=True, trial=trial)
    if corruption == "undercharged":
        state["budgets"]["survey"]["charged"] = 0
    else:
        state["pending"] = {"run_id": "unknown"}
    atomic_json(root / "comparison.json", state)
    with pytest.raises(ValueError, match="body_comparison_prior"):
        comparison.prepare(
            tmp_path / "next-suite", cfg, tmp_path / "previous", prior_comparison=root
        )


@pytest.mark.asyncio
async def test_pilot_only_manifest_cannot_dispatch_formal_even_after_all_candidates_pass(
    tmp_path, monkeypatch
):
    """Enforce the six-round boundary in the actual scheduler and frozen manifest."""
    root, cfg, manifest = prepared(tmp_path, formal_enabled=False)
    calls = []

    async def trial(slot, config, plan, **kwargs):
        calls.append(plan["run_id"])
        return trial_result(plan, passed=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Formal requests are outside this bounded execution")

    monkeypatch.setattr(comparison, "FormalBodyRunner", forbidden)
    dry = await comparison.run(root, cfg)
    assert dry["rounds"] == 6 and dry["formal_conditional"] == 0
    result = await comparison.run(root, cfg, live=True, trial=trial)
    assert len(calls) == 6 and manifest["formal_enabled"] is False
    assert result["formal_eligible"] is True
    assert result["stop_reasons"] == ["formal_disabled_for_bounded_pilot"]
    assert not result.get("pending_formal")
    assert result["budgets"]["survey"]["charged"] == 60
    await comparison.run(root, cfg, live=True, trial=trial)
    assert len(calls) == 6
