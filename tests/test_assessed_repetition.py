"""Preserve independent findings while rejecting an observed redundant repair."""

import copy
import json
from pathlib import Path

import pytest
from test_bound_body_repair import client_factory, parent_fixture

from litagent.observability.recorder import ArchiveRepository
from litagent.quality import evaluated_repair_repetitions, remove_assessed_repetition
from litagent.report_audit import atomic_json
from scripts import run_bound_body_repair as repair


def observed():
    """Load a public automatic observation, without treating it as human gold."""
    return json.loads(
        (Path(__file__).parent / "fixtures/assessed-repetition-real.json").read_text()
    )


def check(data):
    """Use actual source positions and complete assessment identities."""
    return evaluated_repair_repetitions(
        *(
            data[k]
            for k in ("original", "initial_evaluation", "candidate", "evaluation")
        )
    )


def test_observed_paraphrase_duplicate_has_same_assessed_finding_and_citation():
    data = observed()
    result = check(data)
    assert result["status"] == "failed"
    pair = result["duplicates"][0]
    assert pair["protected"]["claim_id"] == "claim_11"
    assert pair["repeated"]["claim_id"] == "claim_12"
    assert pair["protected"]["source_quote"] != pair["repeated"]["source_quote"]
    assert len(result["duplicates"]) == 1


@pytest.mark.parametrize(
    "difference",
    [
        "在无预训练条件",
        "在预训练条件",
        "准确率2%",
        "准确率3%",
        "不稳定",
        "在ImageNet上",
    ],
)
def test_distinct_conditions_numbers_and_negations_are_not_fuzzy_merged(difference):
    data = observed()
    data["evaluation"]["faithfulness"]["details"]["assertions"][12][
        "claim_text"
    ] += difference
    assert check(data)["status"] == "passed"


def test_cross_section_reuse_is_not_treated_as_same_section_repetition():
    data = observed()
    data["evaluation"]["faithfulness"]["details"]["assertions"][12][
        "section"
    ] = "另一章节"
    assert check(data)["status"] == "passed"


@pytest.mark.parametrize("fault", ["sha", "location", "mixed", "binding"])
def test_cleanup_refuses_unbound_identity_or_partial_scientific_unit(fault):
    data = observed()
    details = data["evaluation"]["faithfulness"]["details"]
    if fault == "sha":
        details["source_report_sha256"] = "wrong"
    elif fault == "location":
        details["assertions"][12]["source_start"] += 1
    elif fault == "mixed":
        details["assertions"].append(copy.deepcopy(details["assertions"][12]))
    else:
        data["bindings"]["report_sha256"] = "wrong"
    with pytest.raises(ValueError):
        remove_assessed_repetition(
            *(
                data[k]
                for k in (
                    "original",
                    "initial_evaluation",
                    "candidate",
                    "evaluation",
                    "bindings",
                )
            )
        )


def test_local_cleanup_preserves_every_other_byte_and_requires_full_new_evaluation():
    data = observed()
    untouched = copy.deepcopy(data)
    body, bindings, receipt = remove_assessed_repetition(
        *(
            data[k]
            for k in (
                "original",
                "initial_evaluation",
                "candidate",
                "evaluation",
                "bindings",
            )
        )
    )
    a, b = receipt["removed_start"], receipt["removed_end"]
    assert body == data["candidate"][:a] + data["candidate"][b:]
    assert receipt["removed_text"] == data["candidate"][a:b]
    assert receipt["duplicates"][0]["protected"]["source_quote"] in body
    assert all(row["rendered_unit"] in body for row in bindings["locations"])
    assert len(bindings["locations"]) == len(data["bindings"]["locations"]) - 1
    assert bindings["report_sha256"] != data["bindings"]["report_sha256"]
    assert receipt["requires_full_reassessment"]
    assert (
        receipt["independent_findings_removed"] == receipt["model_rewrite_calls"] == 0
    )
    assert data == untouched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [None, "unknown_usage", "refund", "counter", "interrupted"]
)
async def test_editorial_reassessment_inherits_ledger_without_any_model_rewrite(
    tmp_path, monkeypatch, fault
):
    parent, config, _ = parent_fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    repo, wire = ArchiveRepository(tmp_path / "runs"), []
    repair.prepare(first, parent, config)
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        await repair.run(
            first,
            config,
            live=True,
            client_factory=client_factory(monkeypatch, wire, fault="inline"),
            repository=repo,
        )
    repair.prepare(second, parent, config, prior_repair=first)
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        await repair.run(
            second,
            config,
            live=True,
            client_factory=client_factory(monkeypatch, wire),
            repository=repo,
        )
    state = json.loads((second / "checkpoint.json").read_text())
    data = observed()
    # Replay the pre-fix automatic observation; the new runner must reject it.
    state["report"]["survey"] = data["candidate"]
    state["report"]["evaluation"]["faithfulness"]["details"] = data["evaluation"][
        "faithfulness"
    ]["details"]
    state["report"]["metadata"]["writing_bindings"] = data["bindings"]
    candidate = json.loads((second / "candidate.json").read_text())
    candidate.update(
        candidate=data["candidate"], candidate_evaluation=state["report"]["evaluation"]
    )
    atomic_json(second / "candidate.json", candidate)
    if fault == "refund":
        state["budget"]["used_tokens"] -= 1
    elif fault == "counter":
        state["report"]["metadata"]["targeted_repair_state"]["attempts"].pop()
    atomic_json(second / "checkpoint.json", state)
    before = {p.name: p.read_bytes() for p in second.iterdir() if p.is_file()}
    root = tmp_path / "editorial"
    if fault in {"refund", "counter"}:
        with pytest.raises(ValueError):
            repair.prepare_editorial(root, second, config)
        return
    plan = repair.prepare_editorial(root, second, config)
    assert plan["max_repairs"] == 0
    assert plan["source_budget"]["used_tokens"] == 100
    assert (await repair.run(root, config))["paid_calls"] == 0
    if fault == "interrupted":
        atomic_json(
            root / "checkpoint.json",
            {"status": "started_no_replay", "binding": plan["binding"]},
        )
    wire.clear()
    factory = client_factory(
        monkeypatch, wire, fault="unknown_usage" if fault == "unknown_usage" else None
    )
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        result = await repair.run(
            root, config, live=True, client_factory=factory, repository=repo
        )
        again = await repair.run(
            root, config, live=True, client_factory=factory, repository=repo
        )
    assert again == result
    assert all(
        json.loads(r["messages"][1]["content"]).get("repair_targets") is None
        for r in wire
    )
    if fault == "interrupted":
        assert len(wire) == 0
        assert result["status"] == "interrupted_no_replay"
        return
    assert len(result["report"]["metadata"]["targeted_repair_state"]["attempts"]) == 2
    assert result["report"]["metadata"]["evaluation_only"]["model_rewrite_calls"] == 0
    assert result["budget"]["requests_sent"] == 5 + len(wire)
    assert "approval" not in result["report"]["metadata"]
    assert result["report"]["survey"] != data["candidate"]
    assert all(
        (second / name).read_bytes() == contents for name, contents in before.items()
    )
    if fault is None:
        assert len(wire) == 3
        assert result["usage_complete"]
        assert result["budget"]["used_tokens"] == 160
        assert result["report"]["evaluation"]["faithfulness"]["details"]["coverage"][
            "complete"
        ]
    else:
        assert len(wire) == 1
        assert not result["report"]["delivery"]["publishable"]
        assert not result["usage_complete"]
