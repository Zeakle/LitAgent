"""Bind three distinct public inputs to one version and an intact paid ledger."""

import copy
import json
from pathlib import Path

import pytest
from test_frozen_generation import prepared
from test_frozen_generation_parent import child, finish, save

from litagent.contracts import build_config_summary
from scripts import run_frozen_generation as trial


def matrix_fixture(tmp_path, monkeypatch):
    """Prepare real input contracts and a known failed parent without transport."""
    parent, config, plan, inputs, history, grant = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    finish(parent, plan)
    cases = []
    for i, domain in enumerate(("vit", "few-shot", "nerf")):
        root = child(parent, domain)
        value = copy.deepcopy(inputs)
        source = json.loads(Path(value["source_artifact"]).read_text())
        source["query"] = domain
        source["run_id"] = "source-" + domain
        source_path = tmp_path / f"source-{domain}.json"
        save(source_path, source)
        value.update(
            question=domain,
            source_run_id=source["run_id"],
            source_artifact=str(source_path),
            source_artifact_sha256=trial.file_sha(source_path),
        )
        save(root / "input-snapshot.json", value)
        cases.append(
            {
                "domain": domain,
                "root": str(root),
                "run_id": f"new-{domain}",
                "input_sha256": trial.file_sha(root / "input-snapshot.json"),
            }
        )
    manifest = {
        "version": "frozen-three-domain-v1",
        "cases": cases,
        "rounds_before": 2,
        "parent_root": str(parent),
        "parent_binding": plan["binding"],
        "config": build_config_summary(config, "strict_auto_v2"),
        "code_hashes": {
            str(Path(trial.__file__).resolve()): trial.file_sha(Path(trial.__file__))
        },
    }
    path = tmp_path / "matrix.json"
    save(path, manifest)
    return parent, config, history, grant, path, manifest


def prepare_case(fixture, index):
    """Consume the real prepare path with the explicitly ordered prior trial."""
    parent, config, history, grant, path, manifest = fixture
    case = manifest["cases"][index]
    return trial.prepare(
        Path(case["root"]),
        config,
        case["run_id"],
        history,
        grant,
        parent_root=(
            parent if index == 0 else Path(manifest["cases"][index - 1]["root"])
        ),
        quality_policy="strict_auto_v2",
        matrix_path=path,
    )


def complete_case(fixture, index, plan):
    """Persist an ordinary scientific block with exact transport accounting."""
    case = fixture[-1]["cases"][index]
    root = Path(case["root"])
    state = finish(root, plan, operation=f"matrix-call-{index}")
    state.update(status="blocked", reason=["strict_scientific_support_incomplete"])
    save(root / "checkpoint.json", state)
    return state


@pytest.mark.asyncio
async def test_three_distinct_inputs_inherit_all_costs_without_replaying(
    tmp_path, monkeypatch
):
    fixture = matrix_fixture(tmp_path, monkeypatch)
    for index in range(3):
        plan = prepare_case(fixture, index)
        assert plan["rounds_before"] == 2 + index
        assert plan["cumulative_before"]["survey"] == {
            "charged": 110 + index * 10,
            "requests": 2 + index,
            "usage_complete": True,
        }
        assert plan["matrix"]["index"] == index
        root = Path(fixture[-1]["cases"][index]["root"])
        assert (await trial.run_trial(root, fixture[1], plan))["paid_calls"] == 0
        assert not (root / "checkpoint.json").exists()
        complete_case(fixture, index, plan)


def test_default_still_rejects_cross_question_parent(tmp_path, monkeypatch):
    parent, cfg, history, grant, _, manifest = matrix_fixture(tmp_path, monkeypatch)
    case = manifest["cases"][0]
    with pytest.raises(ValueError, match="parent_binding_invalid"):
        trial.prepare(
            Path(case["root"]),
            cfg,
            case["run_id"],
            history,
            grant,
            parent_root=parent,
            quality_policy="strict_auto_v2",
        )


@pytest.mark.parametrize(
    "change,reason",
    [
        ("code", "code_changed"),
        ("input", "input_changed"),
        ("order", "contract_invalid"),
        ("duplicate", "contract_invalid"),
        ("anchor", "anchor_changed"),
        ("config", "contract_invalid"),
        ("missing", "contract_invalid"),
        ("source", "source_changed"),
    ],
)
def test_changed_matrix_inputs_stop_before_any_trial(
    tmp_path, monkeypatch, change, reason
):
    fixture = matrix_fixture(tmp_path, monkeypatch)
    manifest = copy.deepcopy(fixture[-1])
    if change == "code":
        manifest["code_hashes"] = {}
    elif change == "input":
        manifest["cases"][2]["input_sha256"] = "wrong"
    elif change == "order":
        manifest["cases"].reverse()
    elif change == "duplicate":
        manifest["cases"][1]["run_id"] = manifest["cases"][0]["run_id"]
    elif change == "anchor":
        manifest["parent_binding"] = "wrong"
    elif change == "config":
        manifest["config"] = {}
    elif change == "missing":
        del manifest["cases"]
    elif change == "source":
        sp = tmp_path / "source-nerf.json"
        source = json.loads(sp.read_text())
        source["query"] = "changed"
        save(sp, source)
    save(fixture[-2], manifest)
    with pytest.raises(ValueError, match=reason):
        prepare_case(fixture, 0)
    assert not list(tmp_path.glob("vit/checkpoint.json"))


@pytest.mark.parametrize(
    "reason",
    [
        "evaluation_inventory_unit_coverage_invalid",
        "rewrite_context_budget_exceeded",
        "unknown_usage",
    ],
)
def test_technical_or_usage_failure_blocks_next_case(tmp_path, monkeypatch, reason):
    fixture = matrix_fixture(tmp_path, monkeypatch)
    plan = prepare_case(fixture, 0)
    state = complete_case(fixture, 0, plan)
    if reason == "unknown_usage":
        state["costs"]["survey"]["usage_complete"] = False
    else:
        state["reason"] = [reason]
    save(Path(fixture[-1]["cases"][0]["root"]) / "checkpoint.json", state)
    with pytest.raises(ValueError, match="matrix_stopped"):
        prepare_case(fixture, 1)


def test_prior_manifest_cannot_be_rewritten_between_cases(tmp_path, monkeypatch):
    fixture = matrix_fixture(tmp_path, monkeypatch)
    complete_case(fixture, 0, prepare_case(fixture, 0))
    fixture[-1]["note"] = "changed identity"
    save(fixture[-2], fixture[-1])
    with pytest.raises(ValueError, match="prior_identity_invalid"):
        prepare_case(fixture, 1)


def test_cannot_skip_first_case(tmp_path, monkeypatch):
    fixture = matrix_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="matrix_contract_invalid"):
        prepare_case(fixture, 1)


def test_candidate_protocol_failure_stops_even_when_original_is_retained():
    state = {
        "status": "blocked",
        "costs": {s: {"usage_complete": True} for s in trial.LIMITS},
        "report": {
            "metadata": {
                "evidence_rewrite_attempts": [
                    {
                        "candidate_evaluation": {
                            "faithfulness": {
                                "details": {
                                    "unverified_reason": "evaluation_global_issue_identity_invalid"
                                }
                            }
                        }
                    }
                ]
            }
        },
    }
    assert trial.matrix_stop_reason(state) == "evaluation_global_issue_identity_invalid"


@pytest.mark.parametrize(
    "reason", ["strict_writing_scope_notice_invalid", "strict_new_unknown_error", ""]
)
def test_unknown_execution_failure_does_not_continue(reason):
    state = {
        "status": "failed",
        "reason": reason,
        "costs": {s: {"usage_complete": True} for s in trial.LIMITS},
    }
    assert trial.matrix_stop_reason(state) is not None
