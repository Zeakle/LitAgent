"""Verify independent repair ownership and complete, optional target changes."""

import copy
import json

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent import review_policy
from litagent.config import load_config
from litagent.quality import (
    begin_targeted_repair,
    remaining_repair_allowances,
    targeted_repair_state,
)
from litagent.run_limits import (
    configured_run_limits,
    extended_repair_stages,
    repair_allowances,
)
from litagent.writing_flat import encode, parse
from litagent.writing_v2 import WriterError, schema_for


def test_independent_four_allowances_and_legacy_identity():
    from litagent.config import AppConfig

    assert not review_policy.enabled({"run_limits": None})
    assert not review_policy.enabled({})
    assert not review_policy.enabled(None)
    assert AppConfig(agent={}, logging={}).safety.planned_run_limits is None
    old = targeted_repair_state(None, "old", limit=11)
    assert remaining_repair_allowances(old) == {
        "capacity": 3,
        "inventory": 3,
        "prose": 5,
    }
    current = targeted_repair_state(None, "new", limit=12)
    for category, stage in (
        ("review_revision", "review"),
        ("evaluation_revision", "evaluation"),
        ("capacity", "native_initial"),
        ("inventory", "evaluation"),
    ):
        for i in range(3):
            entry = begin_targeted_repair(
                current, stage, str(i) * 64, category=category
            )
            entry["status"] = "failed" if i == 1 else "completed"
            assert remaining_repair_allowances(current)[category] == 2 - i
        with pytest.raises(ValueError, match="exhausted|exceeded"):
            begin_targeted_repair(current, stage, "a" * 64, category=category)
    assert len(current["attempts"]) == 12
    assert remaining_repair_allowances(old)["prose"] == 5
    with pytest.raises(ValueError):
        targeted_repair_state(current, "new", limit=11)


def test_independent_inflight_cannot_be_replayed():
    state = targeted_repair_state(None, "x", limit=12)
    begin_targeted_repair(state, "review", "a" * 64)
    with pytest.raises(ValueError, match="restart_no_replay"):
        targeted_repair_state(state, "x")
    assert targeted_repair_state(state, "x", allow_inflight=True) == state


def test_independent_budget_partitions_and_format_counts():
    config = load_config("config/separate-review-evaluation.yaml")
    envelope = {
        "run_limits": configured_run_limits(config),
        "semantic_engine": {"batch_size": 16},
        "body_max_units": 66,
        "summary_max_units_per_section": 2,
    }
    assert repair_allowances(envelope) == review_policy.ALLOWANCES
    stages = extended_repair_stages(envelope, 12)
    assert stages["reviewer"] == 4
    assert stages["review_revision/repair"] == stages["evaluation_revision/repair"] == 3
    assert stages["evaluation_revision/assessment"] == 15
    assert stages["inventory/assessment"] == 15
    assert stages["assessment"] == 5
    assert "repair" not in stages
    with review_policy.partition("review_revision"):
        assert (
            review_policy.stage_name(envelope, "summary") == "review_revision/summary"
        )
    assert review_policy.stage_name(envelope, "summary") == "summary"
    assert review_policy.stage_name({}, "repair", "review_revision") == "repair"
    formats = review_policy.format_stages(stages)
    assert formats["format_review_revision/repair"][1] == 9
    assert formats["format_evaluation_revision/summary"][1] == 9


def response_fixture():
    targets = {f"writer:methods:{i}:0": {"position": i} for i in range(36)}
    schema = review_policy.response_contract(
        schema_for("repair", ["c1"], targets), targets, ["c1"]
    )
    value = {
        "outcome": "patched",
        "explanation": "Respond to every issue independently.",
        "patches": [
            {
                "target_id": tid,
                "sentences": [
                    {
                        "text": "Supported replacement " + str(i),
                        "fact_card_refs": ["c1"],
                    }
                ],
            }
            for i, tid in enumerate(targets)
            if i < 5
        ],
        "target_responses": {
            f"t{i+1}": {
                "action": "replace" if i < 5 else "keep",
                "reason": "Original evidence inspected.",
                "reference_ids": ["c1"],
            }
            for i in range(36)
        },
    }
    return targets, schema, value


def test_all_targets_answered_only_five_replacements_with_original_binding():
    targets, schema, value = response_fixture()
    raw = json.dumps(encode(value, schema), ensure_ascii=False)
    assert '"target_id": "t1"' in raw
    parsed = parse(raw, schema)
    assert parsed["canonical"] == value
    changed = review_policy.validate_responses(parsed["canonical"], targets, ["c1"])
    assert len(changed) == 5
    for unit in parsed["bindings"]["units"]:
        span = unit["text_span"]
        assert json.loads(raw[span["start"] : span["end"]]).startswith(
            "Supported replacement"
        )


@pytest.mark.parametrize(
    "attack",
    [
        "missing",
        "unknown",
        "extra_patch",
        "duplicate_patch",
        "missing_patch",
        "no_basis",
        "foreign_basis",
    ],
)
def test_target_responses_do_not_silently_waive_problems(attack):
    targets, _, value = response_fixture()
    if attack == "missing":
        del value["target_responses"]["t36"]
    elif attack == "unknown":
        value["target_responses"]["t37"] = value["target_responses"].pop("t36")
    elif attack == "extra_patch":
        value["patches"].append({"target_id": list(targets)[-1], "sentences": []})
    elif attack == "duplicate_patch":
        value["patches"].append(copy.deepcopy(value["patches"][0]))
    elif attack == "missing_patch":
        value["patches"].pop()
    else:
        value["target_responses"]["t36"]["reference_ids"] = (
            [] if attack == "no_basis" else ["outside"]
        )
    with pytest.raises(WriterError):
        review_policy.validate_responses(value, targets, ["c1"])


def test_keep_and_cannot_resolve_never_create_synthetic_replacements():
    targets, schema, value = response_fixture()
    value["patches"] = []
    for response in value["target_responses"].values():
        response.update(action="cannot_resolve", reference_ids=[])
    assert (
        review_policy.validate_responses(
            parse(json.dumps(encode(value, schema)), schema)["canonical"],
            targets,
            ["c1"],
        )
        == {}
    )


def test_legacy_schema_does_not_accept_new_responses():
    targets, _, value = response_fixture()
    with pytest.raises(Exception):
        parse(json.dumps(value), schema_for("repair", ["c1"], targets))


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_separate_loops_projection_preserves_unverified_and_sha(locale):
    import hashlib

    from test_quality_view import render_repair_state

    from litagent.review_cycles import public_projection, result_projection

    state = targeted_repair_state(None, "run", limit=12)
    a = begin_targeted_repair(state, "review", "b" * 64)
    a["status"] = "completed"
    report = {
        "survey": "draft",
        "delivery": {"publishable": False},
        "metadata": {
            "targeted_repair_state": state,
            "evaluation_history": {
                "attempts": [
                    {
                        "attempted": False,
                        "reason_code": "ragas_recovery_prose_limit_reached",
                    }
                ]
            },
            "editorial_history": [
                {
                    "round": 0,
                    "status": "unchanged_with_response",
                    "source_report_sha256": "c" * 64,
                    "responses": {"t1": {"action": "keep", "reason": "PRIVATE"}},
                }
            ],
        },
    }
    sha = hashlib.sha256(b"draft").hexdigest()
    result = public_projection(result_projection(report), sha)
    assert result["evaluations"][0]["status"] == "unverified"
    assert result["evaluations"][1]["kind"] == "repair_preflight"
    assert result["allowances"]["evaluation_revision"]["remaining"] == 3
    assert "PRIVATE" not in json.dumps(result)
    assert public_projection(result, "d" * 64) is None
    rendered = render_repair_state(
        {
            "options": {
                "sha": sha,
                "reviewWorkflow": result,
                "targetedRepairState": state,
                "verification": {"judge": False, "jev": True},
            }
        },
        locale,
    )
    assert ("Writing review" if locale == "en" else "写作审阅") in rendered["text"]
    assert "t1: keep" in rendered["text"]
    from litagent.eval.results import CONTRACT, sanitize_summary, summary_tables

    summary = sanitize_summary(
        {"contract": CONTRACT, "report_sha256": sha, "review_workflow": result}
    )
    assert summary["review_workflow"] == result
    md, html = summary_tables([{"case_id": "offline", "evaluation_summary": summary}])
    assert "review_revision" in md and "review_revision" in html and "blocked" in html
    assert "not sent" in html and "未发送" in md


def test_408_only_current_science_targets_are_authorized():
    from pathlib import Path

    from litagent.eval.ragas_repair import native_repair_diagnostics
    from litagent.writing_v2 import repair_targets

    p = Path(
        "artifacts/acceptance/workbench-few-shot-408-20261111/runs/9a4732a08500.json"
    )
    if not p.exists():
        pytest.skip("408 private archive is not packaged")
    original = p.read_bytes()
    report = json.loads(original)["report"]
    m = report["metadata"]
    details = report["evaluation"]["faithfulness"]["details"]
    args = (
        m["writing_bindings"],
        m["writer_document"]["body"],
        m["writer_document"]["summaries"],
    )
    negative = native_repair_diagnostics(details)
    science = repair_targets(args[0], negative, *args[1:], merge=True)
    merged = repair_targets(
        args[0], negative + m["reviewer_diagnostics"]["issues"], *args[1:], merge=True
    )
    assert len(science) == 5 and len(merged) == 36
    assert details["supported_count"] == 404 and details["unsupported_count"] == 9
    assert report["delivery"]["publishable"] is False
    assert p.read_bytes() == original


def test_new_workflow_identity_and_parent_deadline_are_frozen():
    from litagent.config import AppConfig
    from litagent.run_limits import validate_run_limits
    from litagent.writer_source_view import source_view_identity
    from litagent.writing_flat import repair_prompt_identity

    assert (
        source_view_identity()
        == "59742f4dcd202b0e103680f77a5ec7de57d1b31f0aeea4dd977dfcfa92b07bcb"
    )
    assert (
        repair_prompt_identity()
        == "72077e3a4ea2a6e7ca883f62f876e8fb386aeed14638126b2a87c28890ecf078"
    )

    cfg = load_config("config/separate-review-evaluation.yaml")
    limits = configured_run_limits(cfg)
    assert limits["workflow_identity"] == review_policy.identity()
    with pytest.raises(ValueError, match="review_workflow_identity_invalid"):
        validate_run_limits({**limits, "workflow_identity": "a" * 64})
    data = cfg.model_dump()
    data["adversarial"]["task_timeout_ms"] = 27000000
    with pytest.raises(ValueError, match="deadline_insufficient"):
        AppConfig.model_validate(data)
