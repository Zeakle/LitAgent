"""Keep the three-case paid boundary and restart behavior deterministic offline."""

import copy
import csv
import hashlib
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
        "tests/test_writer_sections.py",
        "scripts/run_frozen_writing.py",
        "tests/test_writer_v2.py",
        "tests/test_writer_v2_campaign.py",
        "tests/test_writer_content.py",
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


@pytest.mark.parametrize("fault", [None, "running", "version", "unknown"])
def test_terminal_failed_protocol_cost_is_inherited_without_restarting(
    tmp_path, monkeypatch, fault
):
    """Known paid failures stay in the ledger; uncertain and arbitrary states do not."""
    monkeypatch.chdir(tmp_path)
    history = {"model_requests": 726, "model_known_tokens": 3841217}
    root = accounting_parent(tmp_path, "compat", history, 12537)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.pop("identity")
    manifest["version"] = (
        "unreviewed" if fault == "version" else "writer-v2-protocol-observation-v2"
    )
    manifest["identity"] = campaign.digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    state = json.loads((root / "state.json").read_text())
    state.update(
        identity=manifest["identity"],
        status="running" if fault == "running" else "blocked",
        failure={"family": "protocol", "code": "writer_json_invalid"},
    )
    if fault == "unknown":
        state["usage_complete"] = False
    atomic_json(root / "state.json", state)
    if fault:
        with pytest.raises(ValueError, match="accounting_.*(invalid|unknown)"):
            campaign.accounting_history(history, [root])
    else:
        result, receipts = campaign.accounting_history(history, [root])
        assert result == {"model_requests": 727, "model_known_tokens": 3853754}
        assert receipts[0]["known_tokens"] == 12537


@pytest.mark.parametrize(
    "fault", [None, "sha", "run_id", "missing_reason", "unsafe_reason"]
)
def test_export_reads_exact_bound_evaluation_cause_without_modifying_old_state(
    tmp_path, monkeypatch, fault
):
    """A view can explain a terminal run without changing its original score or cause."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "suite"
    root.mkdir()
    report = {
        "survey": "Original frozen scientific report.",
        "evaluation": {
            "faithfulness": {
                "details": {
                    "skipped_reason": "evaluation_assertion_capacity_exceeded",
                    "unverified_count": 116,
                    "stages": {
                        "expected_assertions": 132,
                        "processed_assertions": 0,
                        "requests": [{"raw": "private response"}],
                    },
                }
            }
        },
        "metadata": {
            "evidence_rewrite_attempts": [
                {"attempted": False, "reason_code": "writer_assessment_incomplete"},
                {
                    "attempted": True,
                    "candidate_evaluation": None,
                    "reason_code": "failed_candidate",
                },
            ]
        },
    }
    if fault == "missing_reason":
        report["evaluation"]["faithfulness"]["details"].pop("skipped_reason")
    if fault == "unsafe_reason":
        report["evaluation"]["faithfulness"]["details"][
            "skipped_reason"
        ] = "Private model response with credentials"
    original_sha = hashlib.sha256(report["survey"].encode()).hexdigest()
    row = {
        "case_id": campaign.CASES[0][0],
        "run_id": "bound",
        "status": "blocked",
        "generated": True,
        "reviewer_completed": True,
        "evaluation": "not_complete",
        "publishable": False,
        "supported": 0,
        "unsupported": 0,
        "unverified": 116,
        "requests": 4,
        "known_tokens": 89534,
        "unknown_hold": 0,
        "report_sha256": "changed" if fault == "sha" else original_sha,
        "failure_code": "evaluation_not_completed",
        "evaluation_history": [
            {"stage": "initial"},
            {"stage": "repair_1"},
            {"stage": "repair_2"},
        ],
    }
    state = {
        "identity": "suite",
        "status": "stopped",
        "stop_reason": "evaluation_not_completed",
        "cases": [row],
    }
    manifest = {
        "identity": "suite",
        "source_commit": "original-execution",
        "history": {},
        "limits": dict(campaign.LIMITS),
    }
    atomic_json(root / "state.json", state)
    archive = {
        "run_id": "different" if fault == "run_id" else "bound",
        "report": report,
    }
    path = tmp_path / "artifacts/runs/bound.json"
    atomic_json(path, archive)
    before = copy.deepcopy(state)
    original_bytes = (root / "state.json").read_bytes()
    original_archive = path.read_bytes()
    campaign.export(root, state, manifest)
    out = tmp_path / "benchmarks/survey/results/suite"
    public = json.loads((out / "benchmark.json").read_text())
    actual = public["cases"][0]
    assert state == before and (root / "state.json").read_bytes() == original_bytes
    assert path.read_bytes() == original_archive
    assert (
        actual["failure_code"] == "evaluation_not_completed"
        and not actual["publishable"]
    )
    assert public["planned"] == 3 and public["run"] == 1 and public["not_run"] == 2
    assert public["new_usage"]["known_tokens"] == 89534
    serialized = (out / "benchmark.json").read_text()
    assert "private response" not in serialized and "credentials" not in serialized
    if fault not in {"sha", "run_id", "missing_reason"}:
        expected = (
            "evaluation_not_completed"
            if fault == "unsafe_reason"
            else "evaluation_assertion_capacity_exceeded"
        )
        assert actual["evaluation_reason"] == expected
        assert (
            actual["expected_assertions"] == 132 and actual["processed_assertions"] == 0
        )
        csv_row = next(csv.DictReader((out / "benchmark.csv").open()))
        assert (
            csv_row["evaluation_reason"] == expected
            and csv_row["expected_assertions"] == "132"
        )
        assert expected in (out / "benchmark.md").read_text()
        assert expected in (out / "benchmark.html").read_text()
    else:
        assert "evaluation_diagnostic" not in actual
        assert actual["expected_assertions"] is None
    if fault not in {"sha", "run_id"}:
        assert [h["stage"] for h in actual["evaluation_history"]] == [
            "initial",
            "repair_2",
        ]
        assert actual["repair_preflight_failures"] == [
            {"attempted": False, "reason_code": "writer_assessment_incomplete"}
        ]


@pytest.mark.parametrize("field", ["error", "reason"])
def test_old_evaluation_reason_fields_remain_readable_without_invented_counts(field):
    """Missing versioned statistics stay unavailable instead of using current defaults."""
    report = {
        "evaluation": {"faithfulness": {"details": {field: "evaluation_incomplete"}}}
    }
    assert campaign.evaluation_diagnostic(report) == {
        "reason_code": "evaluation_incomplete",
        "expected_assertions": None,
        "processed_assertions": None,
        "scientific_unverified": None,
    }


@pytest.mark.parametrize("report_sha", [None, "final-version-sha"])
def test_public_link_does_not_claim_a_report_for_failed_generation(
    tmp_path, monkeypatch, report_sha
):
    """A compiled or failed draft alone only has a diagnostic archive to inspect."""
    monkeypatch.chdir(tmp_path)
    row = {
        "case_id": campaign.CASES[0][0],
        "run_id": "bound",
        "status": "blocked",
        "generated": True,
        "evaluation": "not_complete",
        "publishable": False,
        "report_sha256": report_sha,
        "compiled_report_sha256": "draft-only",
        "requests": 6,
        "known_tokens": 100,
        "unknown_hold": 0,
        "failure_code": "writer_json_invalid",
    }
    state = {"identity": "suite", "status": "stopped", "cases": [row]}
    manifest = {
        "identity": "suite",
        "source_commit": "frozen",
        "history": {},
        "limits": dict(campaign.LIMITS),
    }
    original = copy.deepcopy(state)
    campaign.export(tmp_path / "suite", state, manifest)
    out = tmp_path / "benchmarks/survey/results/suite"
    page = (out / "benchmark.html").read_text()
    assert ("质量与原文 / Report" in page) == bool(report_sha)
    assert ("运行诊断 / Diagnostics" in page) == (not report_sha)
    assert state == original
    public = json.loads((out / "benchmark.json").read_text())
    assert public["cases"][0]["report_sha256"] == report_sha
    assert public["planned"] == 3 and public["run"] == 1 and public["passed"] == 0


def json_failure_call(raw, *, resolved_by=None):
    """Generate diagnostics through the actual writer parser, without a transport."""
    from litagent.writing_v2 import WriterError, parse_response

    with pytest.raises(WriterError) as error:
        parse_response(raw, {})
    assert error.value.code == "writer_json_invalid"
    result = {
        "stage": "synthesis",
        "raw": raw,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "finish_reason": "stop",
        "failure": {
            "code": error.value.code,
            "family": error.value.family,
            "diagnostics": error.value.diagnostics,
        },
    }
    if resolved_by:
        result["resolved_by"] = resolved_by
    return result


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("bound", "body:experiments"),
        ("repair_call", "repair_1"),
        ("legacy_unique", "body:experiments"),
        ("legacy_ambiguous", None),
        ("bound_missing", None),
        ("bound_resolved", None),
        ("different_error", None),
        ("conflicting_terminal", None),
        ("invalid_sidecar", None),
        ("path_escape", None),
        ("missing_diagnostics", None),
        ("offset_bool", None),
        ("offset_negative", None),
        ("offset_overflow", None),
        ("offset_wrong_in_range", None),
        ("call_sha_conflict", None),
        ("diagnostic_sha_conflict", None),
        ("bad_stage", None),
        ("stage_object", None),
        ("finish_object", None),
    ],
)
def test_protocol_export_binds_only_current_unresolved_error(
    tmp_path, monkeypatch, scenario, expected
):
    """Resolved failures cannot impersonate the current location in any format."""
    monkeypatch.chdir(tmp_path)
    early = json_failure_call(
        '{"taxonomy": [\n {"sentences": []}', resolved_by="repair_0"
    )
    final = json_failure_call('{"experiments": [\n  {"sentences": []}')
    journal = {
        "calls": {"body:taxonomy": early, "body:experiments": final},
        "failure": {"code": "writer_json_invalid", "failed_call": "body:experiments"},
    }
    run_id = "bound"
    failure_code = "writer_json_invalid"
    if scenario == "repair_call":
        journal["calls"]["repair_1"] = json_failure_call('{"experiments": [')
        journal["failure"]["call"] = "repair_1"
    if scenario.startswith("legacy"):
        journal.pop("failure")
    if scenario == "legacy_ambiguous":
        early.pop("resolved_by")
    if scenario == "bound_missing":
        journal["failure"]["failed_call"] = "missing"
    if scenario == "bound_resolved":
        journal["failure"]["failed_call"] = "body:taxonomy"
    if scenario == "different_error":
        failure_code = "responses_incomplete"
    if scenario == "conflicting_terminal":
        journal["failure"]["code"] = "responses_incomplete"
    if scenario == "path_escape":
        run_id = "../escape"
    diagnostic = final["failure"]["diagnostics"][0]
    if scenario == "missing_diagnostics":
        final["failure"]["diagnostics"] = []
    if scenario.startswith("offset_"):
        diagnostic["offset"] = {
            "offset_bool": True,
            "offset_negative": -1,
            "offset_overflow": len(final["raw"]) + 1,
            "offset_wrong_in_range": 1,
        }[scenario]
    if scenario == "call_sha_conflict":
        final["raw_sha256"] = "f" * 64
    if scenario == "diagnostic_sha_conflict":
        diagnostic["raw_sha256"] = "f" * 64
    if scenario == "bad_stage":
        final["stage"] = "untrusted scientific prose"
    if scenario == "stage_object":
        final["stage"] = {"text": "must not be published"}
    if scenario == "finish_object":
        final["finish_reason"] = {"text": "must not be published"}
    path = Path("artifacts/targeted-repairs", run_id + ".json.writer-v2.json")
    atomic_json(path, journal)
    if scenario == "invalid_sidecar":
        path.write_text("not JSON")
    before_journal = path.read_bytes()
    state = {
        "identity": "suite",
        "status": "stopped",
        "cases": [
            {
                "case_id": campaign.CASES[0][0],
                "run_id": run_id,
                "status": "blocked",
                "generated": False,
                "evaluation": "not_complete",
                "publishable": False,
                "requests": 6,
                "known_tokens": 100,
                "failure_code": failure_code,
                "protocol_diagnostic": {"obsolete": "must be replaced"},
            }
        ],
    }
    original = copy.deepcopy(state)
    manifest = {
        "identity": "suite",
        "source_commit": "actual-run-source",
        "history": {},
        "limits": dict(campaign.LIMITS),
    }
    public = campaign.export(tmp_path / "suite", state, manifest)
    assert state == original and path.read_bytes() == before_journal
    assert public["source_commit"] == "actual-run-source"
    assert public["view_version"] == "frozen-writing-view-v10"
    assert public["passed"] == 0 and public["cases"][0]["failure_code"] == failure_code
    row = public["cases"][0]
    out = tmp_path / "benchmarks/survey/results/suite"
    html = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            html.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    csv_row = list(csv.DictReader((out / "benchmark.csv").open()))[0]
    if expected is None:
        assert "protocol_diagnostic" not in row
        assert row["protocol_call"] is None and csv_row["protocol_call"] == ""
    else:
        call = journal["calls"][expected]
        bound = row["protocol_diagnostic"]
        assert bound["call"] == csv_row["protocol_call"] == expected
        assert (
            bound["raw_response_sha256"]
            == call["raw_sha256"]
            == csv_row["protocol_raw_response_sha256"]
        )
        assert bound["offset"] == len(call["raw"]) == int(csv_row["protocol_offset"])
        markdown = (out / "benchmark.md").read_text()
        assert expected in markdown and call["raw_sha256"] in markdown
        assert bound["scientific_assessment"] == "not_run"
