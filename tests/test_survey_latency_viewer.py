"""Verify archived benchmark runs open through the normal workbench frontend."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from litagent.benchmark.survey_latency import (
    CONTRACT,
    SUMMARY_CHECKPOINT_SCOPE,
    render_table,
    run_slots,
    summary_rows,
)
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY
from scripts.view_survey_latency import viewer_app


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, obj) -> str:
    raw = json.dumps(obj, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha(raw)


def write_manifest(root, manifest):
    approval = sha(json.dumps(manifest, sort_keys=True).encode())
    write(root / "manifest.json", {**manifest, "approval_sha256": approval})
    write(root / "started.json", {"approval_sha256": approval})


def fixture_campaign(root, *, diagnostic=False, blocked_replays=()):
    slots = run_slots()
    manifest = {
        "contract": CONTRACT,
        "faithfulness_policy": BENCHMARK_FAITHFULNESS_POLICY,
        "threshold": 0.95,
        "runs": [asdict(slot) for slot in slots],
    }
    write_manifest(root, manifest)
    rows = []
    for index, slot in enumerate(slots):
        if diagnostic and index:
            rows.append({**asdict(slot), "status": "未运行"})
            continue
        run_id = "diagnostic-run" if diagnostic else f"run-{index}"
        survey = f"# Survey for {run_id}\n\nArchived result."
        blocked = index == 0 or index in blocked_replays
        artifact = {
            "run_id": run_id,
            "query": f"Research question {run_id}",
            "status": "completed",
            "report": {
                "survey": survey,
                "delivery": {"status": "blocked" if blocked else "ready"},
            },
        }
        snapshot = f"snapshots/{slot.key}/run.json"
        snapshot_sha = write(root / snapshot, artifact)
        rows.append(
            {
                **asdict(slot),
                "status": "completed",
                "run_id": run_id,
                "snapshot": snapshot,
                "snapshot_sha256": snapshot_sha,
                "report_sha256": sha(survey.encode()),
                "delivery": artifact["report"]["delivery"]["status"],
                "evaluation": {"faithfulness": {"score": 0.97}},
                "external_inputs": f"sha256:{sha(slot.case_id.encode())}",
            }
        )
    digest = write(
        root / "benchmark.json",
        {
            "contract": CONTRACT,
            "status": "partial" if diagnostic else "completed",
            "rows": rows,
        },
    )
    return rows, digest


def fixture_retry(
    baseline, retries, *, fault=None, index=3, blocked_replays=(3,), checkpoint=False
):
    rows, _ = fixture_campaign(baseline, blocked_replays=blocked_replays)
    slot = run_slots()[index]
    original = rows[index]
    if fault == "original_delivery":
        snapshot = baseline / original["snapshot"]
        archived = json.loads(snapshot.read_bytes())
        archived["report"]["delivery"]["status"] = "ready"
        original["snapshot_sha256"] = write(snapshot, archived)
        benchmark = baseline / "benchmark.json"
        write(benchmark, {**json.loads(benchmark.read_bytes()), "rows": rows})
    shutil.copytree(baseline, retries)
    origin = {key: original[key] for key in ("run_id", "snapshot_sha256")}
    if fault == "origin":
        origin["snapshot_sha256"] = rows[0]["snapshot_sha256"]
    manifest = json.loads((retries / "manifest.json").read_bytes())
    del manifest["approval_sha256"]
    scope = {"timing_scope": SUMMARY_CHECKPOINT_SCOPE} if checkpoint else {}
    write_manifest(
        retries,
        {
            **manifest,
            **scope,
            "retry_replay_slots": [slot.key],
            "retry_of": {slot.key: origin},
        },
    )
    run_id = (
        original["run_id"] if fault == "identity" else f"{slot.key}-retry-0123456789ab"
    )
    survey = "# Survey after retry\n\nRevised result."
    fingerprint = original["external_inputs"] if fault != "inputs" else "different"
    events = []
    if fault != "receipt":
        events.append(
            {
                "event": "external_inputs_ready",
                "at": "2026-11-25T00:00:00+00:00",
                "data": {"fingerprint": fingerprint},
            }
        )
    if checkpoint and fault != "checkpoint_receipt":
        events.extend(
            (
                {
                    "event": "benchmark.summary_checkpoint",
                    "at": "2026-11-25T00:00:02+00:00",
                    "data": {
                        "source_run_id": origin["run_id"],
                        "source_snapshot_sha256": origin["snapshot_sha256"],
                        "reused_task_ids": ["dedup", "extract", "relevance_gate"],
                    },
                },
                {
                    "event": "survey.complete",
                    "at": "2026-11-25T00:00:07.400+00:00",
                },
            )
        )
    artifact = {
        "run_id": run_id,
        "query": "Original research question",
        "status": "completed",
        "report": {
            "survey": survey,
            "delivery": {"status": "blocked" if fault == "delivery" else "ready"},
        },
        "events": events,
    }
    retry = {
        **original,
        "run_id": run_id,
        "retry_of": origin,
        "delivery": "ready",
        "report_sha256": sha(survey.encode()),
        "snapshot_sha256": write(retries / original["snapshot"], artifact),
        "external_inputs": fingerprint,
        **scope,
    }
    if checkpoint:
        retry["downstream_ms"] = 7_400 if fault == "checkpoint_window" else 5_400
    if fault == "checkpoint_scope":
        del retry["timing_scope"]
    if fault == "unexecuted":
        retry = {**asdict(slot), "status": "未运行"}
    rows[index] = retry
    if fault == "carried":
        rows[0] = {**asdict(run_slots()[0]), "status": "未运行"}
    write(
        retries / "benchmark.json",
        {
            "contract": CONTRACT,
            "status": "partial" if fault == "unexecuted" else "completed",
            "rows": rows,
        },
    )
    return original, retry


def revalidation(root, rows, digest):
    path = root / "revalidation.json"
    write(
        path,
        {
            "contract": "survey-benchmark-matrix-delivery-revalidation-v1",
            "mode": "offline_revalidation_of_real_surveys",
            "dataset": {"sha256": digest},
            "policy": BENCHMARK_FAITHFULNESS_POLICY,
            "threshold": 0.95,
            "additional_model_requests": 0,
            "recorded_ready": len(rows) - 1,
            "revalidated_ready": len(rows),
            "total": len(rows),
            "rows": [
                {
                    **{
                        key: row[key]
                        for key in (
                            "run_id",
                            "case_id",
                            "profile_id",
                            "repetition",
                            "report_sha256",
                        )
                    },
                    "run": {"sha256": row["snapshot_sha256"]},
                    "recorded_status": row["delivery"],
                    "revalidated_status": "ready",
                    "faithfulness": 0.97,
                    "gates": {"evaluation": "passed"},
                }
                for row in rows
            ],
        },
    )
    return path


def test_benchmark_links_open_exact_archived_workbench_runs(tmp_path):
    baseline = tmp_path / "baseline"
    diagnostics = tmp_path / "diagnostics"
    rows, digest = fixture_campaign(baseline)
    diagnostic_rows, _ = fixture_campaign(diagnostics, diagnostic=True)
    receipt = revalidation(tmp_path, rows, digest)
    app = viewer_app(baseline, diagnostics, receipt)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        benchmark = client.get("/benchmark")
        assert benchmark.status_code == 200
        assert "21/21" in benchmark.text and "诊断复验" in benchmark.text
        assert "/snapshots/" not in benchmark.text
        header, body = benchmark.text.split("</thead>", 1)
        assert "终稿引用召回" not in header
        assert (
            '<th scope="col">内部一致性</th><th scope="col">耗时（秒）</th>' in header
        )
        assert (
            body.split("</tr>", 1)[0].count("<td>")
            == header.count('<th scope="col">') - 1
        )
        runs = rows + diagnostic_rows[:1]
        assert {r["run_id"] for r in client.get("/flow-demo/runs").json()} == {
            r["run_id"] for r in runs
        }
        for row in runs:
            run_id = row["run_id"]
            href = "/flow-demo?" + urlencode(
                {"run": f"archive:{run_id}", "view": "execution"}
            )
            assert href.replace("&", "&amp;") in benchmark.text
            workbench = client.get(href)
            assert (
                workbench.status_code == 200
                and "LitAgent Research Workbench" in workbench.text
            )
            response = client.get(f"/flow-demo/runs/{run_id}/view")
            assert response.status_code == 200
            view = response.json()
            assert view["source"] == "archive" and view["summary"]["run_id"] == run_id
            assert view["rendered_report"]["source_sha256"] == row["report_sha256"]
            assert view["capabilities"] == {
                "can_download_artifact": False,
                "can_export": False,
            }
        assert client.get("/static/flow-demo/app.js").status_code == 200
        context = client.get("/workbench/context").json()
        assert context["live_enabled"] is False
        assert context["benchmark_overview"] == "/benchmark"
        assert client.head("/benchmark").status_code == 200
        assert (
            client.post("/survey", json={"query": "unwanted dispatch"}).status_code
            == 405
        )
        assert client.get("/flow-demo/runs/unrelated/view").status_code == 404
        assert client.get("/flow-demo/runs/run-0/download").status_code == 404
        assert client.get("/static/flow-demo/samples.json").status_code == 404
        assert client.get("/static/presentation/suite.json").status_code == 404
        assert (
            client.get("/benchmark", headers={"Host": "other.test"}).status_code == 400
        )
        with (baseline / rows[0]["snapshot"]).open("ab") as file:
            file.write(b"\n")
        assert client.get("/flow-demo/runs/run-0/view").status_code == 409


def test_viewer_rejects_mismatched_snapshot_and_offline_result(tmp_path):
    rows, digest = fixture_campaign(tmp_path)
    receipt = revalidation(tmp_path, rows, digest)
    bad = json.loads(receipt.read_bytes())
    bad["rows"][0]["report_sha256"] = "other-report"
    write(receipt, bad)
    with pytest.raises(ValueError, match="different run or result"):
        viewer_app(tmp_path, revalidation_path=receipt)
    with (tmp_path / rows[0]["snapshot"]).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(ValueError, match="snapshot SHA-256 mismatch"):
        viewer_app(tmp_path)


def test_retry_opens_as_a_separate_attempt_beside_the_original(tmp_path):
    baseline, retries = tmp_path / "baseline", tmp_path / "retry"
    original, retry = fixture_retry(baseline, retries)
    app = viewer_app(baseline, retries_path=retries)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        benchmark = client.get("/benchmark")
        assert "执行 21/21 · 原始交付 19/21" in benchmark.text
        assert "计入真实复跑交付 20/21 · 修复复跑 1 次" in benchmark.text
        ids = [item["run_id"] for item in client.get("/flow-demo/runs").json()]
        assert len(ids) == len(set(ids)) == 22
        links = [
            (
                "/flow-demo?"
                + urlencode({"run": f"archive:{row['run_id']}", "view": "execution"})
            ).replace("&", "&amp;")
            for row in (original, retry)
        ]
        first, second = (benchmark.text.index(href) for href in links)
        assert first < second and benchmark.text[first:second].count("</tr>") == 1
        for row in (original, retry):
            view = client.get(f"/flow-demo/runs/{row['run_id']}/view").json()
            assert view["summary"]["run_id"] == row["run_id"]
            assert view["rendered_report"]["source_sha256"] == row["report_sha256"]


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("origin", "origin does not match"),
        ("original_delivery", "origin does not match"),
        ("identity", "identity or external inputs do not match"),
        ("inputs", "identity or external inputs do not match"),
        ("delivery", "delivery does not match"),
        ("receipt", "did not record the frozen external inputs"),
        ("carried", "changed a carried baseline run"),
    ],
)
def test_retry_viewer_rejects_inauthentic_comparisons(tmp_path, fault, message):
    baseline, retries = tmp_path / "baseline", tmp_path / "retry"
    fixture_retry(baseline, retries, fault=fault)
    with pytest.raises(ValueError, match=message):
        viewer_app(baseline, retries_path=retries)


def test_viewer_does_not_list_an_unexecuted_retry(tmp_path):
    baseline, retries = tmp_path / "baseline", tmp_path / "retry"
    fixture_retry(baseline, retries, fault="unexecuted")
    with TestClient(
        viewer_app(baseline, retries_path=retries), base_url="http://127.0.0.1"
    ) as client:
        assert len(client.get("/flow-demo/runs").json()) == 21
        assert "修复复跑 1 次" not in client.get("/benchmark").text


def test_viewer_keeps_summary_continuations_adjacent_with_separate_timing(tmp_path):
    baseline, replay, checkpoint = (tmp_path / p for p in ("base", "replay", "summary"))
    original_replay, replay_attempt = fixture_retry(
        baseline, replay, blocked_replays=(3, 4)
    )
    original_checkpoint, checkpoint_attempt = fixture_retry(
        baseline, checkpoint, index=4, blocked_replays=(3, 4), checkpoint=True
    )
    with TestClient(
        viewer_app(baseline, retries_path=[replay, checkpoint]),
        base_url="http://127.0.0.1",
    ) as client:
        page = client.get("/benchmark").text
        assert "原始交付 18/21" in page
        assert "计入真实复跑交付 19/21 · 修复复跑 1 次" in page
        assert "计入摘要阶段续跑交付 20/21 · 摘要阶段续跑 1 次" in page
        assert "摘要修订开始至 Survey 完成，单独展示" in page
        assert "<td>5.4</td>" in page and "<td>7.4</td>" not in page
        for original, retry in (
            (original_replay, replay_attempt),
            (original_checkpoint, checkpoint_attempt),
        ):
            first, second = (
                page.index(f"archive%3A{row['run_id']}&amp;view=execution")
                for row in (original, retry)
            )
            assert first < second and page[first:second].count("</tr>") == 1
        assert len(client.get("/flow-demo/runs").json()) == 23
    baseline_rows = [original_replay, original_checkpoint]
    checkpoint_row = {**checkpoint_attempt, "source": "retry"}
    rows = [*baseline_rows, checkpoint_row]
    assert summary_rows(rows) == summary_rows(baseline_rows)
    compact = render_table(rows, summary_rows(rows), "completed")
    assert "摘要阶段续跑 · external-abstract" in compact and "| 5.4 |" in compact
    assert "终稿引用召回" not in compact
    assert "| 内部一致性 | 耗时（秒） | Tokens |" in compact


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("checkpoint_scope", "identity or external inputs"),
        ("checkpoint_receipt", "summary checkpoint provenance or timing invalid"),
        ("checkpoint_window", "summary checkpoint provenance or timing invalid"),
    ],
)
def test_viewer_rejects_unproven_summary_continuation(tmp_path, fault, message):
    baseline, checkpoint = tmp_path / "baseline", tmp_path / "checkpoint"
    fixture_retry(baseline, checkpoint, fault=fault, checkpoint=True)
    with pytest.raises(ValueError, match=message):
        viewer_app(baseline, retries_path=checkpoint)


def test_viewer_rejects_duplicate_retry_campaigns(tmp_path):
    baseline, retries = tmp_path / "baseline", tmp_path / "retries"
    fixture_retry(baseline, retries)
    with pytest.raises(ValueError, match="retry run identities overlap"):
        viewer_app(baseline, retries_path=[retries, retries])
