"""Read-only benchmark archive security and API regressions."""

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from litagent.api import app
from litagent.benchmark.survey_archive import SurveyBenchmarkArchive


def _archive(tmp_path):
    """Build a synthetic archive without depending on private live artifacts."""
    result = {
        "benchmark_type": "survey",
        "run_id": "archive-test",
        "status": "succeeded",
        "stage": "pilot",
        "dataset_id": "fixture",
        "dataset_version": "1",
        "dataset_fingerprint": "fixture",
        "judgment_status": "owner_approved_ai_assisted",
        "annotation_version": "1",
        "manifest_fingerprint": "fixture",
        "git_sha": "fixture",
        "git_dirty": False,
        "generation_model": "fixture",
        "judge_model": "fixture",
        "self_judged": True,
        "formal_eligible": False,
        "prompt_source_fingerprint": "fixture",
        "judge_prompt_fingerprint": "fixture",
        "config_fingerprint": "fixture",
        "profile_fingerprints": {},
        "case_aggregates": [],
        "profile_summaries": [],
        "observations": [
            {
                "run_id": "fixture-observation",
                "case_id": "few-shot-metric-space",
                "domain": "few_shot",
                "profile_id": "external-only",
                "repetition": 1,
                "status": "completed",
            }
        ],
    }
    project = tmp_path / "project"
    root = project / "artifacts/benchmarks/survey"
    runs = project / "artifacts/runs"
    root.mkdir(parents=True)
    runs.mkdir(parents=True)
    result["run_id"] = "archive-test"
    result["observations"] = result["observations"][:1]
    observation = result["observations"][0]
    artifact = {"run_id": observation["run_id"], "report": {"survey": "Verified"}}
    content = json.dumps(artifact).encode("utf-8")
    path = runs / "one.json"
    path.write_bytes(content)
    observation["artifact_path"] = "artifacts/runs/one.json"
    observation["artifact_sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()
    (root / "archive-test.json").write_text(json.dumps(result), encoding="utf-8")
    return SurveyBenchmarkArchive(root, runs), root, runs, observation["run_id"]


def test_benchmark_archive_skips_bad_and_ambiguous_results(tmp_path):
    archive, root, _, _ = _archive(tmp_path)
    (root / "broken.json").write_text("{", encoding="utf-8")
    checkpoint = root / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "ignore.json").write_text("{}", encoding="utf-8")
    assert [item["benchmark_id"] for item in archive.list()] == ["archive-test"]
    duplicate = root / "duplicate.json"
    duplicate.write_bytes((root / "archive-test.json").read_bytes())
    assert archive.list() == []
    try:
        archive.get("archive-test")
        assert False, "duplicate ID must not select an arbitrary result"
    except ValueError as exc:
        assert str(exc) == "duplicate_benchmark_id"


def test_benchmark_observation_validates_root_and_sha(tmp_path):
    archive, root, runs, observation_id = _archive(tmp_path)
    assert archive.observation("archive-test", observation_id)["artifact"][
        "report"
    ] == {"survey": "Verified"}
    (runs / "one.json").write_text("tampered", encoding="utf-8")
    try:
        archive.observation("archive-test", observation_id)
        assert False, "tampered artifact must fail"
    except ValueError as exc:
        assert str(exc) == "observation_artifact_sha_mismatch"
    raw = json.loads((root / "archive-test.json").read_text(encoding="utf-8"))
    raw["observations"][0]["artifact_path"] = "../../outside.json"
    (root / "archive-test.json").write_text(json.dumps(raw), encoding="utf-8")
    try:
        archive.observation("archive-test", observation_id)
        assert False, "path traversal must fail"
    except ValueError as exc:
        assert str(exc) == "observation_artifact_outside_runs_root"


def test_benchmark_api_returns_read_only_result(tmp_path):
    archive, _, _, observation_id = _archive(tmp_path)
    app.state.benchmark_archive = archive
    try:
        client = TestClient(app)
        assert (
            client.get("/workbench/benchmarks").json()[0]["benchmark_id"]
            == "archive-test"
        )
        assert (
            client.get("/workbench/benchmarks/archive-test").json()["stage"] == "pilot"
        )
        response = client.get(
            f"/workbench/benchmarks/archive-test/observations/{observation_id}"
        )
        assert response.status_code == 200
        assert response.json()["artifact"]["report"]["survey"] == "Verified"
        assert response.json()["corrected_observation"] is None
        assert client.get("/workbench/benchmarks/unknown").status_code == 404
    finally:
        del app.state.benchmark_archive


def test_corrected_diagnostic_requires_exact_source_sha(tmp_path, monkeypatch):
    archive, root, _, observation_id = _archive(tmp_path)
    source = root / "archive-test.json"
    diagnostic = {
        "benchmark_type": "survey_corrected",
        "source_run_id": "archive-test",
        "source_sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
        "observations": [{"run_id": observation_id, "metrics": {"coverage": 1}}],
    }
    corrected = root / "corrected"
    corrected.mkdir()
    path = corrected / "archive-test-corrected-v4.json"
    path.write_text(json.dumps(diagnostic), encoding="utf-8")
    assert archive.corrected_diagnostic("archive-test") == diagnostic
    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert archive.corrected_diagnostic("archive-test") is None

    original_read_bytes = Path.read_bytes

    def missing_source(path):
        if path == source:
            raise OSError("source vanished")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", missing_source)
    assert archive.corrected_diagnostic("archive-test") is None


def test_observation_attaches_valid_citation_audits_and_marks_stale_sha(tmp_path):
    archive, root, runs, observation_id = _archive(tmp_path)
    audit_root = root.parent / "citation-audit"
    audit_root.mkdir()
    source_sha = json.loads((root / "archive-test.json").read_text(encoding="utf-8"))[
        "observations"
    ][0]["artifact_sha256"]
    payload = {
        "benchmark_type": "citation_audit",
        "schema_version": 1,
        "protocol_version": "jev-citation-audit-v1-ledger",
        "run_id": "citation-audit-one",
        "status": "completed",
        "source_run_id": observation_id,
        "source_benchmark_id": "archive-test",
        "case_id": "few-shot-metric-space",
        "profile_id": "external-only",
        "source_artifact_sha256": source_sha,
        "inventory_sha256": "sha256:" + "1" * 64,
        "evidence_sha256": "sha256:" + "2" * 64,
        "requested_model": "jev-1.13.0",
        "actual_model": "jev-1.13.0",
        "started_at": "2026-09-22T00:00:00Z",
        "completed_at": "2026-09-22T00:00:01Z",
    }
    path = audit_root / "citation-audit-one.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    linked = SurveyBenchmarkArchive(root, runs, audit_root).observation(
        "archive-test", observation_id
    )["citation_audits"]
    assert linked[0]["association_status"] == "current"

    payload["source_artifact_sha256"] = "sha256:" + "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    stale = SurveyBenchmarkArchive(root, runs, audit_root).observation(
        "archive-test", observation_id
    )["citation_audits"]
    assert stale[0]["association_status"] == "stale_diagnostic"
