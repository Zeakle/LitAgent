"""Exercise the public replay without private repositories, executors or credentials."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from litagent.cli import build_parser
from litagent.presentation import (
    PUBLIC_ROOT,
    PresentationBenchmarkArchive,
    PresentationRepository,
)


def test_presentation_cli_is_explicit_and_live_is_mutually_exclusive():
    assert build_parser().parse_args(["demo", "--presentation"]).presentation
    assert not build_parser().parse_args(["demo"]).presentation
    with pytest.raises(SystemExit):
        build_parser().parse_args(["demo", "--presentation", "--live"])


def test_packaged_result_retains_denominators_and_four_export_consistency():
    archive = PresentationBenchmarkArchive()
    result = archive.get(archive.list()[0]["benchmark_id"])
    assert len(result.observations) == 1
    assert result.counts["formal"]["run"] == 0
    assert result.counts["formal"]["planned"] == 15
    folder = PUBLIC_ROOT / "presentation" / result.run_id
    public = json.loads((folder / "benchmark.json").read_text())
    rows = list(csv.DictReader(io.StringIO((folder / "benchmark.csv").read_text())))
    assert len(public["observations"]) == 1
    assert len(rows) == len(result.planned) == 18
    assert sum(row["status"] != "not_run" for row in rows) == 1
    assert public["observations"][0]["evaluation_execution"] == "not_run"
    assert "not_run" in (folder / "benchmark.html").read_text()
    assert result.scientific_accuracy is None
    assert result.observations[0]["acceptance"]["publishable"] is False
    payload = archive.observation(result.run_id, result.observations[0]["run_id"])
    assert payload["artifact"]["provenance"] == "sanitized_real_diagnostic_replay"
    assert payload["artifact"]["nodes"] == {}
    with pytest.raises(KeyError):
        archive.observation(result.run_id, "../../private")
    assert PresentationRepository().get("../../private") is None


def test_publication_has_no_private_payloads_paths_or_secrets():
    for file in PUBLIC_ROOT.rglob("*"):
        if not file.is_file():
            continue
        raw = file.read_text()
        for forbidden in (
            "/Users/",
            "D:\\\\",
            "sk-lf-",
            "pk-lf-",
            '"Authorization"',
            '"messages"',
            '"source_quote"',
            '"supporting_text"',
        ):
            assert forbidden not in raw, (file.name, forbidden)
    manifest = json.loads((PUBLIC_ROOT / "manifest.json").read_text())
    assert len(manifest["source_archive_sha256"]) == 64
    assert (
        manifest["interpretation"] == "sanitized_diagnostic_not_scientific_acceptance"
    )


def test_actual_import_lifespan_exports_and_routes_never_open_private_state(tmp_path):
    """Fresh-process ASGI proves import-time isolation as well as runtime isolation."""
    code = r"""
import os
os.environ["LITAGENT_DEMO_MODE"] = "presentation"
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("LLM_API_KEY", None)
from unittest.mock import patch
from contextlib import ExitStack
from fastapi.testclient import TestClient
with ExitStack() as stack:
    original_import = __import__
    def no_inference_import(name, *args, **kwargs):
        if name.split(".")[0] in {"sentence_transformers", "transformers", "torch"}:
            raise AssertionError("inference runtime imported during presentation: " + name)
        return original_import(name, *args, **kwargs)
    stack.enter_context(patch("builtins.__import__", side_effect=no_inference_import))
    for target in (
        "litagent.observability.recorder.ArchiveRepository.__init__",
        "litagent.interaction_store.SqliteReviewRepository.__init__",
        "litagent.interactive_service.InteractiveRunService.__init__",
        "litagent.report_audit.ReportAuditService.__init__",
        "litagent.llm.client.OpenAICompatibleClient.__init__",
    ):
        stack.enter_context(patch(target, side_effect=AssertionError(target)))
    from litagent.api import app
    stack.enter_context(patch("litagent.api.load_config", side_effect=AssertionError("private live configuration read")))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        context = client.get("/workbench/context").json()
        assert context["mode"] == "presentation" and not context["live_enabled"]
        assert client.get("/workbench/runs").json() == []
        runs = client.get("/flow-demo/runs").json()
        assert {r["run_id"] for r in runs} >= {"sample-ready", "sample-blocked", "sample-partial"}
        for path in ("/survey", "/flow-demo/runs", "/survey/x/review/decision", "/survey/x/versions/1/audits", "/survey/x/versions/1/audits/y/cancel"):
            assert client.post(path, json={}).status_code == 403, path
        assert client.delete("/survey/x").status_code == 403
        for path in ("/flow-demo/runs/private/view", "/survey/private/view", "/survey/private/versions/1/audits", "/survey/private/export?revision=1&format=md"):
            assert client.get(path).status_code == 404, path
        view = client.get("/flow-demo/runs/sample-ready/view").json()
        assert view["summary"]["publishable"] is True
        assert view["presentation"] is True
        for fmt in ("md", "pdf", "docx"):
            response = client.get(f"/survey/sample-ready/export?revision=1&format={fmt}")
            assert response.status_code == 200, (fmt, response.text[:200] if fmt == "md" else response.status_code)
            assert len(response.headers["x-report-sha256"]) == 64
        for fmt in ("bibtex", "ris", "csl_json"):
            assert client.get(f"/survey/sample-ready/bibliography?revision=1&format={fmt}").status_code == 200
        assert client.get("/survey/sample-ready/export?revision=2&format=md").status_code == 422
        bibliography = client.get("/survey/sample-ready/bibliography?revision=1&format=csl_json").json()
        assert len(bibliography) == 1 and bibliography[0]["title"] == "Metric-based Visual Adaptation"
        assert client.get("/presentation").status_code == 200
        suite = client.get("/workbench/benchmarks").json()[0]["benchmark_id"]
        detail = client.get(f"/workbench/benchmarks/{suite}").json()
        assert detail["observations"][0]["evaluation_execution"] == "not_run"
        run = detail["observations"][0]["run_id"]
        assert client.get(f"/workbench/benchmarks/{suite}/observations/{run}").status_code == 200
        for fmt in ("json", "csv", "md", "html"):
            assert client.get(f"/workbench/benchmarks/{suite}/export/{fmt}").status_code == 200
print("presentation isolated lifecycle and exports passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stderr[-5000:] + result.stdout[-1500:]
    assert "isolated lifecycle" in result.stdout


def test_locale_updates_document_language_in_real_render():
    app = Path("src/litagent/static/flow-demo/app.js").read_text()
    assert "document.documentElement.lang = state.locale" in app
