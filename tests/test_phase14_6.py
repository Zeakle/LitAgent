"""Validate Phase 14.6 reproducible demo and delivery contracts."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from litagent.demo import (
    BUNDLED_SAMPLE_PATH,
    load_bundled_samples,
    project_flow_demo_view,
)


def test_bundled_samples_cover_terminal_delivery_states() -> None:
    samples = load_bundled_samples()

    assert set(samples) == {"sample-ready", "sample-blocked", "sample-partial"}
    assert {sample["report"]["delivery"]["status"] for sample in samples.values()} == {
        "ready",
        "blocked",
        "partial",
    }
    for sample in samples.values():
        assert sample["status"] in {"completed", "cancelled"}
        assert sample["provenance"] == "synthetic_offline_fixture"
        assert sample["formal_benchmark_evidence"] is False
        assert all(
            task["status"] not in {"pending", "running"}
            for task in sample["graph"]["tasks"].values()
        )
        assert all(
            node["status"] not in {"pending", "running"}
            for node in sample["nodes"].values()
        )


def test_bundled_samples_do_not_contain_private_or_credential_material() -> None:
    raw = BUNDLED_SAMPLE_PATH.read_text(encoding="utf-8")
    lowered = raw.lower()
    payload = json.loads(raw)

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    assert "openai_api_key" not in lowered
    assert "langfuse_secret_key" not in lowered
    assert "authorization" not in lowered
    assert "d:\\" not in lowered
    assert "c:\\users\\" not in lowered
    assert max(len(value) for value in strings(payload)) < 1000


def test_flow_demo_projection_uses_real_artifact_contract() -> None:
    artifact = load_bundled_samples()["sample-ready"]

    view = project_flow_demo_view(artifact, source="sample")

    assert view["schema_version"] == 1
    assert view["summary"]["delivery_status"] == "ready"
    assert view["summary"]["publishable"] is True
    assert view["dag"]["dependencies"]["extract"] == ["relevance_gate"]
    assert view["report"]["survey"].startswith("# Reproducible Survey")
    assert view["evidence"][0]["evidence_id"] == "E:demo-1"
    assert view["graph"]["tier_counts"] == {"core": 1}
    assert view["evaluation"]["faithfulness"]["score"] == 0.94


def test_flow_demo_projection_accepts_legacy_minimal_archive() -> None:
    view = project_flow_demo_view(
        {
            "version": 1,
            "run_id": "legacy-run",
            "query": "legacy query",
            "status": "completed",
            "events": [],
            "nodes": {},
            "graph": {},
            "report": None,
        },
        source="archive",
    )

    assert view["summary"]["run_id"] == "legacy-run"
    assert view["summary"]["quality_status"] == "unverified"
    assert view["delivery"] == {}
    assert view["evidence"] == []


def test_demo_context_and_sample_view_work_without_external_services(
    monkeypatch,
) -> None:
    from litagent.api import app

    monkeypatch.setenv("LITAGENT_DEMO_MODE", "offline")
    client = TestClient(app)
    context = client.get("/flow-demo/context")
    view = client.get("/flow-demo/runs/sample-ready/view")
    denied = client.post("/flow-demo/runs")

    assert context.status_code == 200
    assert context.json()["mode"] == "offline"
    assert context.json()["live_enabled"] is False
    assert context.json()["default_run_id"] == "sample-ready"
    assert view.status_code == 200
    assert view.json()["source"] == "sample"
    assert denied.status_code == 403
    assert denied.json()["detail"] == "live_demo_disabled"


def test_demo_cli_defaults_to_offline_and_exposes_language_neutral_flags() -> None:
    from litagent.cli import build_parser

    parser = build_parser()
    default = parser.parse_args(["demo"])
    live = parser.parse_args(
        ["demo", "--live", "--no-browser", "--host", "127.0.0.1", "--port", "9000"]
    )

    assert default.command == "demo"
    assert default.live is False
    assert default.no_browser is False
    assert live.live is True
    assert live.no_browser is True
    assert live.host == "127.0.0.1"
    assert live.port == 9000


def test_demo_cli_starts_offline_server_without_external_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from litagent.cli import _cmd_demo

    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "live")

    _cmd_demo(
        Namespace(
            live=False,
            host="127.0.0.1",
            port=8123,
            no_browser=True,
        )
    )

    assert calls == [{"host": "127.0.0.1", "port": 8123, "reload": False}]
    assert __import__("os").environ["LITAGENT_DEMO_MODE"] == "offline"


def test_compose_services_expose_startup_healthchecks() -> None:
    import yaml

    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    for service in ("redis", "qdrant", "postgres"):
        healthcheck = compose["services"][service]["healthcheck"]
        assert healthcheck["test"]
        assert healthcheck["retries"] > 0

    qdrant_check = compose["services"]["qdrant"]["healthcheck"]["test"]
    assert "curl" not in " ".join(qdrant_check)
    assert "wget" not in " ".join(qdrant_check)


def test_package_data_declares_flow_demo_samples() -> None:
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["litagent"]

    assert "static/flow-demo/*" in package_data
    assert BUNDLED_SAMPLE_PATH.is_file()


def test_language_readmes_link_to_each_other() -> None:
    english = Path("README.md").read_text(encoding="utf-8")
    chinese = Path("README.zh-CN.md").read_text(encoding="utf-8")

    assert "](./README.md)" in english
    assert "](./README.zh-CN.md)" in english
    assert "](./README.md)" in chinese
    assert "](./README.zh-CN.md)" in chinese
    assert "inter" + "view" not in english.lower()
    assert "面" + "试" not in chinese


@pytest.mark.parametrize("filename", ["README.md", "README.zh-CN.md"])
def test_readme_relative_links_resolve_to_versioned_files(filename: str) -> None:
    content = Path(filename).read_text(encoding="utf-8")
    targets = re.findall(r"\]\((\./[^)#]+)", content)

    assert targets
    for target in targets:
        assert Path(target.removeprefix("./")).exists(), target


@pytest.mark.parametrize(
    "filename", ["docs/design-truth.md", "docs/system-overview.md"]
)
def test_public_documentation_relative_links_resolve(filename: str) -> None:
    document = Path(filename)
    content = document.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#)([^)]+)\)", content)

    for target in targets:
        relative_path = target.split("#", 1)[0]
        assert (document.parent / relative_path).exists(), target


def test_design_truth_no_longer_marks_completed_rag_work_as_pending() -> None:
    design_truth = Path("docs/design-truth.md").read_text(encoding="utf-8")

    assert "Cross-run Paper Corpus | implemented" in design_truth
    assert "PDF ingestion/provenance | implemented" in design_truth
    assert "RAG benchmark | completed" in design_truth
    assert (
        "15-case Survey benchmark | implementation complete, live acceptance pending"
        in design_truth
    )


@pytest.mark.parametrize("filename", ["README.md", "README.zh-CN.md"])
def test_readme_commands_use_the_public_demo_cli(filename: str) -> None:
    content = Path(filename).read_text(encoding="utf-8")

    assert "litagent demo" in content
    assert "litagent demo --live" in content
    assert "benchmarks/rag/RESULTS.md" in content
    assert "docs/design-truth.md" in content
    assert "architecture/" not in content


def test_sample_json_is_valid_utf8_json() -> None:
    payload = json.loads(BUNDLED_SAMPLE_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert len(payload["samples"]) == 3
