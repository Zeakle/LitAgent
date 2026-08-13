"""Contract tests for the Phase 14.4 offline CI boundary."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_pytest_expression_excludes_integration_and_live() -> None:
    """Default pytest runs must not cross integration or live boundaries."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'addopts = "-m \\"not integration and not live\\""' in pyproject


def test_default_environment_forbids_huggingface_downloads() -> None:
    """Collection hooks put Hugging Face libraries into offline mode."""
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_ci_workflow_has_no_service_or_secret_dependency() -> None:
    """The ordinary CI job must run without service containers or secrets."""
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["test"]
    assert "services" not in job
    rendered = str(workflow)
    assert "secrets." not in rendered
    assert "docker compose" not in rendered


def test_ci_runs_format_and_offline_test_gates() -> None:
    """CI includes formatting, whitespace, and deterministic pytest gates."""
    text = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "black --check src tests" in text
    assert "isort --check-only --profile black src tests" in text
    assert "git diff --check HEAD^" in text
    assert 'pytest -v -m "not integration and not live"' in text


def test_live_marker_is_declared() -> None:
    """Live tests have a dedicated opt-in marker."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"live: requires network access' in pyproject
