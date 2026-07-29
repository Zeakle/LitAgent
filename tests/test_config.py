"""Tests for configuration defaults, loading, and environment overrides."""

import os
from pathlib import Path

import pytest

from litagent.config import (
    AdversarialConfig,
    ContextConfig,
    EvalConfig,
    LLMConfig,
    ObservabilityConfig,
    PlannerConfig,
    load_config,
)
from litagent.exceptions import ConfigError


def test_observability_defaults_to_full_redacted_payloads():
    observability = ObservabilityConfig()

    assert observability.enabled is True
    assert observability.payload_mode == "full_redacted"


def test_default_yaml_enables_observability():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(project_root / "config" / "default.yaml"))

    assert cfg.observability.enabled is True


def test_code_defaults_match_default_yaml_token_caps():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(project_root / "config" / "default.yaml"))

    context = ContextConfig()
    assert cfg.context.max_tokens == context.max_tokens == 32_768
    assert (
        cfg.context.synthesis_evidence_max_tokens
        == (context.synthesis_evidence_max_tokens)
        == 16_000
    )
    assert (
        cfg.context.synthesis_papers_max_tokens
        == (context.synthesis_papers_max_tokens)
        == 8_000
    )
    assert (
        cfg.context.review_evidence_max_tokens
        == (context.review_evidence_max_tokens)
        == 10_000
    )
    assert (
        cfg.context.review_draft_max_tokens
        == (context.review_draft_max_tokens)
        == 14_000
    )
    assert (
        cfg.context.review_feedback_max_tokens
        == (context.review_feedback_max_tokens)
        == 2_000
    )

    assert cfg.llm.max_tokens == LLMConfig().max_tokens == 20_000
    assert (
        cfg.adversarial.review_max_tokens
        == (AdversarialConfig().review_max_tokens)
        == 12_288
    )
    assert cfg.eval.max_tokens == EvalConfig().max_tokens == 16_384
    assert cfg.planner.max_tokens == PlannerConfig().max_tokens == 8_192


def test_context_layer_caps_fit_total_input_budget():
    context = ContextConfig()

    assert (
        context.synthesis_evidence_max_tokens + context.synthesis_papers_max_tokens
        <= context.max_tokens
    )
    assert (
        context.review_evidence_max_tokens
        + context.review_draft_max_tokens
        + context.review_feedback_max_tokens
        <= context.max_tokens
    )


def test_worker_fallback_budgets_follow_context_config():
    from unittest.mock import MagicMock

    from litagent.agents.reviewer import ReviewerWorker
    from litagent.agents.synthesis import SynthesisWorker

    context = ContextConfig(
        max_tokens=20_000,
        compact_threshold=0.85,
        synthesis_evidence_max_tokens=12_000,
        synthesis_papers_max_tokens=6_000,
        review_evidence_max_tokens=6_000,
        review_draft_max_tokens=10_000,
        review_feedback_max_tokens=2_000,
    )
    synthesis = SynthesisWorker(MagicMock(), context_config=context)
    reviewer = ReviewerWorker(MagicMock(), context_config=context)

    assert synthesis._budget.max_tokens == 20_000
    assert reviewer._budget.max_tokens == 20_000
    assert synthesis._budget.needs_compact(16_001) is False
    assert reviewer._budget.needs_compact(16_001) is False


@pytest.fixture
def temp_project(tmp_path):
    """Create a minimal project configuration tree."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "default.yaml"
    config_file.write_text("""
agent:
  max_loops: 15
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'litagent'\n")
    return tmp_path, config_file


class TestLoadConfig:
    """Tests configuration loading and validation."""

    def test_load_default(self, temp_project):
        tmp, cfg_file = temp_project
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp))
            cfg = load_config(str(cfg_file))
            assert cfg.agent.max_loops == 15
            assert cfg.logging.level == "INFO"
        finally:
            os.chdir(old_cwd)

    def test_env_var_override_int(self, temp_project, monkeypatch):
        tmp, cfg_file = temp_project
        monkeypatch.setenv("LITAGENT_AGENT_MAX_LOOPS", "20")
        cfg = load_config(str(cfg_file))
        assert cfg.agent.max_loops == 20
        assert isinstance(cfg.agent.max_loops, int)

    def test_env_var_override_str(self, temp_project, monkeypatch):
        tmp, cfg_file = temp_project
        monkeypatch.setenv("LITAGENT_LOGGING_LEVEL", "DEBUG")
        cfg = load_config(str(cfg_file))
        assert cfg.logging.level == "DEBUG"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("agent: not_a_dict\nlogging:\n  level: INFO\n  format: ''")
        with pytest.raises((ConfigError, TypeError)):
            load_config(str(bad_file))


class TestApplyEnvOverrides:
    """Tests environment-variable overrides."""

    def _run(self, data, env, monkeypatch):
        from litagent.config import _apply_env_overrides

        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return _apply_env_overrides(data)

    def test_bool_override_does_not_crash(self, monkeypatch):
        """Boolean overrides are parsed without type errors."""
        data = {"observability": {"enabled": False}}
        out = self._run(data, {"LITAGENT_OBSERVABILITY_ENABLED": "true"}, monkeypatch)
        assert out["observability"]["enabled"] is True

    def test_bool_falsy_values(self, monkeypatch):
        """Supported false-like values map to ``False``."""
        data = {"observability": {"enabled": True}}
        out = self._run(data, {"LITAGENT_OBSERVABILITY_ENABLED": "false"}, monkeypatch)
        assert out["observability"]["enabled"] is False

    def test_unknown_key_is_skipped(self, monkeypatch):
        """Unknown keys are ignored without mutating valid settings."""
        data = {"agent": {"max_loops": 15}}
        out = self._run(data, {"LITAGENT_AGENT_TYPO": "x"}, monkeypatch)
        assert "typo" not in out["agent"]
        assert out["agent"]["max_loops"] == 15

    def test_int_and_float_conversion(self, monkeypatch):
        data = {"agent": {"max_loops": 15}, "llm": {"temperature": 0.1}}
        out = self._run(
            data,
            {
                "LITAGENT_AGENT_MAX_LOOPS": "20",
                "LITAGENT_LLM_TEMPERATURE": "0.7",
            },
            monkeypatch,
        )
        assert out["agent"]["max_loops"] == 20 and isinstance(
            out["agent"]["max_loops"], int
        )
        assert out["llm"]["temperature"] == 0.7 and isinstance(
            out["llm"]["temperature"], float
        )

    def test_unknown_section_is_skipped(self, monkeypatch):
        """Unknown configuration sections are ignored."""
        data = {"agent": {"max_loops": 15}}
        out = self._run(data, {"LITAGENT_NOSUCH_KEY": "x"}, monkeypatch)
        assert "nosuch" not in out
