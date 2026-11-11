"""Keep historical Mini replay fixtures explicit as production defaults evolve."""

from pathlib import Path

from litagent.config import load_config


def load_legacy_mini_config(config_path=None):
    """Recreate the original model and selection settings for historical replay."""
    config = load_previous_default_config(config_path)
    config.safety = config.safety.model_copy(update={"planned_run_limits": None})
    config.context = config.context.model_copy(
        update={"evidence_top_k_per_section": 12}
    )
    config.llm = config.llm.model_copy(
        update={
            "model": "doubao-seed-2-0-mini-260428",
            "extraction_model": None,
            "generation_model": None,
            "evaluation_model": None,
            "request_timeouts_seconds": {},
        }
    )
    config.planner = config.planner.model_copy(update={"thinking_mode": None})
    config.extractor = config.extractor.model_copy(
        update={
            "record_identity": None,
            "task_timeout_ms": None,
            "automatic_timeout_seconds": None,
        }
    )
    config.orchestrator = config.orchestrator.model_copy(
        update={
            "timeout_ms": 900000,
            "task_timeouts_ms": {},
        }
    )
    config.adversarial = config.adversarial.model_copy(
        update={"task_timeout_ms": 720000}
    )
    config.eval = config.eval.model_copy(
        update={"total_timeout_seconds": None, "semantic_engine": None}
    )
    return config


def load_previous_default_config(config_path=None):
    """Select the historical YAML explicitly, including after a test changes cwd."""
    return load_config(
        config_path or str(Path(__file__).resolve().parents[1] / "config/default.yaml")
    )
