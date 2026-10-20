"""Keep historical Mini replay fixtures explicit as production defaults evolve."""

from litagent.config import load_config


def load_legacy_mini_config(config_path=None):
    """Load unchanged non-model settings with the original single-model identity."""
    config = load_config(config_path)
    config.llm = config.llm.model_copy(
        update={
            "model": "doubao-seed-2-0-mini-260428",
            "extraction_model": None,
            "generation_model": None,
            "evaluation_model": None,
        }
    )
    config.planner = config.planner.model_copy(update={"thinking_mode": None})
    config.extractor = config.extractor.model_copy(update={"record_identity": None})
    return config
