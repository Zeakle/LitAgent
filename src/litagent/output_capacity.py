"""Resolve output reservations consistently without changing historical plans."""

from collections.abc import Mapping

LEGACY_OUTPUT_TOKENS = {
    "body": 20000,
    "summary": 20000,
    "reviewer": 4096,
    "repair": 20000,
    "inventory": 16384,
    "assessment": 16384,
    "global": 16384,
}
MAX_OUTPUT_TOKENS = 32768
EXPANDED_CONTEXT_TOKENS = 196608


def validate_output_tokens(value: Mapping) -> dict[str, int]:
    """Reject incomplete, coerced, or unsupported server output reservations."""
    if (
        not isinstance(value, Mapping)
        or set(value) != set(LEGACY_OUTPUT_TOKENS)
        or any(
            type(value[key]) is not int
            or not minimum <= value[key] <= MAX_OUTPUT_TOKENS
            for key, minimum in LEGACY_OUTPUT_TOKENS.items()
        )
        or len({value[key] for key in ("body", "summary", "repair")}) != 1
        or len({value[key] for key in ("inventory", "assessment", "global")}) != 1
    ):
        raise ValueError("stage_output_capacity_invalid")
    return dict(value)


def configured_output_tokens(config) -> dict[str, int] | None:
    """Read existing server settings; omit the field for legacy identities."""
    if isinstance(config, Mapping):
        generation = config["llm"]["max_tokens"]
        reviewer = config["adversarial"]["review_max_tokens"]
        evaluation = config["eval"]["max_tokens"]
    else:
        generation = config.llm.max_tokens
        reviewer = config.adversarial.review_max_tokens
        evaluation = config.eval.max_tokens
    value = {
        **dict.fromkeys(("body", "summary", "repair"), generation),
        "reviewer": reviewer,
        **dict.fromkeys(("inventory", "assessment", "global"), evaluation),
    }
    if value == LEGACY_OUTPUT_TOKENS:
        return None
    context = (
        config["eval"]["context_tokens"]
        if isinstance(config, Mapping)
        else config.eval.context_tokens
    )
    if type(context) is not int or context < EXPANDED_CONTEXT_TOKENS:
        raise ValueError("stage_output_context_capacity_invalid")
    return validate_output_tokens(value)


def envelope_output_tokens(envelope=None) -> dict[str, int]:
    """Use frozen stage limits, falling back only when the opt-in field is absent."""
    if envelope is None or "output_tokens" not in envelope:
        return dict(LEGACY_OUTPUT_TOKENS)
    return validate_output_tokens(envelope["output_tokens"])


def envelope_context_tokens(envelope=None) -> int:
    """Bind expanded requests to a fixed, conservatively sized model context."""
    if envelope is not None and "output_tokens" in envelope:
        validate_output_tokens(envelope["output_tokens"])
        return EXPANDED_CONTEXT_TOKENS
    return 131072


def envelope_context_limit(envelope=None) -> int:
    """Retain at least ten percent context headroom for both request profiles."""
    return int(envelope_context_tokens(envelope) * 0.9)
