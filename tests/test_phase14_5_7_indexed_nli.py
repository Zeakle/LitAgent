"""Verify stable entailment identity without weakening inventory validation."""

import json

import pytest

from litagent.eval.ragas_contract import build_nli_prompt, resolve_nli_verdicts


def verdict(identity, value=1):
    """Create one provider-shaped indexed judgment."""
    return {
        "statement_id": identity,
        "reason": "Evidence entails it.",
        "verdict": value,
    }


def test_nli_reordering_preserves_original_claim_text():
    """Restore the local statement instead of asking the model to reproduce it."""
    original = "NeRF-VAE enables distribution learning over scenes."
    result = resolve_nli_verdicts(
        json.dumps({"statements": [verdict("claim_1", 0), verdict("claim_0")]}),
        [original, "Unsupported claim."],
    )
    assert [r["statement"] for r in result] == [original, "Unsupported claim."]
    assert [r["verdict"] for r in result] == [1, 0]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [verdict("claim_0")],
        [verdict("claim_0"), verdict("claim_0")],
        [verdict("claim_0"), verdict("claim_2")],
        [verdict("claim_0"), verdict("claim_1", True)],
        [verdict("claim_0"), verdict("claim_1", "1")],
        [verdict("claim_0"), verdict("claim_1", 2)],
        [verdict("claim_0"), {**verdict("claim_1"), "statement": "rewritten"}],
    ],
)
def test_nli_rejects_incomplete_or_invalid_identity(rows):
    """Missing, duplicate, unknown, or loosely typed decisions fail closed."""
    with pytest.raises(ValueError):
        resolve_nli_verdicts(json.dumps({"statements": rows}), ["A", "B"])


def test_original_ragas_example_has_identical_support_fraction():
    """The transport adaptation preserves RAGAS's fixed example and formula."""
    from ragas.metrics.collections.faithfulness.util import NLIStatementPrompt

    prompt = NLIStatementPrompt()
    source, expected = prompt.examples[0]
    rendered = build_nli_prompt(source.statements, [source.context])
    assert rendered.startswith(prompt.instruction)
    payload = json.loads(rendered[len(prompt.instruction) :])
    converted = payload["examples"][0]["output"]
    result = resolve_nli_verdicts(json.dumps(converted), source.statements)
    assert [r["statement"] for r in result] == source.statements
    assert [r["reason"] for r in result] == [r.reason for r in expected.statements]
    assert sum(r["verdict"] for r in result) / len(result) == 0.25
    assert payload["input"]["statements"][0]["statement_id"] == "claim_0"
