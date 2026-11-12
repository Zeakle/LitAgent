"""Keep semantic failures deterministic without granting them scientific support."""

import json
import os
import subprocess
import sys

import pytest
from test_evaluation_dimensions_identity import dimension_envelope
from test_independent_evaluation import global_regions  # noqa: F401
from test_support_passages import selected

from litagent.eval import batched, scoped
from litagent.eval.results import CONTRACT


def evaluate_entities(text, scope):
    """Use full support validation with real host-generated source passages."""
    source = "This method learns a comparison function."
    claim = {"claim_text": text, "evidence_ids": ["a"]}
    ledger = {"a": {"text": source, "supporting_text": source}}
    decision, passages = selected(claim, ledger, {"a": "e1"})
    return scoped.validate_support_basis(
        claim,
        decision,
        ledger,
        {"a": "e1"},
        passages=passages,
        original_only=True,
        support_validation_revision="host-support-receipts-v2",
        evaluation_dimensions_envelope=scope,
    )


@pytest.mark.parametrize(
    "text,first",
    [
        ("SRL and K-tuplet learn a comparison function.", "SRL"),
        ("K-tuplet and SRL learn a comparison function.", "K-tuplet"),
    ],
)
def test_missing_entities_keep_unverified_status_and_source_order(text, first):
    scope = dimension_envelope(evaluation_result_contract=CONTRACT)
    result = evaluate_entities(text, scope)
    assert result["status"] == "unverified"
    assert result["issue"] == "support_entity_unverified"
    assert result["entity"] == first
    assert result["spans"]
    assert isinstance(scoped.ordered_support_entities(text), set)


def test_hash_seed_cannot_change_current_support_diagnostic():
    script = """
import json
from test_support_diagnostic_order import evaluate_entities
from test_evaluation_dimensions_identity import dimension_envelope
from litagent.eval.results import CONTRACT
r=evaluate_entities('SRL and K-tuplet learn a comparison function.', dimension_envelope(evaluation_result_contract=CONTRACT))
print(json.dumps(r,sort_keys=True))
"""
    outputs = []
    for seed in ("0", "1", "2", "3"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "tests", "PYTHONHASHSEED": seed},
        )
        outputs.append(json.loads(result.stdout.splitlines()[-1]))
    assert all(value == outputs[0] for value in outputs)


def test_limitation_projection_retains_model_verdict_but_not_host_approval(
    global_regions,
):
    payload, value = global_regions
    value["limitations"] = {
        "adequate": True,
        "claim_ids": ["claim-one"],
        "explanation": "Controlled model verdict.",
    }
    result = batched.global_review_type(payload).model_validate(value)
    assertions = [
        {
            "claim_id": "claim-one",
            "kind": "scientific",
            "section": "Limitations and Open Problems",
        }
    ]
    projected = batched.limitations_check(result, assertions, set(), independent=True)
    assert projected["model_adequate"] is True
    assert projected["adequate"] is False
    legacy = batched.limitations_check(result, assertions, set())
    assert "model_adequate" not in legacy and legacy["adequate"] is False
