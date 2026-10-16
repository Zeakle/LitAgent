"""Reject explicit mechanism and result-context borrowing in strict evaluation."""

from unittest.mock import patch

import pytest
from test_quantitative_attribution import evaluate_claim
from test_support_passages import selected

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.eval.automatic import frozen_judge_fingerprint
from litagent.eval.scoped import validate_support_basis


def check(claim_text, source, *, strict=True):
    """Exercise the real support consumer with a deliberately positive decision."""
    claim = {"claim_id": "c", "claim_text": claim_text, "evidence_ids": ["a"]}
    ledger = {"a": {"text": claim_text, "supporting_text": source}}
    decision, passages = selected(claim, ledger, {"a": "e1"})
    return validate_support_basis(
        claim, decision, ledger, {"a": "e1"}, passages=passages, original_only=strict
    )


MECHANISM_SOURCE = (
    "EViT uses token reorganization. Evo-ViT uses slow-fast token evolution."
)
AVERAGE_SOURCE = (
    "Model-A achieves average accuracy of 80%. Model-A achieves peak accuracy of 90%."
)
SHOT_SOURCE = "Model-A achieves accuracy of 80% under one-shot support. Model-A achieves accuracy of 90% under five-shot support."


@pytest.mark.parametrize("separator", [". ", "; ", "；", ", whereas "])
@pytest.mark.parametrize(
    "claim,first,second,issue",
    [
        (
            "EViT uses slow-fast token evolution.",
            "EViT uses token reorganization",
            "Evo-ViT uses slow-fast token evolution",
            "support_mechanism_attribution_unverified",
        ),
        (
            "Model-A achieves average accuracy of 90%.",
            "Model-A achieves average accuracy of 80%",
            "Model-A achieves peak accuracy of 90%",
            "support_result_context_unverified",
        ),
        (
            "Model-A achieves accuracy of 90% under one-shot support.",
            "Model-A achieves accuracy of 80% under one-shot support",
            "Model-A achieves accuracy of 90% under five-shot support",
            "support_result_context_unverified",
        ),
    ],
)
def test_explicit_other_owner_or_context_cannot_supply_the_claim(
    separator, claim, first, second, issue
):
    result = check(claim, first + separator + second + ".")
    assert result["status"] == "unverified" and result["issue"] == issue
    assert result["spans"]


@pytest.mark.parametrize(
    "claim,source",
    [
        ("EViT uses token reorganization.", MECHANISM_SOURCE),
        ("Evo-ViT uses slow-fast token evolution.", MECHANISM_SOURCE),
        (
            "EViT uses token reorganization.",
            "EViT uses token reorganization. Evo-ViT uses token reorganization.",
        ),
        (
            "EViT uses token reorganization.",
            "EViT is evaluated first. It uses token reorganization.",
        ),
        ("Model-A achieves average accuracy of 80%.", AVERAGE_SOURCE),
        ("Model-A achieves peak accuracy of 90%.", AVERAGE_SOURCE),
        ("Model-A achieves accuracy of 90% under five-shot support.", SHOT_SOURCE),
        (
            "Model-A achieves accuracy of 90% under one-shot support.",
            "Model-A is evaluated under one-shot support. Its accuracy is 90%.",
        ),
        (
            "Model-A achieves average accuracy of 90%.",
            "Model-A achieves average accuracy of 80%. Model-A achieves accuracy of 90% under another condition.",
        ),
    ],
)
def test_correct_or_unspecified_context_is_left_to_semantic_review(claim, source):
    assert check(claim, source)["status"] == "passed"


@pytest.mark.parametrize("value,expected", [("25", "passed"), ("50", "unverified")])
def test_single_owner_statistical_clauses_do_not_exchange_values(value, expected):
    result = check(
        f"EViT achieves an average speed improvement of {value}%.",
        "EViT achieves an average speed improvement of 25% and a peak speed improvement of 50%.",
    )
    assert result["status"] == expected


@pytest.mark.parametrize(
    "claim,source,issue",
    [
        (
            "EViT采用token evolution。",
            "EViT采用token reorganization。Evo-ViT采用token evolution。",
            "support_mechanism_attribution_unverified",
        ),
        (
            "Model-A的平均准确率为90%。",
            "Model-A的平均准确率为80%。Model-A的最高准确率为90%。",
            "support_result_context_unverified",
        ),
        (
            "Model-A在1-shot条件下的准确率为90%。",
            "Model-A在1-shot条件下的准确率为80%。Model-A在5-shot条件下的准确率为90%。",
            "support_result_context_unverified",
        ),
    ],
)
def test_chinese_explicit_bindings_are_preserved(claim, source, issue):
    assert check(claim, source)["issue"] == issue


@pytest.mark.parametrize(
    "claim,source",
    [
        ("EViT uses slow-fast token evolution.", MECHANISM_SOURCE),
        ("Model-A achieves average accuracy of 90%.", AVERAGE_SOURCE),
        ("Model-A achieves accuracy of 90% under one-shot support.", SHOT_SOURCE),
    ],
)
def test_legacy_support_consumer_keeps_its_previous_semantics(claim, source):
    assert check(claim, source, strict=False)["status"] == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim,source",
    [
        ("EViT uses slow-fast token evolution.", MECHANISM_SOURCE),
        ("Model-A achieves average accuracy of 90%.", AVERAGE_SOURCE),
        ("Model-A achieves accuracy of 90% under one-shot support.", SHOT_SOURCE),
    ],
)
async def test_positive_model_decision_cannot_make_strict_report_publishable(
    claim, source
):
    body, _, result = await evaluate_claim(claim, source)
    assert result["score"] != 1.0
    assertion = result["assertions"][0]
    assert assertion["status"] == "unverifiable"
    assert (
        body[assertion["source_start"] : assertion["source_end"]]
        == assertion["source_quote"]
    )


def test_strict_identity_and_frozen_judge_bind_host_support_contract():
    cfg = load_config()
    legacy = build_config_summary(cfg)
    with patch("litagent.eval.scoped.STRICT_BINDING_CONTRACT", "changed-host-contract"):
        changed = build_config_summary(cfg, "strict_auto_v1")
        changed_judge = frozen_judge_fingerprint()
        assert build_config_summary(cfg) == legacy
    assert changed != build_config_summary(cfg, "strict_auto_v1")
    assert changed_judge != frozen_judge_fingerprint()
