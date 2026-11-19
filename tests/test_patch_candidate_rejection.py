"""Exercise conflicting dispositions through real writer and evaluation consumers."""

import copy

import pytest
from test_inline_citations import InlineGateway, inline_config
from test_separate_review_pipeline import execute
from test_writer_v2 import case  # noqa: F401

from litagent.review_policy import validate_responses
from litagent.writing_v2 import WriterError


def test_conflicts_are_complete_and_do_not_mutate_response():
    targets = {"a": {}, "b": {}}
    response = {
        "target_responses": {
            "t1": {"action": "keep", "reason": "source", "reference_ids": ["c1"]},
            "t2": {"action": "replace", "reason": "source", "reference_ids": ["c1"]},
        },
        "patches": [{"target_id": "a", "sentences": []}],
    }
    saved = copy.deepcopy(response)
    with pytest.raises(WriterError) as caught:
        validate_responses(response, targets, ["c1"], diagnose=True)
    assert [d["reason"] for d in caught.value.diagnostics] == [
        "patch_for_non_replace",
        "missing_replace_patch",
    ]
    assert response == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["review", "evaluation"])
async def test_conflicting_patch_retains_draft_and_continues(
    case, tmp_path, monkeypatch, category
):
    class Gateway(InlineGateway):
        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            if "target_responses" in schema.get("properties", {}):
                first = (
                    (not self.global_calls)
                    if category == "review"
                    else (self.global_calls and self.repair_count == 1)
                )
                if first:
                    key = result["patches"][0]["target_id"]
                    result["target_responses"][key]["action"] = "keep"
            return result

    gateway = Gateway(case, reviews=1 if category == "review" else 0)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=inline_config())
    metadata = report["metadata"]
    if category == "review":
        rejection = metadata["editorial_history"][0]["patch_rejection"]
        assert gateway.editorial_edits == 1
        assert rejection["action"] == "evaluate_retained"
    else:
        rejection = metadata["evaluation_history"]["attempts"][0]["patch_rejection"]
        assert rejection["action"] == "retry_evaluation_revision"
        assert gateway.repair_count == 2
    assert gateway.global_calls >= 1
    assert rejection["remaining"] == 2


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "missing_response"])
def test_ambiguous_targets_do_not_get_retry_diagnostics(fault):
    result = {
        "target_responses": {
            "t1": {"action": "keep", "reason": "source", "reference_ids": ["c1"]}
        },
        "patches": [{"target_id": "a"}],
    }
    if fault == "duplicate":
        result["patches"] *= 2
    elif fault == "unknown":
        result["patches"][0]["target_id"] = "other"
    else:
        result["target_responses"] = {}
    with pytest.raises(WriterError) as caught:
        validate_responses(result, {"a": {}}, ["c1"], diagnose=True)
    assert not caught.value.diagnostics


def test_patch_projection_excludes_private_receipts():
    from litagent.review_cycles import patch_rejection_projection

    value = {
        "contract": "bound-patch-rejection-v1",
        "category": "review_revision",
        "remaining": 2,
        "actual_tokens": 10,
        "call": "repair_0",
        "retained_report_sha256": "a" * 64,
        "diagnostics": [
            {
                "target_key": "t1",
                "target_id": "writer:methods:1:1",
                "action": "keep",
                "reason": "patch_for_non_replace",
                "patch_index": 0,
            }
        ],
        "provider_receipt": "private",
    }
    projected = patch_rejection_projection(value)
    assert projected["action"] == "evaluate_retained"
    assert "provider_receipt" not in projected
    assert patch_rejection_projection(projected) == projected
    assert patch_rejection_projection({**value, "remaining": True}) is None
