"""Check complete v3 batching through the actual evaluator and source contracts."""

import copy
import json

import pytest

from litagent.config import EvalConfig
from litagent.eval.batched import BATCH_PROMPT, plan_batches, request_size
from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
from litagent.llm.client import LLMResponse
from litagent.quality import build_fact_cards, digest


class BatchLLM:
    """Produce controlled inventories and decisions from actual host identities."""

    def __init__(self, *, fail_batch=None, incomplete_global=False):
        self.calls = []
        self.fail_batch = fail_batch
        self.incomplete_global = incomplete_global
        self.batches = 0

    async def chat(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        if "allowed_citation_aliases" in payload:
            result = (
                {
                    "units": {
                        u["source_unit_id"]: {
                            f["source_fragment_id"]: {
                                "assertions": [
                                    {"kind": "scientific", "claim_text": f["text"]}
                                ],
                                "exclusion_reason": "",
                            }
                            for f in u["source_fragments"]
                        }
                        for u in payload["source_units"]
                    }
                }
                if payload["schema"]["properties"]["units"]["type"] == "object"
                else {
                    "units": [
                        {
                            "source_unit_id": u["source_unit_id"],
                            "assertions": [
                                {
                                    "kind": "scientific",
                                    "claim_text": f["text"],
                                    "source_fragment_id": f["source_fragment_id"],
                                    "evidence_aliases": f["citation_aliases"],
                                }
                                for f in u["source_fragments"]
                            ],
                            "exclusion_reason": "",
                        }
                        for u in payload["source_units"]
                    ]
                }
            )
        elif "cited_materials" in payload:
            self.batches += 1
            if self.batches == self.fail_batch:
                raise TimeoutError("controlled timeout")
            decisions = []
            for claim in payload["assertions"]:
                aliases = {e["evidence_alias"] for e in claim["evidence"]}
                support_ids = [
                    pid
                    for pid, p in payload["support_passages"].items()
                    if p[0] in aliases
                    and p[1] != p[0]
                    and p[2] == 0
                    and p[3] == len(payload["original_supports"][p[1]]["text"])
                ]
                decisions.append(
                    {
                        "claim_id": claim["claim_id"],
                        "correct_kind": "scientific",
                        "atomic": True,
                        "classification_valid": True,
                        "source_faithful": True,
                        "status": "supported",
                        "reason_code": "supported",
                        "explanation": "Controlled original source match.",
                        "support_basis": [{"support_id": pid} for pid in support_ids],
                        "aspect_checks": [
                            {
                                "aspect": a,
                                "supported": True,
                                "support_ids": support_ids,
                                "explanation": "Controlled original source match.",
                            }
                            for a in claim["required_support_aspects"]
                        ],
                    }
                )
            result = {
                "units": [
                    {
                        "source_unit_id": u["source_unit_id"],
                        "complete": True,
                        "explanation": "This subset is faithful.",
                    }
                    for u in payload["source_units"]
                ],
                "decisions": decisions,
                "conflicts": [],
            }
        else:
            result = {
                "units": [
                    {
                        "source_unit_id": u["source_unit_id"],
                        "complete": not self.incomplete_global,
                        "explanation": "Full inventory independently checked.",
                    }
                    for u in payload["source_units"]
                ],
                "conflicts": [],
                "issues": [],
                "coverage": [
                    {
                        "requirement_id": r["requirement_id"],
                        "answered": True,
                        "claim_ids": [c["claim_id"] for c in payload["assertions"]],
                        "explanation": "The frozen claims answer this requirement.",
                    }
                    for r in payload["coverage_requirements"]
                ],
            }
        return LLMResponse(
            content=json.dumps(result),
            model="offline",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )


def materials(count=42):
    """Use more than the old forty-item cap without dropping cited papers."""
    ledger = {
        f"a{i}": {
            "evidence_id": f"a{i}",
            "paper_id": f"paper{i}",
            "text": "ViT uses image patches.",
            "supporting_text": "ViT uses image patches.",
            "chunk_key": "abstract",
        }
        for i in range(count)
    }
    facts = build_fact_cards(ledger)
    context = {
        "query": "ViT mechanisms",
        "quality_policy": "strict_v1",
        "evaluation_version": "survey-evaluation-v3",
        "fact_cards": facts,
        "evidence_snapshot_sha256": digest(facts),
        "evidence": ledger,
        "coverage_requirements": [
            {"requirement_id": "question", "kind": "question", "text": "ViT mechanisms"}
        ],
        "structure_checks": {"status": "not_applicable"},
        "method_checks": {"status": "not_applicable"},
    }
    survey = "# Report\n\n## Methods\n\n" + "\n\n".join(
        f"ViT uses image patches. [E:a{i}]" for i in range(count)
    )
    return survey, context


@pytest.mark.asyncio
async def test_full_materials_split_into_three_batches_and_one_global_check():
    survey, context = materials()
    llm = BatchLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    assert result["evaluation_version"] == "survey-evaluation-v3"
    assert len(llm.calls) == 5 and llm.batches == 3
    assert [len(c["assertions"]) for c in llm.calls[1:-1]] == [16, 16, 10]
    assert result["coverage"]["complete"] and result["factual_claim_count"] == 42
    assert result["score"] == 1.0
    assert result["provider"]["call_count"] == 5
    for payload in llm.calls[1:-1]:
        assert len(payload["cited_materials"]) == len(payload["assertions"])
        assert all(len(m["text"]) > 0 for m in payload["original_supports"].values())


@pytest.mark.asyncio
async def test_partial_failure_keeps_batch_progress_without_a_passing_score():
    survey, context = materials()
    checkpoints = []
    context["evaluation_checkpoint"] = lambda p: checkpoints.append(copy.deepcopy(p))
    llm = BatchLLM(fail_batch=2)
    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    details = error.value.details
    assert not details["coverage"]["complete"]
    assert details["stages"]["processed_assertions"] == 16
    assert details["stages"]["batches"][-1]["status"] == "interrupted_no_replay"
    assert (
        len(llm.calls) == 3
        and checkpoints[-1]["batches"][-1]["status"] == "interrupted_no_replay"
    )


@pytest.mark.asyncio
async def test_global_inventory_failure_cannot_be_overridden_by_supported_batches():
    survey, context = materials(2)
    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(
            llm=BatchLLM(incomplete_global=True),
            survey=survey,
            context=context,
            config=EvalConfig(),
        )
    assert not error.value.details["coverage"]["complete"]


@pytest.mark.asyncio
async def test_mutated_card_source_stops_before_any_model_call():
    survey, context = materials(1)
    llm = BatchLLM()
    next(iter(context["fact_cards"].values()))["source_quote"] = "forged source"
    with pytest.raises(ScopedEvaluationError):
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert not llm.calls


@pytest.mark.asyncio
async def test_capacity_overflow_is_not_silently_truncated():
    survey, context = materials(129)
    llm = BatchLLM()
    # Isolate the assertion-count gate from the complete-request context gate.
    config = EvalConfig(faithfulness_max_evidence_items=150, context_tokens=262144)
    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(llm=llm, survey=survey, context=context, config=config)
    assert not error.value.details["coverage"]["complete"]
    assert len(llm.calls) == 1


def test_single_oversized_source_is_preserved_and_request_is_blocked():
    payload = {
        "assertions": [
            {
                "claim_id": "c1",
                "source_unit_id": "u1",
                "evidence": [{"evidence_alias": "e1"}],
            }
        ],
        "source_units": [{"source_unit_id": "u1", "text": "x"}],
        "report_structure": [],
        "cited_materials": {"e1": {"original_support_ref": "s1"}},
        "original_supports": {"s1": {"text": "x" * 130000}},
        "support_passages": {},
    }
    with pytest.raises(ValueError, match="context_budget_exceeded"):
        plan_batches(payload, EvalConfig())
    assert len(payload["original_supports"]["s1"]["text"]) == 130000


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_identity", [False, True])
async def test_global_diagnostic_is_located_by_host_and_invalid_identity_is_rejected(
    invalid_identity,
):
    survey, context = materials(1)

    class GlobalProblem(BatchLLM):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "coverage_requirements" in payload and "cited_materials" not in payload:
                data = json.loads(response.content)
                data["issues"] = [
                    {
                        "source_unit_id": (
                            "forged"
                            if invalid_identity
                            else payload["source_units"][0]["source_unit_id"]
                        ),
                        "explanation": "Four categories but five listed.",
                    }
                ]
                response.content = json.dumps(data)
            return response

    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(
            llm=GlobalProblem(), survey=survey, context=context, config=EvalConfig()
        )
    if invalid_identity:
        assert str(error.value) == "evaluation_global_issue_identity_invalid"
        assert error.value.details["stages"]["global_check"] == "interrupted_no_replay"
        assert not error.value.details.get("global_issues")
        return
    issue = error.value.details["unsupported_claims"][0]
    assert survey[issue["source_start"] : issue["source_end"]] == issue["source_quote"]
    assert issue["evidence_refs"] == ["a0"] and issue["section"] == "Methods"
    assert not error.value.details["coverage"]["complete"]


@pytest.mark.asyncio
async def test_cancellation_preserves_issued_batch_and_never_retries():
    import asyncio

    survey, context = materials(2)
    snapshots = []
    context["evaluation_checkpoint"] = lambda p: snapshots.append(copy.deepcopy(p))

    class CancelledBatch(BatchLLM):
        async def chat(self, messages, **kwargs):
            if "cited_materials" in json.loads(messages[1]["content"]):
                raise asyncio.CancelledError()
            return await super().chat(messages, **kwargs)

    llm = CancelledBatch()
    with pytest.raises(asyncio.CancelledError):
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert len(llm.calls) == 1
    assert snapshots[-1]["batches"][0]["status"] == "interrupted_no_replay"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim,source",
    [
        (
            "Evo-ViT improves speed by 50% on DeiT-S.",
            "EViT improves speed by 50% on DeiT-S.",
        ),
        ("DeiT uses soft inductive biases.", "ConViT uses soft inductive biases."),
        (
            "BSNet improves few-shot classification through two similarity measures.",
            "BiT improves few-shot classification through training and fine-tuning.",
        ),
        ("EViT improves speed by 50%.", "EViT reduces FLOPs by 50%."),
        (
            "EViT improves accuracy by 10 percentage points.",
            "EViT improves accuracy by 10%.",
        ),
        ("EViT improves accuracy on CIFAR-10.", "EViT improves accuracy on ImageNet."),
        ("EViT improves accuracy by 20%.", "EViT improves accuracy by 10%."),
    ],
)
async def test_supported_model_label_cannot_override_wrong_attribution_or_conditions(
    claim, source
):
    survey, context = materials(1)
    context["evidence"]["a0"]["text"] = source
    context["evidence"]["a0"]["supporting_text"] = source
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    survey = "# Report\n\n## Results\n\n" + claim + " [E:a0]"
    result = await evaluate_scoped(
        llm=BatchLLM(), survey=survey, context=context, config=EvalConfig()
    )
    assert result["score"] != 1.0
    assert result["assertions"][0]["status"] != "supported"
