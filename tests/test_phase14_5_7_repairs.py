"""Regression tests for pilot ranking, identity, audit, and cost repairs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from litagent.config import load_config
from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import EvidenceSelector, SectionSpec
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.rag.models import PaperCandidate, normalize_paper_identity
from litagent.runner import Infra, LitAgent


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("huggingface:1703.05175v2", "arxiv:1703.05175"),
        ("arxiv:2010.11929v1", "arxiv:2010.11929"),
        ("other:1703.05175", "other:1703.05175"),
        ("huggingface:abc", "huggingface:abc"),
    ],
)
def test_verified_identity_aliases_only(raw, expected):
    """Only supported namespaces and arXiv-shaped identifiers alias."""
    assert normalize_paper_identity(raw) == expected


def test_external_candidate_and_chunk_share_identity():
    """New HF candidates use the canonical identity before chunk construction."""
    result = PaperCandidate.from_external(
        {
            "source": "huggingface",
            "paper_id": "1703.05175",
            "title": "ProtoNet",
            "abstract": "Evidence.",
        }
    )
    assert result.paper_id == "arxiv:1703.05175"
    assert result.chunks[0].paper_id == result.paper_id
    assert result.source == "huggingface"


@pytest.mark.asyncio
async def test_selector_refills_after_paper_cap_without_early_topk():
    """Candidates after the former top-k cutoff can fill accepted slots."""
    ledger = {
        f"p{i}:claim": {
            "evidence_id": f"p{i}:claim",
            "paper_id": "same" if i < 12 else "other",
            "paper_title": "vision",
            "text": "vision evidence",
        }
        for i in range(13)
    }
    selector = EvidenceSelector(
        None, BudgetManager(), top_k_per_section=2, per_paper_cap=1
    )
    result = await selector.select(
        "vision",
        ledger,
        (SectionSpec("methods", "vision", "vision"),),
        max_tokens=16000,
    )
    assert set(result.selected_items) == {"p0:claim", "p12:claim"}
    assert any(d["reason_code"] == "paper_limit" for d in result.ranking_decisions)


@pytest.mark.asyncio
async def test_ragas_two_steps_preserve_verdicts_and_usage():
    """Use the RAGAS schema through traced chat, preserving its score formula."""
    client = object.__new__(OpenAICompatibleClient)
    client._model = "test"
    client.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                json.dumps({"statements": ["A", "B"]}),
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
            LLMResponse(
                json.dumps(
                    {
                        "statements": [
                            {
                                "statement_id": "claim_0",
                                "reason": "supported",
                                "verdict": 1,
                            },
                            {
                                "statement_id": "claim_1",
                                "reason": "absent",
                                "verdict": 0,
                            },
                        ]
                    }
                ),
                usage={"prompt_tokens": 20, "completion_tokens": 6},
            ),
        ]
    )
    score, metadata = await client.score_ragas_faithfulness(
        user_input="q",
        response="A. B.",
        retrieved_contexts=["A"],
        max_tokens=1000,
        thinking_mode="disabled",
    )
    assert score == 0.5
    assert len(metadata["verdicts"]) == 2
    assert metadata["usage"] == {"prompt_tokens": 30, "completion_tokens": 11}
    assert client.chat.await_count == 2
    assert client.chat.await_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_unmapped_negative_verdict_is_not_lost():
    """A paraphrased negative claim remains actionable without invented references."""
    result = RagasFaithfulnessEvaluator._diagnostics_from_verdicts(
        "The original sentence [E:p1].",
        [{"statement": "A paraphrase", "verdict": 0, "reason": "unsupported"}],
        {"p1"},
    )
    assert result[0]["alignment_status"] == "unmapped"
    assert result[0]["evidence_ids"] == []
    assert result[0]["diagnostic_source"] == "ragas_verdict"


def test_negative_subclaim_does_not_capture_other_sentence_references():
    """A mixed paragraph cannot authorize deletion of unrelated citations."""
    result = RagasFaithfulnessEvaluator._diagnostics_from_verdicts(
        "A is false [E:e1]. B is supported [E:e2].",
        [{"statement": "A is false", "verdict": 0, "reason": "not entailed"}],
        {"e1", "e2"},
    )
    assert result[0]["alignment_status"] == "unmapped"
    assert result[0]["evidence_ids"] == []


@pytest.mark.asyncio
async def test_ragas_rejects_missing_verdict():
    """Partial NLI output cannot inflate the supported fraction."""
    client = object.__new__(OpenAICompatibleClient)
    client._model = "test"
    client.chat = AsyncMock(
        side_effect=[
            LLMResponse('{"statements":["A","B"]}'),
            LLMResponse(
                '{"statements":[{"statement_id":"claim_0","reason":"yes","verdict":1}]}'
            ),
        ]
    )
    with pytest.raises(ValueError, match="verdicts"):
        await client.score_ragas_faithfulness(
            user_input="q", response="A B", retrieved_contexts=["A"], max_tokens=10
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rag_enabled,rag_rerank", [(False, False), (True, False), (True, True)]
)
async def test_product_rankers_do_not_depend_on_rag(rag_enabled, rag_rerank):
    """RAG profile switches neither disable nor duplicate product model loads."""
    cfg = load_config()
    cfg.rag.enabled = rag_enabled
    cfg.rag.reranker_enabled = rag_rerank
    agent = LitAgent(cfg)
    model = object()
    agent._create_reranker_async = AsyncMock(return_value=model)
    infra = Infra()
    await agent._connect_rankers(cfg, infra)
    assert infra.relevance_reranker is model
    assert infra.evidence_reranker is model
    assert infra.reranker is (model if rag_enabled and rag_rerank else None)
    assert agent._create_reranker_async.await_count == 1


@pytest.mark.asyncio
async def test_ranker_failure_and_cancel_are_distinct():
    """Load failures degrade per component; cancellation propagates immediately."""
    cfg = load_config()
    agent = LitAgent(cfg)
    agent._emit = Mock()
    agent._create_reranker_async = AsyncMock(side_effect=RuntimeError("missing model"))
    infra = Infra()
    await agent._connect_rankers(cfg, infra)
    assert infra.evidence_reranker is None
    assert agent._create_reranker_async.await_count == 1
    assert len(agent._emit.call_args_list) == 3
    agent._create_reranker_async = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await agent._connect_rankers(cfg, Infra())


def test_rewrite_reports_exact_unexpected_removed_ids():
    """Reference deletion is only permitted by an explicit target whitelist."""
    agent = LitAgent(load_config())
    original = "Claim [E:e1]. Second [E:e2]."
    candidate = "Claim has been revised. Second remains [E:e2]."
    rejected = agent._validate_rewrite_candidate(
        original=original, candidate=candidate, evidence_ledger={"e1": {}, "e2": {}}
    )
    assert rejected.unexpected_removed_evidence_ids == ("e1",)
    accepted = agent._validate_rewrite_candidate(
        original=original,
        candidate=candidate,
        evidence_ledger={"e1": {}, "e2": {}},
        allowed_removed_refs=["e1"],
    )
    assert accepted.accepted


@pytest.mark.asyncio
async def test_negative_verdicts_drive_diagnostics_without_second_llm():
    """The same NLI decision supplies both the low score and its repair target."""
    cfg = load_config()
    llm = SimpleNamespace(
        score_ragas_faithfulness=AsyncMock(
            return_value=(
                0.0,
                {
                    "verdicts": [
                        {"statement": "Claim", "verdict": 0, "reason": "not entailed"}
                    ]
                },
            )
        ),
        chat=AsyncMock(),
    )
    ev = RagasFaithfulnessEvaluator(cfg, llm=llm)
    item = {"evidence_id": "e1", "text": "Different", "paper_title": "Paper"}
    result = await ev.evaluate(
        "Claim [E:e1].",
        {
            "query": "q",
            "evidence": {"e1": item},
            "referenced_evidence": [item],
            "referenced_evidence_ids": ["e1"],
        },
    )
    assert not result.passed
    assert result.details["unsupported_claims"][0]["evidence_ids"] == ["e1"]
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_judge_cannot_omit_inventory_claims():
    """An incomplete audit cannot inflate Judge faithfulness."""
    from test_phase14_5 import _case

    from litagent.benchmark.survey_judge import SurveyBenchmarkJudge
    from litagent.benchmark.survey_models import SurveyJudgeConfig

    llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                LLMResponse('{"claims":["A","B"]}'),
                LLMResponse(
                    '{"topic_assessments":[{"topic_id":"metric-space","covered":true}],"claim_assessments":[{"claim_id":"claim_0","supported":true,"evidence_ids":["e1"]}],"contradictions":[]}'
                ),
            ]
        )
    )
    result, _ = await SurveyBenchmarkJudge(
        llm,
        SurveyJudgeConfig(base_url="https://example.test", model="test", max_retries=0),
    ).evaluate(case=_case(), survey="A and B [E:e1].", ledger={"e1": {"text": "A"}})
    assert result.status == "failed"


def test_legacy_alias_metrics_and_unknown_ragas_cost():
    """Historical evidence IDs survive while paper IDs and unknown costs are corrected."""
    from test_phase14_5 import _artifact, _case

    from litagent.benchmark.survey_metrics import evaluate_survey_artifact

    artifact = _artifact("Claim [E:e1].")
    artifact["nodes"]["worker:extract"]["output"][0]["evidence_items"][0][
        "paper_id"
    ] = "huggingface:1606.04080"
    artifact["events"][0]["data"] = {
        "content": {"faithfulness": 0.5},
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    result = evaluate_survey_artifact(case=_case(), artifact=artifact)
    assert result.citations.recall == 1
    assert result.citations.unknown_evidence_ids == []
    assert result.cost.missing_usage_calls == 1
    assert not result.cost.usage_complete


@pytest.mark.asyncio
async def test_known_usage_recorded_before_later_scoring_failure():
    """A successful extraction stays counted if the subsequent NLI call fails."""
    import httpx

    client = OpenAICompatibleClient("https://example.test/v1", "test", api_key="test")
    events = []
    client._trace_hook = lambda event, data: events.append((event, data))
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"statements":["A"]}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    client._client.chat.completions.create = AsyncMock(
        side_effect=[response, ValueError("bad response")]
    )
    with pytest.raises(ValueError):
        await client.score_ragas_faithfulness(
            user_input="q", response="A", retrieved_contexts=["A"], max_tokens=100
        )
    complete = [data for event, data in events if event == "llm.complete"]
    assert len(complete) == 1 and complete[0]["total_tokens"] == 15
    assert [data for event, data in events if event == "llm.failed"][0][
        "usage_available"
    ] is False
    await client._client.close()


@pytest.mark.asyncio
async def test_llm_retry_event_uses_next_attempt_number(monkeypatch):
    """LLM retry events use the same next-attempt semantics as workers and tools."""
    import httpx
    from openai import APIConnectionError
    from openai.types.chat import ChatCompletion

    client = OpenAICompatibleClient("https://example.test/v1", "test", api_key="test")
    events = []
    client._trace_hook = lambda event, data: events.append((event, data))
    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    client._client.chat.completions.create = AsyncMock(
        side_effect=[APIConnectionError(request=request), response]
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    result = await client.chat([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    retry = next(data for event, data in events if event == "llm.retry")
    assert retry["attempt"] == 2
    assert retry["max_attempts"] == 3
    assert retry["reason_code"] == "transient_provider_error"
    await client._client.close()


def test_profile_cost_is_unavailable_if_any_case_usage_is_incomplete():
    """A profile cannot appear cheaper by averaging only its measured cases."""
    from test_phase14_5 import _artifact, _case

    from litagent.benchmark.survey_metrics import (
        aggregate_survey_cases,
        evaluate_survey_artifact,
    )
    from litagent.benchmark.survey_models import SurveyCaseObservation

    measured = _artifact()
    measured["events"][0]["data"] = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "usage_available": True,
    }
    known = evaluate_survey_artifact(case=_case(), artifact=measured)
    unknown = evaluate_survey_artifact(case=_case(), artifact=_artifact())
    observations = [
        SurveyCaseObservation(
            run_id=f"run-{i}",
            case_id=f"case-{i}",
            domain="few_shot",
            profile_id="external-only",
            repetition=1,
            status="completed",
            metrics=m,
        )
        for i, m in enumerate([known, unknown])
    ]
    _, summary = aggregate_survey_cases(observations)
    assert summary.total_tokens is None
