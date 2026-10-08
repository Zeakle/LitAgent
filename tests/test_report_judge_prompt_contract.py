"""Validate sent Judge examples against consumers without paid model calls."""

import copy
import hashlib
import html
import json
from unittest.mock import AsyncMock

import pytest

from litagent.audit_context import SurveyAuditContext
from litagent.benchmark import survey_judge as module
from litagent.benchmark.survey_models import SurveyBenchmarkCase, SurveyJudgeConfig
from litagent.config import AppConfig
from litagent.llm.client import LLMResponse
from litagent.report_audit import ReportAuditService


def prompt_example(prompt):
    """Read the sole complete output example, including unchanged benchmarks."""
    if "```json" in prompt:
        assert prompt.count("```json") == 1
        return json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
    return json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])[0]


def assert_schema_fields(example, schema, root):
    """Require every illustrated object to match all strict consumer fields."""
    if "$ref" in schema:
        schema = root["$defs"][schema["$ref"].rsplit("/", 1)[1]]
    if isinstance(example, dict):
        assert set(example) == set(schema["properties"]) == set(schema["required"])
        for key, value in example.items():
            assert_schema_fields(value, schema["properties"][key], root)
    elif isinstance(example, list):
        for value in example:
            assert_schema_fields(value, schema["items"], root)


@pytest.mark.parametrize(
    "prompt,consumer",
    [
        (module.REPORT_CLAIM_PROMPT, module._ReportClaims),
        (module.REPORT_AUDIT_PROMPT, module._ReportResponse),
        (module.JUDGE_CLAIM_PROMPT, module._JudgeClaims),
        (module.JUDGE_SYSTEM_PROMPT, module._JudgeResponse),
    ],
)
def test_prompt_example_matches_every_consumer_field(prompt, consumer):
    """Catch omitted required fields before a perfect mock can hide them."""
    example = prompt_example(prompt)
    consumer.model_validate(example)
    schema = module._strict_schema(consumer)
    assert_schema_fields(example, schema, schema)


def supplied_data(message, tag):
    """Decode host data from the actual call's escaped trust wrapper."""
    return json.loads(
        html.unescape(message.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0])
    )


class PromptExampleClient:
    """Follow sent examples rather than silently adding absent schema fields."""

    def __init__(self, missing_field=None):
        """Optionally replay a provider's missing-field contract violation."""
        self.messages = []
        self.missing_field = missing_field

    async def chat(self, messages, **kwargs):
        """Use example field shapes and fill only supplied IDs and decisions."""
        self.messages.append(copy.deepcopy(messages))
        result = prompt_example(messages[0]["content"])
        body = messages[1]["content"]
        if len(self.messages) == 1:
            units = supplied_data(body, "source_units")
            scientific = [unit for unit in units if "Claim A" in unit["text"]]
            methods = [unit for unit in units if "This run" in unit["text"]]
            for key, selected, text in [
                ("claims", scientific, "Claim A."),
                ("method_claims", methods, "This run used abstract evidence."),
            ]:
                if key not in result:
                    continue
                template = result[key][0]
                result[key] = [
                    {
                        **template,
                        "claim_text": text,
                        "source_unit_id": unit["source_unit_id"],
                        "evidence_aliases": unit["evidence_aliases"],
                    }
                    for unit in selected
                ]
        else:
            topics = supplied_data(body, "expected_topics")
            template = result["topic_assessments"][0]
            result["topic_assessments"] = [
                {**template, "topic_id": topic["topic_id"]} for topic in topics
            ]
            claims = supplied_data(body, "claim_inventory")
            template = result["claim_assessments"][0]
            result["claim_assessments"] = [
                {
                    **template,
                    "claim_id": claim["claim_id"],
                    "evidence_ids": claim["evidence_ids"],
                    "explanation": "Supported by the supplied evidence.",
                }
                for claim in claims
            ]
            result["contradictions"] = []
            if "method_assessments" in result:
                methods = supplied_data(body, "method_inventory")
                template = result["method_assessments"][0]
                result["method_assessments"] = [
                    {**template, "claim_id": method["claim_id"]} for method in methods
                ]
            if self.missing_field == "classification_valid":
                for claim in result["claim_assessments"]:
                    claim.pop("classification_valid")
            elif self.missing_field == "method_assessments":
                result.pop("method_assessments")
        return LLMResponse(
            content=json.dumps(result),
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="offline",
        )

    async def close(self):
        """Provide the audit service's client ownership interface."""


def judge(client):
    """Exercise the production JSON-object mode that failed in the live run."""
    return module.SurveyBenchmarkJudge(
        client,
        SurveyJudgeConfig(
            base_url="https://offline.example",
            model="offline",
            response_format_mode="json_object",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scientific,methods", [(True, True), (True, False), (False, True)]
)
async def test_actual_report_calls_follow_their_single_complete_examples(
    scientific, methods
):
    """Explicit empty arrays must survive when either inventory is absent."""
    survey = "\n\n".join(
        part
        for part, present in [
            ("Claim A [E:ev-1].", scientific),
            ("This run used abstract evidence.", methods),
        ]
        if present
    )
    client = PromptExampleClient()
    state = {}
    result, usage = await judge(client).evaluate(
        case=SurveyAuditContext(
            query="Question", review_methodology={"content_mode": "abstract"}
        ),
        survey=survey,
        ledger={"ev-1": {"text": "Claim A."}},
        checkpoint=state,
    )
    assert result.status == "completed"
    assert result.topic_coverage is None
    assert result.factual_claim_count == int(scientific)
    assert len(state["method_assessments"]) == int(methods)
    assert usage.call_count == len(client.messages) == 2
    assert client.messages[0][0]["content"] == module.REPORT_CLAIM_PROMPT
    assert client.messages[1][0]["content"] == module.REPORT_AUDIT_PROMPT
    assert "method_assessments: []" in client.messages[1][0]["content"]


@pytest.mark.asyncio
async def test_benchmark_still_sends_and_accepts_original_three_field_contract():
    """Do not impose ordinary-report fields on strict annotated benchmarks."""
    case = SurveyBenchmarkCase(
        case_id="benchmark-1",
        domain="vision_transformer",
        query="Question",
        required_paper_ids=["paper-1"],
        relevant_paper_ids=["paper-1"],
        expected_topics=[
            {
                "topic_id": "topic-1",
                "description": "Claim A",
                "source_paper_ids": ["paper-1"],
            }
        ],
        evidence_sources=[
            {
                "source_id": "source-1",
                "kind": "paper",
                "url": "https://offline.example/paper",
                "accessed_at": "2026-10-02",
            }
        ],
        knowledge_cutoff="2026-10-01",
        allowed_delivery_statuses=["ready", "needs_review"],
        annotation_version="v1",
    )
    client = PromptExampleClient()
    result, usage = await judge(client).evaluate(
        case=case,
        survey="Claim A [E:ev-1].",
        ledger={"ev-1": {"paper_id": "paper-1", "text": "Claim A."}},
    )
    assert result.status == "completed" and result.topic_coverage == 1
    assert usage.call_count == 2
    assert client.messages[0][0]["content"] == module.JUDGE_CLAIM_PROMPT
    assert client.messages[1][0]["content"] == module.JUDGE_SYSTEM_PROMPT
    assert "classification_valid" not in client.messages[1][0]["content"]
    assert "method_assessments" not in client.messages[1][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_field", ["classification_valid", "method_assessments"]
)
async def test_missing_new_fields_still_fail_and_block_jev(
    tmp_path, monkeypatch, missing_field
):
    """Do not repair paid responses by guessing required verification values."""
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    client = PromptExampleClient(missing_field=missing_field)
    jev_factory = AsyncMock()
    service = ReportAuditService(
        AppConfig(agent={}, logging={}, observability={"enabled": False}),
        tmp_path,
        client_factory=lambda *_: client,
        jev_factory=jev_factory,
    )
    survey = "Claim A [E:ev-1]."
    source = {
        "source_run_id": "report-1",
        "source_report_revision": 1,
        "source_report_sha256": hashlib.sha256(survey.encode()).hexdigest(),
        "report": {
            "survey": survey,
            "quality": {"status": "passed"},
            "metadata": {"human_review": {"status": "approved"}},
        },
        "ledger": {"ev-1": {"text": "Claim A."}},
        "context": {"query": "Question"},
        "artifact": {"run_id": "report-1"},
    }
    original = copy.deepcopy(source)
    started = await service.submit(source, include_jev=True)
    await service._tasks[started["audit_id"]]
    result = await service.get("report-1", 1, started["audit_id"])
    assert result["status"] == result["judge"]["status"] == "failed"
    assert result["judge"]["reason_codes"] == ["judge_audit_schema_invalid"]
    errors = result["judge"]["failure_details"][0]["validation_errors"]
    assert any(f"{missing_field}:missing" in error for error in errors)
    assert result["jev"]["status"] == "blocked"
    jev_factory.assert_not_called()
    assert result["judge_usage"]["call_count"] == len(client.messages) == 2
    assert source == original
    await service.shutdown()


@pytest.mark.asyncio
async def test_old_prompt_checkpoint_rejected_before_any_model_call(monkeypatch):
    """A prompt change must never resume a paid checkpoint under new rules."""
    assert module.JUDGE_PROMPT_VERSION == "survey-judge-v9-report-contract"
    client = AsyncMock()
    auditor = judge(client)
    state = {}
    args = {"case": SurveyAuditContext(query="Question"), "survey": "", "ledger": {}}
    with monkeypatch.context() as old:
        old.setattr(module, "judge_prompt_fingerprint", lambda: "sha256:old-v8")
        await auditor.evaluate(**args, checkpoint=state)
    old_state = copy.deepcopy(state)
    with pytest.raises(ValueError, match="checkpoint input mismatch"):
        await auditor.evaluate(**args, checkpoint=state)
    assert state == old_state
    client.chat.assert_not_called()
