"""Lock the source-unit Judge protocol before adding the Jev sidecar."""

import json
from unittest.mock import AsyncMock

import pytest
from test_phase14_5 import _case

from litagent.benchmark.survey_judge import SurveyBenchmarkJudge
from litagent.benchmark.survey_models import SurveyJudgeConfig
from litagent.llm.client import LLMResponse
from litagent.report_render import build_report_source_units


def _response(payload):
    return LLMResponse(
        content=json.dumps(payload),
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def test_report_source_units_round_trip_repeated_table_cells_and_unicode():
    report = (
        "# 标题\n\n"
        "同一个结论 [E:ev-1]。\n\n"
        "- 列表结论 [E:ev-2]\n\n"
        "| 方法 | 方法 | 结果 |\n"
        "| --- | --- | --- |\n"
        "| A | A | `x|y` 与转义 \\| [E:ev-3] |\n"
    )

    first = build_report_source_units(report)
    second = build_report_source_units(report)

    assert first == second
    assert all(report[item.start : item.end] == item.text for item in first)
    assert [item.text for item in first].count("方法") == 2
    assert any("同一个结论" in item.text for item in first)
    assert any("列表结论" in item.text for item in first)
    assert any("`x|y` 与转义 \\|" in item.text for item in first)


@pytest.mark.asyncio
async def test_judge_inventory_uses_host_source_units_and_freezes_snapshot():
    survey = "Claim [E:long-evidence-id]."
    unit = build_report_source_units(survey)[0]
    llm = AsyncMock()
    llm.chat.side_effect = [
        _response(
            {
                "claims": [
                    {
                        "claim_text": "Claim.",
                        "source_unit_id": unit.unit_id,
                        "evidence_aliases": ["e1"],
                    }
                ]
            }
        ),
        _response(
            {
                "topic_assessments": [{"topic_id": "metric-space", "covered": True}],
                "claim_assessments": [
                    {
                        "claim_id": "claim_0",
                        "source_faithful": True,
                        "supported": True,
                        "evidence_ids": ["e1"],
                        "explanation": "The source and evidence agree.",
                    }
                ],
                "contradictions": [],
            }
        ),
    ]

    metrics, usage = await SurveyBenchmarkJudge(
        llm,
        SurveyJudgeConfig(base_url="https://judge.example", model="judge"),
    ).evaluate(
        case=_case(),
        survey=survey,
        ledger={"long-evidence-id": {"text": "Claim."}},
    )

    assert metrics.status == "completed"
    assert metrics.inventory_snapshot is not None
    anchored = metrics.inventory_snapshot.claims[0]
    assert anchored.source_quote == survey
    assert survey[anchored.source_start : anchored.source_end] == survey
    assert anchored.evidence_ids == ("long-evidence-id",)
    assert metrics.claim_assessments[0].claim_id == "claim_0"
    assert metrics.claim_assessments[0].source_faithful is True
    assert usage.call_count == 2
    inventory_message = llm.chat.await_args_list[0].args[0][1]["content"]
    assert unit.unit_id in inventory_message
    assert "source_quote" not in inventory_message


@pytest.mark.asyncio
async def test_source_unfaithful_claim_fails_without_shrinking_inventory():
    survey = "Claim [E:long-evidence-id]."
    unit = build_report_source_units(survey)[0]
    llm = AsyncMock()
    llm.chat.side_effect = [
        _response(
            {
                "claims": [
                    {
                        "claim_text": "Different claim.",
                        "source_unit_id": unit.unit_id,
                        "evidence_aliases": ["e1"],
                    }
                ]
            }
        ),
        _response(
            {
                "topic_assessments": [{"topic_id": "metric-space", "covered": True}],
                "claim_assessments": [
                    {
                        "claim_id": "claim_0",
                        "source_faithful": False,
                        "supported": False,
                        "evidence_ids": ["e1"],
                        "explanation": "The extraction changes the report meaning.",
                    }
                ],
                "contradictions": [],
            }
        ),
    ]

    metrics, _ = await SurveyBenchmarkJudge(
        llm,
        SurveyJudgeConfig(base_url="https://judge.example", model="judge"),
    ).evaluate(
        case=_case(),
        survey=survey,
        ledger={"long-evidence-id": {"text": "Claim."}},
    )

    assert metrics.status == "failed"
    assert metrics.reason_codes == ["judge_inventory_source_unfaithful"]
    assert metrics.factual_claim_count == 1
    assert len(metrics.claim_assessments) == 1
    assert metrics.faithfulness is None
