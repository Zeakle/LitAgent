"""Keep concise discovery queries and explicit provider failures observable."""

import json

import httpx
import pytest

from litagent.agents.planner import SurveyPlanner
from litagent.agents.search import SearchSourceStatus, SearchWorker
from litagent.config import PlannerConfig
from litagent.llm.client import MockLLMClient
from litagent.tools.builtin.search import (
    build_arxiv_query,
    search_huggingface,
    search_semantic_scholar,
)
from litagent.tools.executor import ToolExecutor, ToolResult


@pytest.mark.asyncio
async def test_short_queries_keep_named_methods_without_long_request_slot():
    llm = MockLLMClient(
        [
            json.dumps(
                {
                    "sub_queries": [
                        "vision transformer foundations",
                        "DeiT data efficiency",
                        "Swin hierarchical attention",
                    ],
                    "named_methods": ["DeiT", "Swin", "invented"],
                }
            )
        ]
    )
    planner = SurveyPlanner(llm=llm, config=PlannerConfig())
    question = "Compare foundational vision transformer methods including DeiT and Swin with convolutional networks."
    graph = await planner.plan(question)
    queries = [
        t.input_data["query"] for t in graph.tasks.values() if t.agent_type == "search"
    ]
    assert len(queries) == 6 and question not in queries
    assert (
        "DeiT data efficiency" in queries and "Swin hierarchical attention" in queries
    )
    assert graph.tasks["adversarial_review"].input_data["query"] == question
    assert graph.tasks["adversarial_review"].input_data["requested_methods"] == [
        "DeiT",
        "Swin",
    ]


@pytest.mark.parametrize(
    "query,expected",
    [
        ('"vision transformer" DeiT', 'all:"vision transformer" AND all:"DeiT"'),
        ("DeiT OR Swin", 'all:"DeiT" OR all:"Swin"'),
        (
            'ti:"vision transformer" AND abs:classification',
            'ti:"vision transformer" AND abs:classification',
        ),
    ],
)
def test_arxiv_uses_explicit_fields_phrases_and_boolean_operators(query, expected):
    assert build_arxiv_query(query) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [search_huggingface, search_semantic_scholar])
@pytest.mark.parametrize("status", [400, 429, 503])
async def test_provider_http_failure_is_not_an_empty_search(
    monkeypatch, provider, status
):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return httpx.Response(status, request=httpx.Request("GET", url))

    monkeypatch.setattr(
        "litagent.tools.builtin.search.httpx.AsyncClient", lambda **kwargs: Client()
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await provider("short query")
    failure = ToolExecutor._tool_failure("search", {}, caught.value)
    papers, outcome = SearchWorker._classify("task", "provider", failure, 1)
    assert papers == []
    assert outcome.status == (
        SearchSourceStatus.RATE_LIMITED if status == 429 else SearchSourceStatus.FAILED
    )
    assert outcome.reason_code == (
        "provider_rate_limited" if status == 429 else f"http_status_{status}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [search_huggingface, search_semantic_scholar])
async def test_real_empty_response_and_timeout_are_distinct(monkeypatch, provider):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return httpx.Response(
                200,
                json=[] if provider == search_huggingface else {"data": []},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(
        "litagent.tools.builtin.search.httpx.AsyncClient", lambda **kwargs: Client()
    )
    assert await provider("short query") == []
    _, empty = SearchWorker._classify(
        "task", "provider", ToolResult(name="search", args={}, output=[]), 1
    )
    assert empty.status == SearchSourceStatus.EMPTY
    failure = ToolExecutor._tool_failure("search", {}, httpx.ReadTimeout("timeout"))
    _, timeout = SearchWorker._classify("task", "provider", failure, 1)
    assert timeout.status == SearchSourceStatus.TIMEOUT
