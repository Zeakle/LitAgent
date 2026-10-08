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
async def test_joint_method_queries_do_not_require_each_original_paper_to_name_all_methods():
    """A survey of several methods must discover their individual source papers."""
    llm = MockLLMClient(
        [
            json.dumps(
                {
                    "sub_queries": [
                        "ViT DeiT Swin Transformer image classification methods",
                        "ViT DeiT Swin Transformer experimental comparison",
                        "ViT DeiT Swin Transformer limitations",
                    ],
                    "named_methods": ["ViT", "DeiT", "Swin Transformer"],
                }
            )
        ]
    )
    planner = SurveyPlanner(llm=llm, config=PlannerConfig())
    graph = await planner.plan(
        "Compare ViT, DeiT and Swin Transformer for image classification."
    )
    queries = {
        task.input_data["query"]
        for task in graph.tasks.values()
        if task.agent_type == "search"
    }
    assert queries == {"ViT", "DeiT", "Swin Transformer"}
    assert sum(task.agent_type == "search" for task in graph.tasks.values()) == 6


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


LIVE_METHODS = ["Vision Transformer (ViT)", "DeiT", "Swin Transformer"]
LIVE_JOINT_QUERIES = [
    "Vision Transformer (ViT) DeiT Swin Transformer image classification",
    "Vision Transformer (ViT) DeiT Swin Transformer experimental comparison",
    "Vision Transformer (ViT) DeiT Swin Transformer limitations",
]
ORIGINAL_TITLES = [
    {"method": "Vision Transformer (ViT)", "query": '"An Image is Worth 16x16 Words"'},
    {"method": "DeiT", "query": '"Training data-efficient image transformers"'},
    {
        "method": "Swin Transformer",
        "query": '"Swin Transformer Hierarchical Vision Transformer using Shifted Windows"',
    },
]


class RecordedDecompositionLLM(MockLLMClient):
    """Capture bounded planner calls while returning one deterministic JSON object."""

    def __init__(self, payload):
        """Prepare an isolated decomposition response without provider access."""
        super().__init__([json.dumps(payload)])
        self.requests = []

    async def chat(self, messages, **kwargs):
        """Record the actual planner message before using the local fake response."""
        self.requests.append(messages)
        return await super().chat(messages, **kwargs)


async def plan_named_methods(payload, question, monkeypatch, max_queries=3):
    """Run the public planner under a fixed two-provider call cap."""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    llm = RecordedDecompositionLLM(payload)
    graph = await SurveyPlanner(
        llm=llm, config=PlannerConfig(max_sub_queries=max_queries)
    ).plan(question)
    queries = [
        task.input_data["query"]
        for task in graph.tasks.values()
        if task.agent_type == "search" and task.input_data["source"] == "arxiv"
    ]
    assert len(llm.requests) == 1
    assert len(queries) <= max_queries
    assert (
        sum(task.agent_type == "search" for task in graph.tasks.values())
        == len(queries) * 2
    )
    assert sum(task.agent_type == "recall" for task in graph.tasks.values()) == len(
        queries
    )
    return graph, queries, llm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spelling",
    [
        "Vision Transformer（ViT）",
        "Vision Transformer (ViT)",
        "Vision Transformer(ViT)",
        "Vision Transformer （ViT）",
    ],
)
async def test_live_parenthetical_method_names_survive_query_unicode_and_spacing(
    spelling, monkeypatch
):
    """The live Chinese request must retain ViT alongside DeiT and Swin."""
    payload = {"named_methods": LIVE_METHODS, "sub_queries": LIVE_JOINT_QUERIES}
    question = f"综述 {spelling}、DeiT 和 Swin Transformer 在图像分类中的方法和实验。"
    graph, queries, _ = await plan_named_methods(payload, question, monkeypatch)
    assert (
        graph.tasks["adversarial_review"].input_data["requested_methods"]
        == LIVE_METHODS
    )
    assert len(queries) == 3
    assert any("ViT" in query or "Vision Transformer" in query for query in queries)
    assert "DeiT" in queries and "Swin Transformer" in queries


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 2, 3])
async def test_joint_search_prioritizes_original_titles_within_existing_call_cap(
    cap, monkeypatch
):
    """Discover individual original papers without adding planner or search calls."""
    payload = {
        "named_methods": LIVE_METHODS,
        "sub_queries": LIVE_JOINT_QUERIES,
        "original_paper_queries": ORIGINAL_TITLES,
    }
    graph, queries, llm = await plan_named_methods(
        payload,
        "综述 Vision Transformer（ViT）、DeiT 和 Swin Transformer 的实验。",
        monkeypatch,
        max_queries=cap,
    )
    assert queries == [entry["query"] for entry in ORIGINAL_TITLES[:cap]]
    assert (
        graph.tasks["adversarial_review"].input_data["requested_methods"]
        == LIVE_METHODS
    )
    assert "original_paper_queries" in llm.requests[0][0]["content"]


@pytest.mark.asyncio
async def test_invalid_original_queries_cannot_replace_valid_method_fallbacks(
    monkeypatch,
):
    """Unknown method mappings and malformed queries do not create query slots."""
    payload = {
        "named_methods": LIVE_METHODS,
        "sub_queries": LIVE_JOINT_QUERIES,
        "original_paper_queries": [
            ORIGINAL_TITLES[0],
            {"method": "InventedMethod", "query": "UNRELATED_INJECTED_SEARCH"},
            {"method": "DeiT", "query": " "},
            {"method": "Swin Transformer", "query": 42},
            {"method": 123, "query": "NONSTRING_METHOD_SEARCH"},
            "not-a-mapping",
        ],
    }
    _, queries, _ = await plan_named_methods(
        payload,
        "综述 Vision Transformer（ViT）、DeiT 和 Swin Transformer 的实验。",
        monkeypatch,
    )
    assert queries == [ORIGINAL_TITLES[0]["query"], "DeiT", "Swin Transformer"]
    assert all(
        "INJECTED" not in query and "NONSTRING" not in query for query in queries
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_mapping", [None, "not-a-list", {"method": "DeiT", "query": "title"}]
)
async def test_malformed_original_mapping_preserves_independent_searches(
    bad_mapping, monkeypatch
):
    """A bad optional producer field falls back within the existing bounded plan."""
    payload = {
        "named_methods": LIVE_METHODS,
        "sub_queries": LIVE_JOINT_QUERIES,
        "original_paper_queries": bad_mapping,
    }
    graph, queries, _ = await plan_named_methods(
        payload,
        "综述 Vision Transformer（ViT）、DeiT 和 Swin Transformer 的实验。",
        monkeypatch,
    )
    assert (
        graph.tasks["adversarial_review"].input_data["requested_methods"]
        == LIVE_METHODS
    )
    assert len(queries) == 3
    assert "DeiT" in queries and "Swin Transformer" in queries
    first_arxiv = build_arxiv_query(queries[0])
    assert 'all:"(ViT)"' not in first_arxiv
    assert " OR " in first_arxiv


@pytest.mark.asyncio
async def test_existing_focused_queries_are_not_overwritten_by_original_title_mapping(
    monkeypatch,
):
    """Original-paper hints repair joint searches without replacing focused queries."""
    focused = [
        "vision transformer foundations",
        "DeiT data efficiency",
        "Swin hierarchical attention",
    ]
    payload = {
        "named_methods": LIVE_METHODS,
        "sub_queries": focused,
        "original_paper_queries": ORIGINAL_TITLES,
    }
    _, queries, _ = await plan_named_methods(
        payload,
        "综述 Vision Transformer（ViT）、DeiT 和 Swin Transformer 的实验。",
        monkeypatch,
    )
    assert queries == focused


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spelling", ["Vision Transformer（ViT）", "Vision Transformer (ViT)"]
)
async def test_single_parenthetical_method_fallback_does_not_require_literal_parentheses(
    spelling,
    monkeypatch,
):
    """A source paper may use either the expanded name or its abbreviation."""
    method = "Vision Transformer (ViT)"
    payload = {"named_methods": [method], "sub_queries": [method]}
    graph, queries, _ = await plan_named_methods(
        payload, f"综述 {spelling}在图像分类中的实验与局限。", monkeypatch
    )
    assert graph.tasks["adversarial_review"].input_data["requested_methods"] == [method]
    assert len(queries) == 1
    compiled = build_arxiv_query(queries[0])
    assert 'all:"(ViT)"' not in compiled
    assert " OR " in compiled
    assert "Vision Transformer" in compiled and "ViT" in compiled
