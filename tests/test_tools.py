"""Tests for tool definitions, registration, and execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from litagent.tools.base import (
    FallbackStep,
    RateLimitConfig,
    ToolCategory,
    ToolDefinition,
)
from litagent.tools.executor import ToolExecutor
from litagent.tools.registry import ToolRegistry, get_registry, reset_registry


def echo_tool(message: str) -> str:
    return f"Echo: {message}"


echo_definition = ToolDefinition(
    name="echo",
    description="Echo back the input message",
    parameters={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
    category=ToolCategory.READ,
    timeout_ms=5000,
    max_retries=1,
)


class TestToolDefinition:
    """Tests tool-definition serialization."""

    def test_to_llm_format_strips_internal_fields(self):
        td = ToolDefinition(
            name="echo",
            description="Echo back",
            timeout_ms=999,
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        )
        llm = td.to_llm_format()
        assert llm["name"] == "echo"
        assert "parameters" in llm
        assert "timeout_ms" not in llm
        assert "category" not in llm

    def test_defaults(self):
        td = ToolDefinition(name="test", description="test")
        assert td.category == ToolCategory.READ
        assert td.timeout_ms == 30000
        assert td.max_retries == 2
        assert td.cache_ttl_ms == 0


class TestToolRegistry:
    """Tests tool registration and lookup."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_registry()
        yield
        reset_registry()

    def test_register_and_get(self):
        registry = get_registry()
        registry.register(ToolDefinition(name="echo", description="test"), echo_tool)
        assert "echo" in registry
        assert registry.get("echo").definition.name == "echo"

    def test_register_duplicate_overwrites(self):
        registry = get_registry()
        td1 = ToolDefinition(name="echo", description="v1", version="1.0.0")
        td2 = ToolDefinition(name="echo", description="v2", version="2.0.0")
        registry.register(td1, echo_tool)
        registry.register(td2, echo_tool)
        assert registry.get("echo").definition.version == "2.0.0"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            get_registry().get("nonexistent")

    def test_list_all(self):
        registry = get_registry()
        registry.register(ToolDefinition(name="a", description="a"), lambda: None)
        registry.register(ToolDefinition(name="b", description="b"), lambda: None)
        assert len(registry.list_all()) == 2

    def test_to_llm_format(self):
        registry = get_registry()
        registry.register(ToolDefinition(name="echo", description="test"), echo_tool)
        result = registry.to_llm_format()
        assert len(result) == 1
        assert result[0]["name"] == "echo"

    def test_list_by_category(self):
        registry = get_registry()
        registry.register(
            ToolDefinition(name="r", description="r", category=ToolCategory.READ),
            lambda: None,
        )
        registry.register(
            ToolDefinition(name="w", description="w", category=ToolCategory.WRITE),
            lambda: None,
        )
        assert len(registry.list_by_category(ToolCategory.READ)) == 1
        assert len(registry.list_by_category(ToolCategory.WRITE)) == 1


class TestToolExecutor:
    """Tests tool execution, timeout, and caching."""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_registry()
        registry = get_registry()
        registry.register(echo_definition, echo_tool)
        registry.register(
            ToolDefinition(
                name="slow", description="slow", timeout_ms=100, max_retries=0
            ),
            lambda: asyncio.sleep(1),
        )
        self.executor = ToolExecutor(registry, allowed_names={"echo", "slow"})

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        result = await self.executor.execute("echo", {"message": "hello"})
        assert result.error is None
        assert result.output == "Echo: hello"

    @pytest.mark.asyncio
    async def test_timeout(self):
        result = await self.executor.execute("slow", {})
        assert result.error is not None
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        result = await self.executor.execute("nonexistent", {})
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        registry = get_registry()
        registry.register(
            ToolDefinition(name="cached", description="c", cache_ttl_ms=60000),
            lambda x: f"result: {x}",
        )
        executor = ToolExecutor(registry, allowed_names={"cached"})
        r1 = await executor.execute("cached", {"x": "a"})
        assert r1.from_cache is False
        r2 = await executor.execute("cached", {"x": "a"})
        assert r2.from_cache is True
        assert r2.output == r1.output

    @pytest.mark.asyncio
    async def test_executor_rejects_tool_outside_allowlist_before_callable(self):
        called = False
        registry = ToolRegistry()

        async def forbidden() -> None:
            nonlocal called
            called = True

        registry.register(
            ToolDefinition(name="forbidden", description="forbidden"), forbidden
        )
        executor = ToolExecutor(registry, allowed_names=set())

        result = await executor.execute("forbidden", {})

        assert result.error_code == "tool_not_allowed"
        assert result.policy_stage == "allowlist"
        assert called is False
        assert executor._breakers == {}
        assert executor._rate_limits == {}
        assert executor._cache == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("category", [ToolCategory.WRITE, ToolCategory.DESTRUCTIVE])
    async def test_executor_rejects_write_and_destructive_categories(self, category):
        called = False
        registry = ToolRegistry()

        async def mutate() -> None:
            nonlocal called
            called = True

        registry.register(
            ToolDefinition(name="mutate", description="mutate", category=category),
            mutate,
        )
        executor = ToolExecutor(registry, allowed_names={"mutate"})

        result = await executor.execute("mutate", {})

        assert result.error_code == "tool_category_denied"
        assert result.policy_stage == "category"
        assert called is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "args",
        [
            {},
            {"message": 1},
            {"message": "ok", "unexpected": True},
            ["not", "a", "mapping"],
        ],
    )
    async def test_executor_rejects_missing_wrong_and_extra_arguments(self, args):
        registry = ToolRegistry()
        strict_definition = echo_definition.model_copy(
            update={
                "parameters": {
                    **echo_definition.parameters,
                    "additionalProperties": False,
                }
            }
        )
        registry.register(strict_definition, echo_tool)
        executor = ToolExecutor(registry, allowed_names={"echo"})

        result = await executor.execute("echo", args)

        assert result.error_code == "tool_arguments_invalid"
        assert result.policy_stage == "schema"

    @pytest.mark.asyncio
    async def test_executor_accepts_valid_nested_json_schema(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="nested",
                description="nested",
                parameters={
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["request"],
                    "additionalProperties": False,
                },
            ),
            lambda request: request["query"],
        )
        executor = ToolExecutor(registry, allowed_names={"nested"})

        result = await executor.execute("nested", {"request": {"query": "rag"}})

        assert result.error is None
        assert result.output == "rag"

    @pytest.mark.asyncio
    async def test_invalid_tool_schema_fails_closed(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="invalid_schema",
                description="invalid",
                parameters={"type": "definitely-not-a-json-schema-type"},
            ),
            lambda: "must not run",
        )
        executor = ToolExecutor(registry, allowed_names={"invalid_schema"})

        result = await executor.execute("invalid_schema", {})

        assert result.error_code == "tool_schema_invalid"
        assert result.policy_stage == "schema"
        assert executor._breakers == {}

    @pytest.mark.asyncio
    async def test_non_json_arguments_fail_before_runtime_policy(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="generic", description="generic"), lambda **_: "run"
        )
        executor = ToolExecutor(registry, allowed_names={"generic"})

        result = await executor.execute("generic", {"value": object()})

        assert result.error_code == "tool_arguments_invalid"
        assert executor._breakers == {}

    @pytest.mark.asyncio
    async def test_policy_rejection_does_not_touch_fallback(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="primary",
                description="primary",
                fallback=[FallbackStep(type="default_value", default_result="unsafe")],
                cache_ttl_ms=60_000,
                rate_limit=RateLimitConfig(max_calls=1),
            ),
            lambda: "primary",
        )
        executor = ToolExecutor(registry, allowed_names=set())

        result = await executor.execute("primary", {})

        assert result.error_code == "tool_not_allowed"
        assert result.from_fallback is False
        assert executor._breakers == {}
        assert executor._rate_limits == {}
        assert executor._cache == {}

    @pytest.mark.asyncio
    async def test_alternative_tool_fallback_cannot_bypass_policy(self):
        called = False
        registry = ToolRegistry()

        async def primary() -> None:
            raise RuntimeError("primary failed")

        async def alternative() -> str:
            nonlocal called
            called = True
            return "unsafe"

        registry.register(
            ToolDefinition(
                name="primary",
                description="primary",
                max_retries=0,
                fallback=[
                    FallbackStep(
                        type="alternative_tool", alternative_tool="alternative"
                    )
                ],
            ),
            primary,
        )
        registry.register(
            ToolDefinition(name="alternative", description="alternative"), alternative
        )
        executor = ToolExecutor(registry, allowed_names={"primary"})

        result = await executor.execute("primary", {})

        assert result.error_code == "tool_not_allowed"
        assert result.fallback_tool == "alternative"
        assert called is False


class TestBuiltinSearch:
    """Tests built-in paper search parsing."""

    @pytest.mark.asyncio
    async def test_search_arxiv_parses_response_without_network(self, monkeypatch):
        from litagent.tools.builtin import search

        response = MagicMock()
        response.text = """<?xml version='1.0'?>
        <feed xmlns='http://www.w3.org/2005/Atom'>
          <entry>
            <id>http://arxiv.org/abs/2401.01234</id>
            <title>Few-shot Learning</title>
            <summary> A test abstract. </summary>
          </entry>
        </feed>"""
        response.raise_for_status = MagicMock()
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        class FakeAsyncClient:
            """HTTP client that returns a fixed response."""

            async def __aenter__(self):
                return client

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        monkeypatch.setattr(search.httpx, "AsyncClient", lambda **_: FakeAsyncClient())

        result = await search.search_arxiv("few-shot learning", max_results=1)
        assert result == [
            {
                "paper_id": "2401.01234",
                "title": "Few-shot Learning",
                "abstract": "A test abstract.",
                "source": "arxiv",
            }
        ]
        client.get.assert_awaited_once()
