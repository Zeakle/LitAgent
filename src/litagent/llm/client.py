"""Provide asynchronous OpenAI-compatible and mock LLM clients."""

from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from litagent.logging import get_logger
from litagent.observability.context import get_task_id
from litagent.safety.budget import CostBudget

logger = get_logger("llm.client")


@dataclass
class LLMResponse:
    """Represent normalized model content, usage, and tool calls."""

    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    # Reasoning providers may require this value when replaying tool-call turns.
    reasoning_content: str = ""


class BaseLLMClient(ABC):
    """Define the asynchronous LLM client interface."""

    @abstractmethod
    async def chat(self, message: list[dict], **kwargs) -> LLMResponse:
        """Generate a normalized response for the supplied messages."""
        ...


class OpenAICompatibleClient(BaseLLMClient):
    """Call OpenAI-compatible providers with tracing and budget accounting."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        cost_budget: CostBudget | None = None,
        trace_hook=None,
    ):
        """Initialize the OpenAI-compatible client."""
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("NO LLM API key found (LLM_API_KEY / OPENAI_API_KEY)")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._cost_budget = cost_budget
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Generate a response and emit its complete I/O lifecycle."""
        model = kwargs.get("model", self._model)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        op_id = uuid.uuid4().hex
        t0 = time.perf_counter()

        self._emit(
            "llm.start",
            {
                "operation_id": op_id,
                "task_id": get_task_id(),
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        try:
            create_kwargs: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if "response_format" in kwargs:
                create_kwargs["response_format"] = kwargs["response_format"]

            if "tools" in kwargs and kwargs["tools"]:
                create_kwargs["tools"] = kwargs["tools"]

            resp = await self._client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            }

            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

            if self._cost_budget:
                self._cost_budget.record(usage)

            response = LLMResponse(
                content=choice.message.content or "",
                model=resp.model,
                usage=usage,
                tool_calls=tool_calls,
                reasoning_content=getattr(choice.message, "reasoning_content", "")
                or "",
            )

            self._emit(
                "llm.complete",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "model": model,
                    "messages": messages,
                    "content": response.content,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("prompt_tokens", 0)
                    + usage.get("completion_tokens", 0),
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "tool_calls": _tool_calls_for_trace(tool_calls),
                },
            )

            return response

        # BaseException keeps cancellation visible in the terminal trace event.
        except BaseException as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._emit(
                "llm.failed",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "model": model,
                    "elapsed_ms": elapsed_ms,
                    "error_type": type(e).__name__,
                    "error": str(e)[:512],
                },
            )
            raise


def _tool_calls_for_trace(tool_calls: list[dict]) -> list[dict]:
    """Copy tool calls and parse JSON arguments for recursive trace redaction."""
    traced: list[dict] = []
    for tool_call in tool_calls:
        function = dict(tool_call.get("function", {}))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        function["arguments"] = arguments
        traced.append(
            {
                "id": tool_call.get("id", ""),
                "type": tool_call.get("type", "function"),
                "function": function,
            }
        )
    return traced


class MockLLMClient(BaseLLMClient):
    """Return deterministic responses for tests and offline workflows."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize the mock LLM client."""
        self._responses = list(responses or ["Mock LLM response"])
        self._call_count = 0

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Return the next configured response."""
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return LLMResponse(content=self._responses[idx], model="mock")
