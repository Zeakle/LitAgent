"""Provide asynchronous OpenAI-compatible and mock LLM clients."""

from __future__ import annotations

import asyncio
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
        *,
        api_key: str | None = None,
    ):
        """Initialize the OpenAI-compatible client."""
        resolved_api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not resolved_api_key:
            logger.warning("NO LLM API key found (LLM_API_KEY / OPENAI_API_KEY)")
        # Retries are explicit so every HTTP attempt has its own trace and usage.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=resolved_api_key, max_retries=0
        )
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
        """Retry transient provider failures with one lifecycle per attempt."""
        from openai import APIConnectionError, APIStatusError

        for attempt in range(3):
            try:
                return await self._chat_attempt(messages, **kwargs)
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if attempt == 2 or (
                    status is not None
                    and status not in {408, 409, 429}
                    and status < 500
                ):
                    raise
                self._emit(
                    "llm.retry",
                    {
                        "task_id": get_task_id(),
                        # Retry events describe the next provider attempt, matching
                        # the worker/tool lifecycle contract.
                        "attempt": attempt + 2,
                        "max_attempts": 3,
                        "error_type": type(exc).__name__,
                        "reason_code": "transient_provider_error",
                        "backoff_ms": 1000 * 2**attempt,
                    },
                )
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable retry state")

    async def _chat_attempt(self, messages: list[dict], **kwargs) -> LLMResponse:
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

            if "extra_body" in kwargs:
                create_kwargs["extra_body"] = kwargs["extra_body"]

            if "tools" in kwargs and kwargs["tools"]:
                create_kwargs["tools"] = kwargs["tools"]

            resp = await self._client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "usage_available": resp.usage is not None,
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
                    "usage_available": resp.usage is not None,
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
                    "usage_available": False,
                },
            )
            raise

    async def score_ragas_faithfulness(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        max_tokens: int,
        thinking_mode: str = "provider_default",
    ) -> tuple[float, dict]:
        """Run RAGAS through the configured provider client and trace lifecycle."""
        from ragas.metrics.collections.faithfulness.util import (
            StatementGeneratorInput,
            StatementGeneratorOutput,
            StatementGeneratorPrompt,
        )

        from litagent.eval.ragas_contract import build_nli_prompt, resolve_nli_verdicts

        if not user_input or not response or not retrieved_contexts:
            raise ValueError("faithfulness requires query, report and evidence")
        started = time.perf_counter()
        kwargs: dict = {
            "max_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if thinking_mode != "provider_default":
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        generated = await self.chat(
            [
                {
                    "role": "user",
                    "content": StatementGeneratorPrompt().to_string(
                        StatementGeneratorInput(question=user_input, answer=response)
                    ),
                }
            ],
            **kwargs,
        )
        statements = StatementGeneratorOutput.model_validate_json(
            generated.content
        ).statements
        if not statements or any(not s.strip() for s in statements):
            raise ValueError("ragas returned no usable statements")
        judged = await self.chat(
            [
                {
                    "role": "user",
                    "content": build_nli_prompt(statements, retrieved_contexts),
                }
            ],
            **kwargs,
        )
        verdicts = resolve_nli_verdicts(judged.content, statements)
        score = sum(v["verdict"] for v in verdicts) / len(statements)
        return score, {
            "algorithm_version": "ragas-traced-v3-indexed",
            "provider_adapter": "openai_compatible",
            "model": self._model,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "statements": statements,
            "verdicts": verdicts,
            "usage": {
                key: sum(int(r.usage.get(key) or 0) for r in (generated, judged))
                for key in ("prompt_tokens", "completion_tokens")
            },
            "usage_complete": all(
                r.usage.get("usage_available", bool(r.usage))
                for r in (generated, judged)
            ),
        }


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
