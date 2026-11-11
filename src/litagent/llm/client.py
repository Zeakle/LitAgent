"""Provide asynchronous OpenAI-compatible and mock LLM clients."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from urllib.parse import urlsplit

from openai import APIStatusError, APITimeoutError, AsyncOpenAI

from litagent.logging import get_logger
from litagent.observability.context import get_task_id, remaining_deadline_seconds
from litagent.safety.budget import CostBudget, strict_request_reservation

logger = get_logger("llm.client")
_active_timeout: ContextVar[float | None] = ContextVar("request_timeout", default=None)
_active_request_stage: ContextVar[str | None] = ContextVar(
    "request_stage", default=None
)
_active_request_deadline: ContextVar[asyncio.Timeout | None] = ContextVar(
    "request_deadline", default=None
)


class _ArkPacer:
    """Space request starts across all Ark clients in this process."""

    def __init__(self, interval=5.0, cooldown=60.0):
        self._interval = interval
        self._cooldown = cooldown
        self._lock = Lock()
        self._next_start = 0.0

    async def wait_turn(self, check_deadline):
        """Wait outside the lock so a new cooldown can extend queued waits."""
        while True:
            with self._lock:
                now = time.monotonic()
                delay = max(0.0, self._next_start - now)
                check_deadline(delay)
                if not delay:
                    self._next_start = now + self._interval
                    return
            await asyncio.sleep(delay)

    def defer_after_429(self):
        """Delay future requests without retrying the rejected request."""
        with self._lock:
            self._next_start = max(self._next_start, time.monotonic() + self._cooldown)


_ark_pacer = _ArkPacer()


def _provider_rejection(response):
    """Extract only bounded diagnostics from a provider's HTTP rejection."""
    if response.status_code < 400:
        return {}
    fields = {"provider_http_status": response.status_code}
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
    except ValueError:
        code = None
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,95}", code):
        fields["provider_error_code"] = code
    request_id = response.headers.get("x-request-id")
    if isinstance(request_id, str) and re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}", request_id
    ):
        fields["provider_request_id"] = request_id
    return fields


def _output_limit(kwargs: dict, default: int) -> tuple[str, int]:
    """Resolve one wire limit before reserving or sending a provider request."""
    if "max_output_tokens" in kwargs:
        value = kwargs["max_output_tokens"]
        if "max_tokens" in kwargs or "max_completion_tokens" in kwargs:
            raise ValueError("output_limit_fields_conflict")
        if type(value) is not int or value <= 0:
            raise ValueError("output_limit_invalid")
        return "max_output_tokens", value
    if "max_completion_tokens" in kwargs:
        value = kwargs["max_completion_tokens"]
        if "max_tokens" in kwargs:
            raise ValueError("output_limit_fields_conflict")
        if type(value) is not int or value <= 0:
            raise ValueError("output_limit_invalid")
        return "max_completion_tokens", value
    # Preserve legacy answer-only requests and defaults unchanged.
    return "max_tokens", kwargs.get("max_tokens", default)


class RequestDeadlineExceeded(TimeoutError):
    """Reject a request locally when its complete time reservation cannot fit."""


class LLMRequestTimeout(TimeoutError):
    """Identify an expired host request deadline without changing its allowance."""

    reason_code = "llm_request_timeout"

    def __init__(self, stage: str, seconds: float):
        """Retain host-owned timeout provenance, never scientific request text."""
        super().__init__(f"LLM request timed out after {seconds:g}s ({stage})")
        self.diagnostics = [
            {
                "timeout_scope": "request",
                "request_stage": stage,
                "request_timeout_seconds": seconds,
                "response_received": False,
            }
        ]


async def _await_request(call, *, timeout, stage):
    """Share one owned deadline between the caller and the actual SDK attempt."""
    existing = _active_request_deadline.get()
    if (
        existing is not None
        and existing.when() is not None
        and existing.when() <= asyncio.get_running_loop().time() + timeout
    ):
        return await call
    deadline = asyncio.timeout(timeout)
    tokens = (
        _active_timeout.set(timeout),
        _active_request_stage.set(stage),
        _active_request_deadline.set(deadline),
    )
    try:
        async with deadline:
            return await call
    except TimeoutError as exc:
        # An inner provider TimeoutError must retain its original provenance.
        if deadline.expired():
            raise LLMRequestTimeout(stage, timeout) from exc
        raise
    finally:
        _active_request_deadline.reset(tokens[2])
        _active_request_stage.reset(tokens[1])
        _active_timeout.reset(tokens[0])


def _failure_timeout_metadata(exc):
    """Classify cancellation at the point where its owning deadline is visible."""
    result = {"request_stage": _active_request_stage.get() or _timeout_stage()}
    timeout = _active_timeout.get()
    if timeout is not None:
        result["request_timeout_seconds"] = timeout
    deadline = _active_request_deadline.get()
    if deadline is not None and deadline.expired():
        result.update(reason_code="llm_request_timeout", timeout_scope="request")
    elif isinstance(exc, asyncio.CancelledError):
        remaining = remaining_deadline_seconds()
        upstream = remaining is not None and remaining <= 0
        result.update(
            reason_code="upstream_deadline" if upstream else "cancelled",
            timeout_scope="upstream" if upstream else "external",
        )
    elif isinstance(exc, APITimeoutError):
        result.update(
            reason_code="provider_transport_timeout", timeout_scope="transport"
        )
    return result


def request_timeout_seconds(llm, stage: str, default: float = 180) -> float:
    """Resolve trusted stage settings without depending on mock-client attributes."""
    timeouts = getattr(llm, "_request_timeouts", None)
    if not isinstance(timeouts, dict):
        return default
    return timeouts.get(stage, timeouts.get("default", default))


async def chat_with_timeout(llm, messages, *, stage: str, default=180, **kwargs):
    """Share one timeout between caller and owned transport, preserving mock inputs."""
    timeout = request_timeout_seconds(llm, stage, default)
    return await _await_request(
        llm.chat(messages, **kwargs), timeout=timeout, stage=stage
    )


def _timeout_stage() -> str:
    """Classify unwrapped requests using the host-owned task identifier."""
    task = get_task_id()
    for prefix, stage in (
        ("query_decomposition", "planning"),
        ("extract", "extraction"),
        ("evaluation", "evaluation"),
        ("evidence_rewrite", "repair"),
        ("reviewer", "reviewer"),
        ("human_revision", "revision"),
        ("synthesis", "synthesis"),
        ("adversarial_review", "revision"),
    ):
        if task.startswith(prefix):
            return stage
    return "default"


@dataclass
class LLMResponse:
    """Represent normalized model content, usage, and tool calls."""

    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    # Reasoning providers may require this value when replaying tool-call turns.
    reasoning_content: str = ""
    finish_reason: str | None = None
    request_id: str | None = None


@dataclass
class ResponsesLLMResponse(LLMResponse):
    """Carry private native receipts without changing legacy response fields."""

    provider_receipt: dict = field(default_factory=dict)
    provider_error: str | None = None
    provider_status: str | None = None


class ProviderResponseError(ValueError):
    """Expose a received private response without changing its billing state."""

    def __init__(self, code, response):
        """Keep the receipt available to the durable caller on identity failures."""
        super().__init__(code)
        self.response = response


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
        transport_max_attempts: int = 3,
        request_timeouts: dict[str, float] | None = None,
        deadline_margin_seconds: float = 5,
    ):
        """Initialize the OpenAI-compatible client."""
        resolved_api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not resolved_api_key:
            logger.warning("NO LLM API key found (LLM_API_KEY / OPENAI_API_KEY)")
        if not 1 <= transport_max_attempts <= 3:
            raise ValueError("transport_max_attempts must be between 1 and 3")
        self._transport_max_attempts = transport_max_attempts
        self._request_timeouts = dict(request_timeouts or {})
        if not set(self._request_timeouts).issubset(
            {
                "default",
                "planning",
                "extraction",
                "synthesis",
                "summary",
                "reviewer",
                "revision",
                "repair",
                "evaluation",
            }
        ):
            raise ValueError("invalid_request_timeout_policy")
        from litagent.operational_limits import (
            PROFILE,
            budget_resource_profile,
            increase,
        )

        timeout_ceiling = (
            increase(600) if budget_resource_profile(cost_budget) == PROFILE else 600
        )
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 < v <= timeout_ceiling
                for v in self._request_timeouts.values()
            )
            or isinstance(deadline_margin_seconds, bool)
            or not isinstance(deadline_margin_seconds, (int, float))
            or not math.isfinite(deadline_margin_seconds)
            or not 0 < deadline_margin_seconds <= 60
        ):
            raise ValueError("invalid_request_timeout_policy")
        self._deadline_margin_seconds = deadline_margin_seconds
        self._pacer = (
            _ark_pacer
            if urlsplit(base_url).hostname == "ark.cn-beijing.volces.com"
            else None
        )

        # Retries are explicit so every HTTP attempt has its own trace and usage.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=resolved_api_key, max_retries=0
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._cost_budget = cost_budget
        self._trace_hook = trace_hook

    async def close(self) -> None:
        """Release the owned HTTP client within its event loop."""
        await self._client.close()

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    def _check_request_deadline(self, timeout: float, wait_seconds: float = 0) -> None:
        """Refuse a new transport before reserving tokens or emitting a start."""
        if self._request_timeouts:
            remaining = remaining_deadline_seconds()
            if (
                remaining is not None
                and remaining < wait_seconds + timeout + self._deadline_margin_seconds
            ):
                self._emit(
                    "llm.not_sent",
                    {
                        "task_id": get_task_id(),
                        "reason_code": "request_deadline_insufficient",
                        "remaining_seconds": remaining,
                        "request_timeout_seconds": timeout,
                        "margin_seconds": self._deadline_margin_seconds,
                    },
                )
                raise RequestDeadlineExceeded("request_deadline_insufficient")

    async def _wait_provider_turn(self, timeout):
        """Take a send slot before opening or reserving the provider attempt."""
        if self._pacer is not None:
            await self._pacer.wait_turn(
                lambda delay: self._check_request_deadline(timeout, delay)
            )

    def _observe_provider_rejection(self, response):
        """Defer future Ark calls and project the independent HTTP outcome."""
        if self._pacer is not None and response.status_code == 429:
            self._pacer.defer_after_429()
        return _provider_rejection(response)

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Retry transient provider failures with one lifecycle per attempt."""
        from openai import APIConnectionError

        provider_api = kwargs.get("provider_api", "chat")
        if provider_api not in ("chat", "responses"):
            raise ValueError("provider_api_invalid")
        limit_parameter, output_limit = _output_limit(kwargs, self._max_tokens)
        if provider_api == "responses":
            from litagent.llm.responses import build_responses_request

            if "transport_max_attempts" in kwargs and (
                type(kwargs["transport_max_attempts"]) is not int
                or kwargs["transport_max_attempts"] != 1
            ):
                raise ValueError("responses_retries_unsupported")
            if limit_parameter != "max_output_tokens":
                raise ValueError("responses_output_limit_required")
            if set(kwargs) - {
                "provider_api",
                "model",
                "max_output_tokens",
                "temperature",
                "response_format",
                "extra_body",
                "reasoning_effort",
                "transport_max_attempts",
            }:
                raise ValueError("responses_arguments_unsupported")
            build_responses_request(
                messages,
                model=kwargs.get("model", self._model),
                output_limit=output_limit,
                response_format=kwargs.get("response_format"),
                temperature=kwargs.get("temperature", self._temperature),
                extra_body=kwargs.get("extra_body"),
                reasoning_effort=kwargs.get("reasoning_effort"),
            )
        elif limit_parameter == "max_output_tokens":
            raise ValueError("chat_output_limit_invalid")
        attempts = kwargs.pop("transport_max_attempts", self._transport_max_attempts)
        timeout = _active_timeout.get() or request_timeout_seconds(
            self, _timeout_stage()
        )
        self._check_request_deadline(timeout)
        if provider_api == "responses":
            attempts = 1
        strict = self._cost_budget is not None and self._cost_budget.strict
        if strict:
            attempts = 1
            reservation = strict_request_reservation(
                messages,
                output_limit,
                kwargs.get("response_format"),
            )
            from litagent.safety.planned_budget import (
                locked_writer_envelope,
                planned_backend,
            )

            planned = planned_backend(self._cost_budget)
            if locked_writer_envelope(self._cost_budget) is not None:
                planned.check_wire(
                    messages, output_limit, kwargs.get("response_format")
                )
            from litagent.operational_limits import (
                PROFILE,
                budget_resource_profile,
                increase,
            )

            context_limit = int(131072 * 0.9)
            if budget_resource_profile(self._cost_budget) == PROFILE:
                context_limit = (
                    planned.plan["context_reservation_limit"]
                    if planned is not None
                    else int(increase(131072) * 0.9)
                )
            if reservation > context_limit:
                raise ValueError("strict_context_budget_exceeded")
            # Scoped evaluation and repair already own their exact reservation.
            if planned is not None:
                planned.check_wire(
                    messages, output_limit, kwargs.get("response_format")
                )
            caller_owns = planned is None and (
                (get_task_id() or "").startswith("evaluation")
                or get_task_id() == "evidence_rewrite"
            )
            await self._wait_provider_turn(timeout)
            if not caller_owns:
                self._cost_budget.reserve(reservation)
            try:
                if planned is not None:
                    planned.bind_wire([messages, kwargs, provider_api])
                self._cost_budget.begin_request()
                try:
                    result = await _await_request(
                        self._chat_attempt(messages, **kwargs),
                        timeout=timeout,
                        stage=_active_request_stage.get() or _timeout_stage(),
                    )
                except BaseException:
                    # Transport may have consumed tokens even when no response arrived.
                    self._cost_budget.charge_unknown_usage(reservation)
                    raise
                if not result.usage.get("usage_available"):
                    self._cost_budget.charge_unknown_usage(reservation)
                    if isinstance(result, ResponsesLLMResponse):
                        raise ProviderResponseError("strict_usage_incomplete", result)
                    raise ValueError("strict_usage_incomplete")
                return result
            finally:
                if not caller_owns:
                    self._cost_budget.release(reservation)
                if planned is not None:
                    snapshot = planned.snapshot()
                    self._emit(
                        "budget.updated",
                        {
                            "task_id": get_task_id(),
                            "budget_state": {
                                "plan_identity": snapshot["plan"]["identity"],
                                "known_tokens": snapshot["used_tokens"],
                                "requests": snapshot["requests_sent"],
                                "usage_uncertain": snapshot["usage_uncertain"],
                                "stop_reason": snapshot["stop_reason"],
                                "unknown_hold": sum(
                                    e["reserved_tokens"]
                                    for e in snapshot["entries"].values()
                                    if e["status"] == "interrupted_no_replay"
                                ),
                            },
                        },
                    )
        if (
            not isinstance(attempts, int)
            or not 1 <= attempts <= self._transport_max_attempts
        ):
            raise ValueError("invalid_transport_max_attempts")
        for attempt in range(attempts):
            self._check_request_deadline(timeout)
            await self._wait_provider_turn(timeout)
            try:
                if self._request_timeouts:
                    return await _await_request(
                        self._chat_attempt(messages, **kwargs),
                        timeout=timeout,
                        stage=_active_request_stage.get() or _timeout_stage(),
                    )
                return await self._chat_attempt(messages, **kwargs)
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if attempt == attempts - 1 or (
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
                        "max_attempts": self._transport_max_attempts,
                        "error_type": type(exc).__name__,
                        "reason_code": "transient_provider_error",
                        "backoff_ms": 1000 * 2**attempt,
                    },
                )
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable retry state")

    async def _chat_attempt(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Generate a response and emit its complete I/O lifecycle."""
        if kwargs.get("provider_api") == "responses":
            return await self._responses_attempt(messages, **kwargs)
        model = kwargs.get("model", self._model)
        limit_parameter, max_tokens = _output_limit(kwargs, self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        op_id = uuid.uuid4().hex
        t0 = time.perf_counter()

        self._emit(
            "llm.start",
            {
                "operation_id": op_id,
                "task_id": get_task_id(),
                "model": model,
                "provider_api": "chat",
                "messages": messages,
                "max_tokens": max_tokens,
                "output_limit_parameter": limit_parameter,
                "temperature": temperature,
                "response_format": kwargs.get("response_format"),
                **(
                    {"reasoning_effort": kwargs["reasoning_effort"]}
                    if "reasoning_effort" in kwargs
                    else {}
                ),
            },
        )

        try:
            create_kwargs: dict = {
                "model": model,
                "messages": messages,
                limit_parameter: max_tokens,
                "temperature": temperature,
            }

            if "response_format" in kwargs:
                create_kwargs["response_format"] = kwargs["response_format"]

            if "extra_body" in kwargs:
                create_kwargs["extra_body"] = kwargs["extra_body"]
            if "reasoning_effort" in kwargs:
                create_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]

            if "tools" in kwargs and kwargs["tools"]:
                create_kwargs["tools"] = kwargs["tools"]

            if self._request_timeouts:
                create_kwargs[
                    "timeout"
                ] = _active_timeout.get() or request_timeout_seconds(
                    self, _timeout_stage()
                )

            resp = await self._client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if not isinstance(finish_reason, str):
                finish_reason = None
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "usage_available": resp.usage is not None,
            }
            details = getattr(resp.usage, "completion_tokens_details", None)
            reasoning_tokens = (
                details.get("reasoning_tokens")
                if isinstance(details, dict)
                else getattr(details, "reasoning_tokens", None)
            )
            if reasoning_tokens is not None:
                available = (
                    type(reasoning_tokens) is int
                    and type(usage["completion_tokens"]) is int
                    and 0 <= reasoning_tokens <= usage["completion_tokens"]
                )
                usage["reasoning_usage_available"] = available
                if available:
                    usage["reasoning_tokens"] = reasoning_tokens

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

            if self._cost_budget and (
                usage["usage_available"] or not self._cost_budget.strict
            ):
                # Unknown strict usage must remain in flight so the outer
                # transport can retain its reservation instead of settling zero.
                self._cost_budget.record(usage)

            response = LLMResponse(
                content=choice.message.content or "",
                model=resp.model,
                usage=usage,
                tool_calls=tool_calls,
                reasoning_content=getattr(choice.message, "reasoning_content", "")
                or "",
                finish_reason=finish_reason,
                request_id=(
                    resp._request_id
                    if isinstance(getattr(resp, "_request_id", None), str)
                    else None
                ),
            )

            self._emit(
                "llm.complete",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "model": model,
                    "provider_api": "chat",
                    "messages": messages,
                    "content": response.content,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("prompt_tokens", 0)
                    + usage.get("completion_tokens", 0),
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "usage_available": resp.usage is not None,
                    "tool_calls": _tool_calls_for_trace(tool_calls),
                    "finish_reason": finish_reason,
                    "request_id": response.request_id,
                    "actual_model": response.model,
                    "max_tokens": max_tokens,
                    "output_limit_parameter": limit_parameter,
                    **(
                        {"reasoning_effort": kwargs["reasoning_effort"]}
                        if "reasoning_effort" in kwargs
                        else {}
                    ),
                    **{
                        k: usage[k]
                        for k in ("reasoning_tokens", "reasoning_usage_available")
                        if k in usage
                    },
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
                    "provider_api": "chat",
                    "max_tokens": max_tokens,
                    "output_limit_parameter": limit_parameter,
                    "elapsed_ms": elapsed_ms,
                    "error_type": type(e).__name__,
                    "error": str(e)[:512],
                    "usage_available": False,
                    **(
                        self._observe_provider_rejection(e.response)
                        if isinstance(e, APIStatusError)
                        else {}
                    ),
                    **_failure_timeout_metadata(e),
                },
            )
            raise

    async def _responses_attempt(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Use one owned SDK request, retaining known charges on invalid output."""
        from litagent.llm.responses import (
            build_responses_request,
            normalize_responses,
            private_receipt,
        )

        model = kwargs.get("model", self._model)
        output_limit = kwargs["max_output_tokens"]
        body = build_responses_request(
            messages,
            model=model,
            output_limit=output_limit,
            response_format=kwargs.get("response_format"),
            temperature=kwargs.get("temperature", self._temperature),
            extra_body=kwargs.get("extra_body"),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
        op_id, started = uuid.uuid4().hex, time.perf_counter()
        metadata = {
            "operation_id": op_id,
            "task_id": get_task_id(),
            "model": model,
            "provider_api": "responses",
            "max_tokens": output_limit,
            "output_limit_parameter": "max_output_tokens",
            "reasoning_effort": kwargs.get("reasoning_effort"),
        }
        self._emit(
            "llm.start",
            {
                **metadata,
                "messages": messages,
                "response_format": kwargs.get("response_format"),
            },
        )
        try:
            thinking = body.pop("thinking", None)
            if thinking is not None:
                body["extra_body"] = {"thinking": thinking}
            if self._request_timeouts:
                body["timeout"] = _active_timeout.get() or request_timeout_seconds(
                    self, _timeout_stage()
                )
            try:
                raw = await self._client.responses.with_raw_response.create(**body)
                http_response = raw.http_response
            except APIStatusError as exc:
                # A received error can still carry billing; normalize before deciding
                # whether the outer owner must retain an unknown reservation.
                http_response = exc.response
            receipt = private_receipt(http_response)
            fields = normalize_responses(http_response.content)
            rejection = self._observe_provider_rejection(http_response)
            if (
                fields["usage"]["usage_available"]
                and fields["usage"]["completion_tokens"] > output_limit
            ):
                fields.update(
                    provider_error="responses_output_limit_exceeded",
                    finish_reason="invalid",
                )
            if http_response.status_code != 200:
                fields.update(
                    provider_error=f"responses_http_{http_response.status_code}",
                    finish_reason="invalid",
                )
            response = ResponsesLLMResponse(
                **fields,
                provider_receipt=receipt,
                request_id=http_response.headers.get("x-request-id"),
            )
            if self._cost_budget and response.usage["usage_available"]:
                self._cost_budget.record(response.usage)
            self._emit(
                "llm.complete",
                {
                    **metadata,
                    "actual_model": response.model,
                    "content": response.content,
                    "messages": messages,
                    **response.usage,
                    "total_tokens": response.usage["prompt_tokens"]
                    + response.usage["completion_tokens"],
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                    "provider_status": response.provider_status,
                    "provider_error": response.provider_error,
                    **rejection,
                },
            )
            return response
        except BaseException as exc:
            self._emit(
                "llm.failed",
                {
                    **metadata,
                    "error_type": type(exc).__name__,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "usage_available": False,
                    **_failure_timeout_metadata(exc),
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
