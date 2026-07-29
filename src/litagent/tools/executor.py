"""Execute tools with tracing, retries, limits, caching, and fallbacks."""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from litagent.errors.circuit_breaker import CircuitBreaker
from litagent.observability.context import get_task_id
from litagent.tools.base import ToolDefinition, RateLimitConfig, FallbackStep
from litagent.tools.registry import ToolRegistry
from litagent.observability.lifecycle import sanitize_input
from litagent.logging import get_logger

logger = get_logger("tools.executor")


@dataclass
class ToolResult:
    """Capture a tool outcome and its execution metadata."""

    name: str
    args: dict
    output: Any = None
    error: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    from_cache: bool = False
    from_fallback: bool = False
    elapsed_ms: float = 0


@dataclass
class _RateLimitState:
    """Track timestamps for one rolling-window rate limit."""

    max_calls: int
    window_seconds: int
    timestamps: list[float] = field(default_factory=list)

    def allow(self) -> bool:
        """Consume a call slot when the rolling-window limit permits it."""
        now = time.time()
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        if len(self.timestamps) >= self.max_calls:
            return False
        self.timestamps.append(now)
        return True


class ToolExecutor:
    """Apply execution policies around registered tool callables."""

    def __init__(
        self,
        registry: ToolRegistry,
        cb_fail_threshold: int = 5,
        cb_cooldown_seconds: int = 60,
        trace_hook=None,
    ):
        self._registry = registry
        self._cache: dict[str, Any] = {}
        self._rate_limits: dict[str, _RateLimitState] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cb_fail_threshold = cb_fail_threshold
        self._cb_cooldown = cb_cooldown_seconds
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    def _get_breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                self._cb_fail_threshold, self._cb_cooldown
            )
        return self._breakers[name]

    @staticmethod
    def _tool_failure(
        name: str,
        args: dict,
        exc: BaseException,
    ) -> ToolResult:
        if isinstance(exc, asyncio.TimeoutError):
            code = "tool_timeout"
            error_type = "TimeoutError"
            error = "Timeout while executing tool"
        elif isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            code = "http_rate_limited"
            error_type = type(exc).__name__
            error = str(exc)[:512]
        else:
            code = "tool_execution_failed"
            error_type = type(exc).__name__
            error = str(exc)[:512]

        return ToolResult(
            name=name,
            args=args,
            error=error,
            error_code=code,
            error_type=error_type,
        )

    async def execute(self, name: str, args: dict, session_id: str = "") -> ToolResult:
        """Execute a registered tool and emit one terminal lifecycle event."""
        op_id = uuid.uuid4().hex
        t0 = time.monotonic()
        self._emit(
            "tool.start",
            {
                "operation_id": op_id,
                "task_id": get_task_id(),
                "name": name,
                "args": sanitize_input(args),
            },
        )
        try:
            result = await self._execute_inner(name, args, session_id)
        # Trace cancellation without converting it into a normal tool result.
        except BaseException as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._emit(
                "tool.failed",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "name": name,
                    "elapsed_ms": elapsed_ms,
                    "error_code": (
                        "tool_cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "tool_execution_failed"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:512],
                },
            )
            raise

        if not result.elapsed_ms:
            result.elapsed_ms = (time.monotonic() - t0) * 1000

        if result.error is not None:
            self._emit(
                "tool.failed",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "name": name,
                    "elapsed_ms": result.elapsed_ms,
                    "error_code": result.error_code or "tool_execution_failed",
                    "error_type": result.error_type or "ToolError",
                    "error": result.error[:512],
                },
            )
        else:
            self._emit(
                "tool.complete",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "name": name,
                    "elapsed_ms": result.elapsed_ms,
                    "from_cache": result.from_cache,
                    "from_fallback": result.from_fallback,
                    "output_size": (
                        len(result.output)
                        if isinstance(result.output, (list, str))
                        else None
                    ),
                    "output": result.output,
                },
            )

        return result

    async def _execute_inner(
        self, name: str, args: dict, session_id: str = ""
    ) -> ToolResult:
        """Apply circuit breaking, limits, caching, retries, and fallbacks."""
        t0 = time.monotonic()

        try:
            registered = self._registry.get(name)
        except KeyError:
            return ToolResult(
                name=name,
                args=args,
                error=f"Tool {name} not registered",
                error_code="tool_not_registered",
                error_type="KeyError",
                elapsed_ms=0,
            )

        td = registered.definition

        breaker = self._get_breaker(name)
        if not breaker.allow():
            return ToolResult(
                name=name,
                args=args,
                error="Circuit breaker open",
                error_code="circuit_breaker_open",
                error_type="CircuitBreakerOpen",
            )

        if td.rate_limit and not self._check_rate_limit(name, td.rate_limit):
            return ToolResult(
                name=name,
                args=args,
                error="Rate limit exceeded",
                error_code="tool_rate_limited",
                error_type="RateLimitExceeded",
                elapsed_ms=0,
            )

        cache_key = self._make_cache_key(name, args)
        if td.cache_ttl_ms > 0 and cache_key in self._cache:
            entry = self._cache[cache_key]

            if (time.time() - entry["ts"]) * 1000 < td.cache_ttl_ms:
                logger.debug(f"Cache hit: {name}")
                return ToolResult(
                    name=name,
                    args=args,
                    output=entry["data"],
                    from_cache=True,
                    elapsed_ms=0,
                )

        result = await self._execute_with_retry(
            name, args, registered.func, td.timeout_ms, td.max_retries
        )

        # Record the primary attempt before fallback changes the outward result.
        if result.error:
            breaker.record_failure()
        else:
            breaker.record_success()

        original_failure = result
        if result.error and td.fallback:
            result = await self._apply_fallback(name, args, td.fallback, result.error)
            result.from_fallback = True

            # Preserve the primary failure classification after fallback recovery.
            if result.error is None:
                result.error_code = original_failure.error_code
                result.error_type = original_failure.error_type
            else:
                result.error_code = result.error_code or "fallback_exhausted"
                result.error_type = result.error_type or original_failure.error_type

        if not result.error and td.cache_ttl_ms > 0:
            self._cache[cache_key] = {"data": result.output, "ts": time.time()}

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    async def _execute_with_retry(
        self, name: str, args: dict, func: Callable, timeout_ms: int, max_retries: int
    ) -> ToolResult:
        last_failure: ToolResult | None = None

        for attempt in range(max_retries + 1):
            try:
                output = await asyncio.wait_for(
                    self._call_func(func, args),
                    timeout=timeout_ms / 1000,
                )
                return ToolResult(name=name, args=args, output=output)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_failure = self._tool_failure(name, args, exc)

            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

        return last_failure or ToolResult(
            name=name,
            args=args,
            error="Unknown tool failure",
            error_code="tool_execution_failed",
            error_type="UnknownError",
        )

    async def _call_func(self, func: Callable, args: dict) -> Any:
        """Invoke either a synchronous or asynchronous tool callable."""
        result = func(**args)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _apply_fallback(
        self,
        name: str,
        args: dict,
        fallback: list[FallbackStep],
        original_error: str = "",
    ) -> ToolResult:
        """Run fallback steps in order and return the first applicable result."""
        for step in fallback:
            if step.type == "cached":
                cache_key = self._make_cache_key(name, args)
                if cache_key in self._cache:
                    return ToolResult(
                        name=name,
                        args=args,
                        output=self._cache[cache_key]["data"],
                        from_fallback=True,
                    )
            elif step.type == "default_value":
                return ToolResult(
                    name=name, args=args, output=step.default_result, from_fallback=True
                )
            elif step.type == "skip":
                return ToolResult(name=name, args=args, output=None, from_fallback=True)
            elif step.type == "alternative_tool" and step.alternative_tool:
                try:
                    alt = self._registry.get(step.alternative_tool)
                    return await self._execute_with_retry(
                        step.alternative_tool,
                        args,
                        alt.func,
                        alt.definition.timeout_ms,
                        alt.definition.max_retries,
                    )
                except KeyError:
                    continue
        msg = f"All fallback steps exhausted"
        if original_error:
            msg += f". Original: {original_error}"
        return ToolResult(name=name, args=args, error=msg, from_fallback=True)

    def _make_cache_key(self, name: str, args: dict) -> str:
        raw = json.dumps({"name": name, "args": args}, sort_keys=True)

        return hashlib.sha256(raw.encode()).hexdigest()

    def _check_rate_limit(self, name: str, config: RateLimitConfig) -> bool:
        if name not in self._rate_limits:
            self._rate_limits[name] = _RateLimitState(
                max_calls=config.max_calls, window_seconds=config.window_seconds
            )
        return self._rate_limits[name].allow()
