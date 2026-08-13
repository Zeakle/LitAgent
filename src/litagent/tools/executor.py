"""Execute tools with tracing, retries, limits, caching, and fallbacks."""

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from litagent.errors.circuit_breaker import CircuitBreaker
from litagent.logging import get_logger
from litagent.observability.context import get_task_id
from litagent.observability.lifecycle import sanitize_input
from litagent.tools.base import (
    FallbackStep,
    RateLimitConfig,
    RegisteredTool,
    ToolCategory,
)
from litagent.tools.registry import ToolRegistry

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
    fallback_tool: str | None = None
    policy_stage: str | None = None
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
        *,
        allowed_names: Collection[str],
        allowed_categories: Collection[ToolCategory] = (ToolCategory.READ,),
    ) -> None:
        """Initialize the tool executor."""
        self._registry = registry
        self._allowed_names = frozenset(allowed_names)
        self._allowed_categories = frozenset(allowed_categories)
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
        """Return the circuit breaker assigned to a tool."""
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
        """Build a normalized failed tool result."""
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

    async def execute(self, name: str, args: Any, session_id: str = "") -> ToolResult:
        """Execute a registered tool and emit one terminal lifecycle event."""
        op_id = uuid.uuid4().hex
        t0 = time.monotonic()
        arguments_valid = isinstance(args, Mapping)
        try:
            stable_args = dict(args) if arguments_valid else {}
        except Exception:
            arguments_valid = False
            stable_args = {}
        self._emit(
            "tool.start",
            {
                "operation_id": op_id,
                "task_id": get_task_id(),
                "name": name,
                "args": sanitize_input(stable_args),
            },
        )
        try:
            if arguments_valid:
                result = await self._execute_inner(name, stable_args, session_id)
            else:
                result = ToolResult(
                    name=name,
                    args={},
                    error="Tool arguments must be a stable mapping",
                    error_code="tool_arguments_invalid",
                    error_type="ValidationError",
                    policy_stage="schema",
                )
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
                    "policy_stage": result.policy_stage,
                    "fallback_tool": result.fallback_tool,
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
                    "fallback_tool": result.fallback_tool,
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
        self, name: str, args: Any, session_id: str = ""
    ) -> ToolResult:
        """Apply circuit breaking, limits, caching, retries, and fallbacks."""
        if not isinstance(args, Mapping):
            return ToolResult(
                name=name,
                args={},
                error="Tool arguments must be a mapping",
                error_code="tool_arguments_invalid",
                error_type="ValidationError",
                policy_stage="schema",
            )

        stable_args = dict(args)
        preflight = self._preflight(name, stable_args)
        if isinstance(preflight, ToolResult):
            return preflight

        return await self._execute_registered(
            name,
            stable_args,
            preflight,
            allow_fallback=True,
        )

    def _preflight(
        self, name: str, args: Mapping[str, Any]
    ) -> RegisteredTool | ToolResult:
        """Validate registry, policy, and schema before execution side effects."""
        stable_args = dict(args)
        try:
            registered = self._registry.get(name)
        except KeyError:
            return ToolResult(
                name=name,
                args=stable_args,
                error=f"Tool {name} not registered",
                error_code="tool_not_registered",
                error_type="KeyError",
                policy_stage="registry",
            )

        if name not in self._allowed_names:
            return ToolResult(
                name=name,
                args=stable_args,
                error=f"Tool {name} is not allowed",
                error_code="tool_not_allowed",
                error_type="ToolPolicyViolation",
                policy_stage="allowlist",
            )

        category = registered.definition.category
        if not isinstance(category, ToolCategory):
            return ToolResult(
                name=name,
                args=stable_args,
                error=f"Tool {name} has an unknown capability",
                error_code="tool_capability_unknown",
                error_type="ToolPolicyViolation",
                policy_stage="category",
            )
        if category not in self._allowed_categories:
            return ToolResult(
                name=name,
                args=stable_args,
                error=f"Tool category {category.value} is not allowed",
                error_code="tool_category_denied",
                error_type="ToolPolicyViolation",
                policy_stage="category",
            )

        schema = registered.definition.parameters
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(stable_args)
        except SchemaError:
            return ToolResult(
                name=name,
                args=stable_args,
                error=f"Tool {name} has an invalid argument schema",
                error_code="tool_schema_invalid",
                error_type="SchemaError",
                policy_stage="schema",
            )
        except ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path)
            location = f"$.{path}" if path else "$"
            return ToolResult(
                name=name,
                args=stable_args,
                error=(
                    f"Invalid tool arguments at {location}: "
                    f"failed {exc.validator} validation"
                ),
                error_code="tool_arguments_invalid",
                error_type="ValidationError",
                policy_stage="schema",
            )

        try:
            json.dumps(stable_args, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return ToolResult(
                name=name,
                args=stable_args,
                error="Tool arguments are not JSON serializable",
                error_code="tool_arguments_invalid",
                error_type="ValidationError",
                policy_stage="schema",
            )

        return registered

    async def _execute_registered(
        self,
        name: str,
        args: dict[str, Any],
        registered: RegisteredTool,
        *,
        allow_fallback: bool,
    ) -> ToolResult:
        """Apply runtime policies after a tool has passed preflight."""
        t0 = time.monotonic()
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
        if result.error and allow_fallback and td.fallback:
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
        """Execute a tool with timeout, retry, and circuit-breaker handling."""
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
                preflight = self._preflight(step.alternative_tool, args)
                if isinstance(preflight, ToolResult):
                    preflight.name = name
                    preflight.from_fallback = True
                    preflight.fallback_tool = step.alternative_tool
                    return preflight

                result = await self._execute_registered(
                    step.alternative_tool,
                    args,
                    preflight,
                    allow_fallback=False,
                )
                result.name = name
                result.from_fallback = True
                result.fallback_tool = step.alternative_tool
                return result
        msg = f"All fallback steps exhausted"
        if original_error:
            msg += f". Original: {original_error}"
        return ToolResult(name=name, args=args, error=msg, from_fallback=True)

    def _make_cache_key(self, name: str, args: dict) -> str:
        """Build a stable cache key from a tool name and arguments."""
        raw = json.dumps({"name": name, "args": args}, sort_keys=True, allow_nan=False)

        return hashlib.sha256(raw.encode()).hexdigest()

    def _check_rate_limit(self, name: str, config: RateLimitConfig) -> bool:
        """Enforce the configured per-tool rate limit."""
        if name not in self._rate_limits:
            self._rate_limits[name] = _RateLimitState(
                max_calls=config.max_calls, window_seconds=config.window_seconds
            )
        return self._rate_limits[name].allow()
