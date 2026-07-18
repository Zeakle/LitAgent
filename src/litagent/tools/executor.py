import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import uuid

from litagent.errors.circuit_breaker import CircuitBreaker
from litagent.observability.context import get_task_id
from litagent.tools.base import ToolDefinition, RateLimitConfig, FallbackStep
from litagent.tools.registry import ToolRegistry
from litagent.observability.lifecycle import sanitize_input
from litagent.logging import get_logger

logger = get_logger('tools.executor')


@dataclass
class ToolResult:
    """工具执行结果"""
    name: str
    args: dict
    output: Any = None
    error: str | None = None
    from_cache: bool = False
    from_fallback: bool = False
    elapsed_ms: float = 0


@dataclass
class _RateLimitState:
    """单个tool的限流状态(滑窗)"""
    max_calls: int
    window_seconds: int
    timestamps: list[float] = field(default_factory=list)

    def allow(self) -> bool:
        """统计过去self.window_seconds内工具调用频次是否超过max_calls"""
        now = time.time()
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        if len(self.timestamps) >= self.max_calls:
            return False
        self.timestamps.append(now)
        return True


class ToolExecutor:
    """工具执行器——超时 + 重试 + 缓存 + 限流 + 降级。

    每个 session 持有一个实例（缓存和限流状态是 session 级别的）。
    """

    def __init__(self, registry: ToolRegistry, cb_fail_threshold: int = 5, cb_cooldown_seconds: int = 60, trace_hook=None):
        self._registry = registry
        self._cache: dict[str, Any] = {}  # TODO Phase 11: add TTL-based eviction
        self._rate_limits: dict[str, _RateLimitState] = {}
        # circuit breaker
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cb_fail_threshold = cb_fail_threshold
        self._cb_cooldown = cb_cooldown_seconds
        self._trace_hook = trace_hook


    def _emit(self, event: str, data: dict) -> None:
        """触发 trace hook"""
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


    async def execute(self, name: str, args: dict, session_id: str = '') -> ToolResult:
        op_id = uuid.uuid4().hex
        t0 = time.monotonic()
        self._emit('tool.start', {
            'operation_id': op_id,
            'task_id': get_task_id(),
            'name': name,
            'args': sanitize_input(args),
        })
        try:
            result = await self._execute_inner(name, args, session_id)
        except BaseException as e:   # CancelledError 也要关闭 observation
            self._emit('tool.failed', {
                'operation_id': op_id, 'task_id': get_task_id(), 'name': name,
                'elapsed_ms': (time.monotonic() - t0) * 1000,
                'error_type': type(e).__name__, 'error': str(e)[:512],
            })
            raise

        if not result.elapsed_ms:
            result.elapsed_ms = (time.monotonic() - t0) * 1000

        if result.error is not None:
            self._emit('tool.failed', {
                'operation_id': op_id, 'task_id': get_task_id(), 'name': name,
                'elapsed_ms': result.elapsed_ms,
                'error_type': 'tool_error', 'error': result.error[:512],
            })
        else:
            self._emit('tool.complete', {
                'operation_id': op_id, 'task_id': get_task_id(), 'name': name,
                'elapsed_ms': result.elapsed_ms,
                'from_cache': result.from_cache,
                'from_fallback': result.from_fallback,
                'output_size': len(result.output) if isinstance(result.output, (list, str)) else None,
            })

        return result

    async def _execute_inner(self, name: str, args: dict, session_id: str = "") -> ToolResult:
        """执行一次工具调用，走完整保护链路。

        链路: 限流检查 → 缓存检查 → 执行(retry) → 降级(fallback)
        """
        t0 = time.monotonic()

        try:
            registered = self._registry.get(name)
        except KeyError:
            return ToolResult(name=name, args=args, error=f'Tool {name} not registered', elapsed_ms=0)

        td = registered.definition

        # 熔断检查
        breaker = self._get_breaker(name)
        if not breaker.allow():
            return ToolResult(name=name, args=args, error='Circuit breaker open')

        # 1.限流检查
        if td.rate_limit and not self._check_rate_limit(name, td.rate_limit):
            return ToolResult(name=name, args=args, error='Rate limit exceeded', elapsed_ms=0)

        # 2.缓存检查 (cache_ttl_ms(time to live, cache持续时间) > 0 且非 destructive)
        cache_key = self._make_cache_key(name, args)
        if td.cache_ttl_ms > 0 and cache_key in self._cache:
            entry = self._cache[cache_key]
            # 上一次同参数工具在缓存内维续时间<ttl_ms
            if (time.time() - entry['ts']) * 1000 < td.cache_ttl_ms:
                logger.debug(f"Cache hit: {name}")
                return ToolResult(name=name, args=args, output=entry['data'], from_cache=True, elapsed_ms=0)

        # 3. 执行(带重试)
        result = await self._execute_with_retry(name, args, registered.func, td.timeout_ms, td.max_retries)

        # record state on breaker
        if result.error:
            breaker.record_failure()
        else:
            breaker.record_success()

        # 4. 执行失败 -> 降级
        if result.error and td.fallback:
            result = await self._apply_fallback(name, args, td.fallback, result.error)

        # 5. 写入缓存
        if not result.error and td.cache_ttl_ms > 0:
            self._cache[cache_key] = {'data': result.output, 'ts': time.time()}

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result


    async def _execute_with_retry(
        self, name: str, args: dict, func: Callable, timeout_ms: int, max_retries: int
    ) -> ToolResult:
        """带超时 + 指数退避重试的执行"""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                output = await asyncio.wait_for(
                    self._call_func(func, args),
                    timeout=timeout_ms / 1000
                )
                return ToolResult(name=name, args=args, output=output)
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout_ms}ms"
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                wait = 2 ** attempt
                logger.debug(f"Retry {attempt + 1}/{max_retries} for tool, waiting {wait}s")
                await asyncio.sleep(wait)

        return ToolResult(name=name, args=args, error=last_error)
    
    async def _call_func(self, func: Callable, args: dict) -> Any:
        """调用工具函数，支持同步和异步"""
        result = func(**args)
        if asyncio.iscoroutine(result):
            result = await result
        return result
    

    async def _apply_fallback(self, name: str, args: dict, fallback: list[FallbackStep],
                             original_error: str = "") -> ToolResult:
        """逐级尝试降级链"""
        for step in fallback:
            if step.type == "cached":
                cache_key = self._make_cache_key(name, args)
                if cache_key in self._cache:
                    return ToolResult(name=name, args=args, output=self._cache[cache_key]['data'], from_fallback=True)
            elif step.type == 'default_value':
                return ToolResult(name=name, args=args, output=step.default_result, from_fallback=True)
            elif step.type == 'skip':
                return ToolResult(name=name, args=args, output=None, from_fallback=True)
            elif step.type == 'alternative_tool' and step.alternative_tool:
                try:
                    alt = self._registry.get(step.alternative_tool)
                    return await self._execute_with_retry(
                        step.alternative_tool, args, alt.func,
                        alt.definition.timeout_ms, alt.definition.max_retries
                    )
                except KeyError:
                    continue
        msg = f"All fallback steps exhausted"
        if original_error:
            msg += f". Original: {original_error}"
        return ToolResult(name=name, args=args, error=msg, from_fallback=True)


    def _make_cache_key(self, name: str, args: dict) -> str:
        raw = json.dumps({'name': name, 'args': args}, sort_keys=True)
        # 对存入缓存数据进行SHA-256哈希，返回十六进制格式
        return hashlib.sha256(raw.encode()).hexdigest()


    def _check_rate_limit(self, name: str, config: RateLimitConfig) -> bool:
        if name not in self._rate_limits:
            self._rate_limits[name] = _RateLimitState(max_calls=config.max_calls, window_seconds=config.window_seconds)
        return self._rate_limits[name].allow()
