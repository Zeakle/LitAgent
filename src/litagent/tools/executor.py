import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from litagent.tools.base import ToolDefinition, RateLimitConfig, FallbackStep
from litagent.tools.registry import ToolRegistry, get_registry
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

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._cache: dict[str, Any] = {}
        self._rate_limits: dict[str, _RateLimitState] = {}

    async def execute(self, name: str, args: dict, session_id: str = "") -> ToolResult:
        """执行一次工具调用，走完整保护链路。

        链路: 限流检查 → 缓存检查 → 执行(retry) → 降级(fallback)
        """
        t0 = time.monotonic()

        try:
            registered = self._registry.get(name)
        except KeyError:
            return ToolResult(name=name, args=args, error=f'Tool {name} not registered', elapsed_ms=0)

        td = registered.definition

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

        # 4. 执行失败 -> 降级
        if result.error and td.fallback:
            result = await self._apply_fallback(name, args, td.fallback)

        # 5. 写入缓存
        if not result.error and td.cache_ttl_ms > 0:
            self._cache[cache_key] = {'data': result.output, 'ts': time.time()}

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result


    async def _execute_with_retry(
        self, name: str, args: dict, func: callable, timeout_ms: int, max_retries: int
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
    
    async def _call_func(self, func: callable, args: dict) -> Any:
        """调用工具函数，支持同步和异步"""
        result = func(**args)
        if asyncio.iscoroutine(result):
            result = await result
        return result
    

    async def _apply_fallback(self, name: str, args: dict, fallback: list[FallbackStep]) -> ToolResult:
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
        return ToolResult(name=name, args=args, error='All fallback steps exhausted')


    def _make_cache_key(self, name: str, args: dict) -> str:
        raw = json.dumps({'name': name, 'args': args}, sort_keys=True)
        # 对存入缓存数据进行SHA-256哈希，返回十六进制格式
        return hashlib.sha256(raw.encode()).hexdigest()


    def _check_rate_limit(self, name: str, config: RateLimitConfig) -> bool:
        if name not in self._rate_limits:
            self._rate_limits[name] = _RateLimitState(max_calls=config.max_calls, window_seconds=config.window_seconds)
        return self._rate_limits[name].allow()
