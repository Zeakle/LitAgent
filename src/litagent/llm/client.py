"""LLM Client--异步 LLM调用封装"""


from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from litagent.observability.context import get_task_id
from litagent.safety.budget import CostBudget
from litagent.logging import get_logger


logger = get_logger('llm.client')


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, message: list[dict], **kwargs) -> LLMResponse:
        ...

    
class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI 兼容的 LLM Client（DeepSeek / GPT / Claude 等）。

    通过 base_url 切换不同的 LLM 提供商。
    API key 从环境变量 LLM_API_KEY 或 OPENAI_API_KEY 读取。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        cost_budget: CostBudget | None = None,
        trace_hook=None
    ):
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
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")


    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        model = kwargs.get('model', self._model)
        max_tokens = kwargs.get('max_tokens', self._max_tokens)
        temperature = kwargs.get('temperature', self._temperature)

        try:
            create_kwargs: dict = {
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            }

            if 'response_format' in kwargs:
                create_kwargs['response_format'] = kwargs['response_format']

            if 'tools' in kwargs and kwargs['tools']:
                create_kwargs['tools'] = kwargs['tools']

            resp = await self._client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            usage = {
                'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
            }

            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments,
                        },
                    })

            if self._cost_budget:
                self._cost_budget.record(usage)

            response = LLMResponse(
                content=choice.message.content or "",
                model=resp.model,
                usage=usage,
                tool_calls=tool_calls
            )

            self._emit("llm.call", {
                "task_id": get_task_id(),
                "model": response.model,
                "messages": messages,
                "content": response.content,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                "tool_calls": [tc.get("function", {}).get("name", "") for tc in tool_calls],
            })

            return response

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


class MockLLMClient(BaseLLMClient):
    """测试用 Mock Client——返回预设响应。"""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or ["Mock LLM response"])
        self._call_count = 0

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return LLMResponse(content=self._responses[idx], model="mock")