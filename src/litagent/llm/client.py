"""LLM Client--异步 LLM调用封装"""


from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from litagent.logging import get_logger


logger = get_logger('llm.client')


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)


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
    ):
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("NO LLM API key found (LLM_API_KEY / OPENAI_API_KEY)")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

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

            resp = await self._client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=resp.model,
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                },
            )
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