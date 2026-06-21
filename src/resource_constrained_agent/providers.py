"""LLM provider abstractions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Protocol, cast

import httpx
import ollama
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from resource_constrained_agent.schemas import LLMResponse, TokenUsage

ChatMessage = dict[str, str]


class LLMProviderError(RuntimeError):
    """Raised when the model provider fails or times out."""


class LLMProvider(Protocol):
    def estimate_cost(self, messages: list[ChatMessage]) -> float:
        """Estimate the next call cost for budget preflight."""

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        """Return model text plus token usage."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OllamaProvider:
    def __init__(
        self,
        model: str,
        host: str,
        price_per_1k_tokens: float,
        timeout_seconds: float = 120.0,
        max_completion_tokens: int = 400,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_completion_tokens = max_completion_tokens
        self.client = ollama.Client(host=host, timeout=timeout_seconds)
        self.price_per_1k_tokens = price_per_1k_tokens

    def estimate_cost(self, messages: list[ChatMessage]) -> float:
        prompt_tokens = estimate_tokens(json.dumps(messages))
        return ((prompt_tokens + self.max_completion_tokens) / 1000) * self.price_per_1k_tokens

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0, "num_predict": self.max_completion_tokens},
            )
        except (httpx.HTTPError, ollama.ResponseError) as exc:
            raise LLMProviderError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc
        text = response["message"]["content"]
        prompt_tokens = int(
            response.get("prompt_eval_count") or estimate_tokens(json.dumps(messages))
        )
        completion_tokens = int(response.get("eval_count") or estimate_tokens(text))
        total = prompt_tokens + completion_tokens
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=(total / 1000) * self.price_per_1k_tokens,
            ),
        )


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        input_price_per_1m_tokens: float = 0.15,
        output_price_per_1m_tokens: float = 0.60,
        timeout_seconds: float = 30.0,
        max_completion_tokens: int = 400,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.input_price_per_1m_tokens = input_price_per_1m_tokens
        self.output_price_per_1m_tokens = output_price_per_1m_tokens
        self.max_completion_tokens = max_completion_tokens

    def estimate_cost(self, messages: list[ChatMessage]) -> float:
        prompt_tokens = estimate_tokens(json.dumps(messages))
        return (
            prompt_tokens / 1_000_000 * self.input_price_per_1m_tokens
            + self.max_completion_tokens / 1_000_000 * self.output_price_per_1m_tokens
        )

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        openai_messages = cast("list[ChatCompletionMessageParam]", messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                response_format={"type": "json_object"},
                max_tokens=self.max_completion_tokens,
            )
        except OpenAIError as exc:
            raise LLMProviderError(f"OpenAI request failed: {type(exc).__name__}: {exc}") from exc
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else estimate_tokens(json.dumps(messages))
        completion_tokens = usage.completion_tokens if usage else estimate_tokens(text)
        cost = (
            prompt_tokens / 1_000_000 * self.input_price_per_1m_tokens
            + completion_tokens / 1_000_000 * self.output_price_per_1m_tokens
        )
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            ),
        )


class ScriptedProvider:
    """Deterministic provider used by tests and the offline demo harness."""

    def __init__(self, responses: Iterable[str], cost_per_call: float = 0.001) -> None:
        self._responses = list(responses)
        self._index = 0
        self._cost_per_call = cost_per_call

    def estimate_cost(self, messages: list[ChatMessage]) -> float:
        _ = messages
        return self._cost_per_call

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        if self._index >= len(self._responses):
            text = json.dumps(
                {
                    "thought": "No scripted response remains.",
                    "progress_assessment": "The deterministic provider is exhausted.",
                    "is_stuck": True,
                    "new_plan": None,
                    "action": {
                        "kind": "final_answer",
                        "final_answer": "Stopped because the scripted provider ran out of steps.",
                    },
                }
            )
        else:
            text = self._responses[self._index]
        self._index += 1
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=estimate_tokens(json.dumps(messages)),
                completion_tokens=estimate_tokens(text),
                cost_usd=self._cost_per_call,
            ),
        )
