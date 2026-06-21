from types import SimpleNamespace
from typing import Any, cast

import pytest

from resource_constrained_agent.cli import build_provider
from resource_constrained_agent.providers import OpenAIProvider
from resource_constrained_agent.settings import Settings


def test_openai_provider_uses_chat_completions_json_mode() -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    provider = OpenAIProvider(model="gpt-test", api_key="test", max_completion_tokens=400)
    provider.client = cast(
        Any, SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    )

    response = provider.complete([{"role": "user", "content": "hi"}])

    assert response.text == '{"ok": true}'
    assert captured["model"] == "gpt-test"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 400


def test_openai_provider_requires_key_in_cli_settings() -> None:
    settings = Settings(agent_llm_provider="openai", openai_api_key=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_provider(settings)
