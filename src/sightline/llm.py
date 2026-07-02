"""LLM client factory: Anthropic direct, or any OpenAI-compatible gateway (e.g. OpenRouter).

Two wire protocols exist in practice:
  - Anthropic's own API (the `anthropic` SDK, /v1/messages)
  - the OpenAI-compatible format (/chat/completions) that gateways like OpenRouter expose
    while routing to many models (including Claude)

Rather than teach the answerer/judge two protocols, this module gives them ONE interface —
the Anthropic SDK's `client.messages.create(...) -> resp.content[0].text` shape — and adapts
the OpenAI-compatible wire format behind it when `LLM_BASE_URL` is set. Callers never know
the difference, and unit tests keep injecting simple fakes of the same shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import settings


@dataclass
class _TextBlock:
    text: str


@dataclass
class _Response:
    content: list[_TextBlock]


class _OpenAICompatMessages:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
        )

    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, Any]]) -> _Response:
        resp = self._client.post(
            "/chat/completions",
            json={"model": model, "max_tokens": max_tokens, "messages": messages},
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:  # OpenRouter can return 200 with an embedded error
            raise RuntimeError(f"LLM gateway error: {data['error']}")
        return _Response(content=[_TextBlock(text=data["choices"][0]["message"]["content"])])


class OpenAICompatClient:
    """Duck-types the tiny slice of the Anthropic SDK that Sightline uses."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.messages = _OpenAICompatMessages(base_url, api_key)


def make_client() -> object:
    """Return an LLM client for the configured provider. Raises if no key is set."""
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY is empty — set it in .env (Anthropic key, or an OpenAI-compatible "
            "gateway key such as OpenRouter with LLM_BASE_URL)."
        )
    if settings.llm_base_url:
        return OpenAICompatClient(settings.llm_base_url, settings.llm_api_key)
    from anthropic import Anthropic

    return Anthropic(api_key=settings.llm_api_key)
