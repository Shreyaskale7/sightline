"""LLM client factory: Anthropic direct, or any OpenAI-compatible gateway (e.g. OpenRouter).

Two wire protocols exist in practice:
  - Anthropic's own API (the `anthropic` SDK, /v1/messages)
  - the OpenAI-compatible format (/chat/completions) that gateways like OpenRouter expose
    while routing to many models (including Claude)

Rather than teach the answerer/judge two protocols, this module gives them ONE interface —
the Anthropic SDK's `client.messages.create(...) -> resp.content[0].text` shape — and adapts
the OpenAI-compatible wire format behind it when `LLM_BASE_URL` is set. Callers never know
the difference, and unit tests keep injecting simple fakes of the same shape.

Every client is wrapped in an exact-match RESPONSE CACHE (SQLite, keyed on a hash of
model+params+messages). Why: on a free tier the daily request quota IS the budget —
typically ~50 requests/day. With the cache, repeating an eval run only pays for
cases it hasn't seen, so a quota-interrupted run finishes for free the next day, and
"run the eval again" after a code refactor costs zero requests. (Semantic caching — matching
*similar* prompts — arrives in M4; exact-match is the safe, obviously-correct version.)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import settings


@dataclass
class _TextBlock:
    text: str


@dataclass
class _Response:
    content: list[_TextBlock]
    # Token usage as reported by the provider (M4 cost accounting reads this).
    # None = unknown; cache hits deliberately report zero-cost usage.
    usage: dict[str, Any] | None = None


def _quota_message(resp: Any, model: str) -> str:
    """A human-readable reason from a 429, preferring whatever the provider actually said.

    Providers vary in where they put the detail, so this digs a little and then falls back to a
    plain sentence — the caller needs *something* it can show a user, never an empty error.
    """
    detail = ""
    try:
        body = resp.json()
        err = body.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or "")
        elif isinstance(err, str):
            detail = err
        detail = detail or str(body.get("message") or "")
    except Exception:
        detail = (getattr(resp, "text", "") or "")[:200]
    detail = " ".join(detail.split())[:240]
    base = f"Usage limit reached for '{model}'"
    return f"{base}: {detail}" if detail else (
        f"{base}. Free tiers reset daily — try again later, or switch LLM_MODEL to a model "
        "with remaining allowance."
    )


class QuotaExceeded(RuntimeError):
    """The provider refused because a usage allowance is spent, not because we went too fast.

    Separated from a transient rate limit because the correct responses are opposite: a
    per-minute limit should be waited out, a spent daily allowance should fail immediately with
    a clear message. Retrying the latter just makes the user stare at a spinner for minutes
    before an opaque error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _OpenAICompatMessages:
    """Free-tier etiquette lives here: gateways cap requests per MINUTE as well as per day.
    We throttle proactively for `:free` models (cheaper than tripping the limit) and retry
    with backoff on 429s — but only a couple of times.

    Both limits arrive as a 429, so the retry budget is deliberately small: if a short wait
    doesn't clear it, the cause is almost certainly a spent allowance rather than pacing, and
    the useful thing is to say so at once (QuotaExceeded) instead of burning minutes on
    exponential backoff and then surfacing a generic failure."""

    _FREE_INTERVAL_S = 4.0  # floor for `:free` models, ~15 req/min
    _MAX_RETRIES = 4
    _RATE_LIMIT_RETRIES = 2      # 429s only: ~5s + ~10s, then give a straight answer

    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
        )
        self._last_call = 0.0

    def _throttle(self, model: str) -> None:
        # Configured interval (e.g. Gemini free tier ~10 rpm -> 6s), with a floor for
        # OpenRouter `:free` models even when the setting is 0.
        interval = settings.llm_min_interval_s
        if ":free" in model:
            interval = max(interval, self._FREE_INTERVAL_S)
        if interval <= 0:
            return
        wait = interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, Any]]) -> _Response:
        backoff = 10.0
        for attempt in range(self._MAX_RETRIES + 1):
            self._throttle(model)
            try:
                resp = self._client.post(
                    "/chat/completions",
                    json={"model": model, "max_tokens": max_tokens, "messages": messages},
                )
            except httpx.TransportError:  # read timeout / dropped connection: transient
                self._last_call = time.monotonic()
                if attempt < self._MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            self._last_call = time.monotonic()
            if resp.status_code == 429:
                # A short wait clears real pacing limits. If it doesn't, the allowance is spent —
                # say so immediately rather than backing off for minutes into a generic error.
                if attempt < self._RATE_LIMIT_RETRIES:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                raise QuotaExceeded(_quota_message(resp, model))
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:  # OpenRouter can return 200 with an embedded error
                raise RuntimeError(f"LLM gateway error: {data['error']}")
            choice = data["choices"][0]
            # Free/gateway models occasionally return null content (content filter, truncation
            # at max_tokens with reasoning models, transient hiccup). A null here used to crash
            # callers doing .strip(); coerce to "" so a bad completion degrades to an empty
            # answer (→ abstain) or a False judge verdict, never a traceback.
            text = choice.get("message", {}).get("content")
            if text is None and attempt < self._MAX_RETRIES:
                time.sleep(backoff)  # give the model another shot before accepting empty
                backoff *= 2
                continue
            return _Response(
                content=[_TextBlock(text=text or "")],
                usage=data.get("usage"),
            )
        raise RuntimeError("unreachable")  # loop always returns or raises


class OpenAICompatClient:
    """Duck-types the tiny slice of the Anthropic SDK that Sightline uses."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.messages = _OpenAICompatMessages(base_url, api_key)


class _CachedMessages:
    def __init__(self, inner: Any, db_path: Path) -> None:
        self._inner = inner
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            "  key TEXT PRIMARY KEY, model TEXT, response TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )

    @staticmethod
    def _key(model: str, max_tokens: int, messages: list[dict[str, Any]]) -> str:
        blob = json.dumps({"m": model, "t": max_tokens, "msgs": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, Any]]) -> _Response:
        key = self._key(model, max_tokens, messages)
        row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            # Zero-cost usage: a cache hit spends nothing, and the dashboard should show that.
            return _Response(content=[_TextBlock(text=row[0])],
                             usage={"prompt_tokens": 0, "completion_tokens": 0, "cached": True})
        resp = self._inner.create(model=model, max_tokens=max_tokens, messages=messages)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response) VALUES (?, ?, ?)",
                (key, model, resp.content[0].text),
            )
        return resp


class CachedClient:
    """Wraps any messages-style client with the exact-match response cache."""

    def __init__(self, inner: Any, db_path: Path | None = None) -> None:
        self.messages = _CachedMessages(
            inner.messages, db_path or Path(settings.data_dir) / "llm_cache.db"
        )


def make_client(cache: bool = True) -> object:
    """Return an LLM client for the configured provider. Raises if no key is set."""
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY is empty — set it in .env (Anthropic key, or an OpenAI-compatible "
            "gateway key such as OpenRouter with LLM_BASE_URL)."
        )
    if settings.llm_base_url:
        client: object = OpenAICompatClient(settings.llm_base_url, settings.llm_api_key)
    else:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.llm_api_key)
    return CachedClient(client) if cache else client
