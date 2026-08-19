"""A spent allowance must fail fast and say so.

Both a per-minute rate limit and a spent daily allowance arrive as a 429, but the right responses
are opposite: wait out the former, stop immediately on the latter. Retrying a spent allowance with
exponential backoff leaves a user staring at a spinner for minutes before an opaque error.
"""
from __future__ import annotations

import httpx
import pytest

from sightline.llm import QuotaExceeded, _OpenAICompatMessages, _quota_message


class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def _messages(responses):
    m = _OpenAICompatMessages.__new__(_OpenAICompatMessages)   # skip the httpx client
    m._last_call = 0.0
    m._client = type("C", (), {"post": lambda self, *a, **k: responses.pop(0)})()
    return m


def test_persistent_429_raises_quota_exceeded(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)          # don't actually wait in tests
    m = _messages([_Resp(429, {"error": {"message": "free-tier daily limit"}})] * 5)
    with pytest.raises(QuotaExceeded) as ei:
        m.create(model="x:free", max_tokens=10, messages=[])
    assert "daily limit" in str(ei.value)


def test_transient_429_still_recovers(monkeypatch):
    """A real pacing limit clears after a short wait — that must not become a hard failure."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    ok = _Resp(200, {"choices": [{"message": {"content": "hello"}}], "usage": {}})
    m = _messages([_Resp(429, {"error": {"message": "slow down"}}), ok])
    out = m.create(model="x:free", max_tokens=10, messages=[])
    assert out.content[0].text == "hello"


def test_retry_budget_is_small(monkeypatch):
    """Bounded retries are the point: the wait before giving a straight answer stays short."""
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    m = _messages([_Resp(429, {"error": {"message": "limit"}})] * 5)
    with pytest.raises(QuotaExceeded):
        m.create(model="x:free", max_tokens=10, messages=[])
    assert sum(slept) <= 30      # seconds, not minutes


def test_message_prefers_the_providers_own_words():
    assert "over quota" in _quota_message(_Resp(429, {"error": {"message": "over quota"}}), "m")
    # Unparseable body must still yield something actionable, never an empty error.
    fallback = _quota_message(_Resp(429, None, text=""), "my-model")
    assert "my-model" in fallback and "reset" in fallback.lower()
