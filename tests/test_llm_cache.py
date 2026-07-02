"""Response-cache tests: a hit must not call the provider; different prompts must not collide."""
from __future__ import annotations

from sightline.llm import CachedClient, _Response, _TextBlock


class _CountingMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, model, max_tokens, messages):
        self.calls += 1
        return _Response(content=[_TextBlock(text=f"reply-{self.calls}")])


class _FakeInner:
    def __init__(self) -> None:
        self.messages = _CountingMessages()


def _mk(tmp_path):
    inner = _FakeInner()
    return inner, CachedClient(inner, db_path=tmp_path / "cache.db")


def test_cache_hit_skips_provider(tmp_path):
    inner, client = _mk(tmp_path)
    msgs = [{"role": "user", "content": "q1"}]
    r1 = client.messages.create(model="m", max_tokens=10, messages=msgs)
    r2 = client.messages.create(model="m", max_tokens=10, messages=msgs)
    assert r1.content[0].text == r2.content[0].text == "reply-1"
    assert inner.messages.calls == 1  # second call was free


def test_different_prompts_or_models_miss(tmp_path):
    inner, client = _mk(tmp_path)
    client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "q1"}])
    client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "q2"}])
    client.messages.create(model="m2", max_tokens=10, messages=[{"role": "user", "content": "q1"}])
    assert inner.messages.calls == 3


def test_cache_persists_across_clients(tmp_path):
    inner1, client1 = _mk(tmp_path)
    msgs = [{"role": "user", "content": "q"}]
    client1.messages.create(model="m", max_tokens=10, messages=msgs)

    inner2 = _FakeInner()
    client2 = CachedClient(inner2, db_path=tmp_path / "cache.db")  # same DB, new process-alike
    r = client2.messages.create(model="m", max_tokens=10, messages=msgs)
    assert r.content[0].text == "reply-1"
    assert inner2.messages.calls == 0  # served entirely from disk
