"""Answerer tests — run with a FAKE client, so no network, no API key, no cost.

The logic worth testing is ours, not Claude's: prompt assembly, citation parsing, and
abstention handling. The injectable-client design exists exactly for this.
"""
from __future__ import annotations

from dataclasses import dataclass

from sightline.answerer import Answerer, build_prompt, parse_answer


@dataclass
class _Page:
    accession: str
    page_no: int
    text: str
    ticker: str = "NVDA"
    form: str = "10-K"


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class _Block:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Resp:
            def __init__(self, text: str) -> None:
                self.content = [_Block(text)]

        return _Resp(self._reply)


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.messages = _FakeMessages(reply)


def test_parse_citations_dedupe_and_order():
    raw = ("Revenue was $215,938M [p:0001045810-26-000021#51] and R&D was $18,497M "
           "[p:0001045810-26-000021#51] vs Intel [p:0000050863-26-000011#72].")
    r = parse_answer(raw)
    assert not r.abstained
    assert [(c.accession, c.page_no) for c in r.citations] == [
        ("0001045810-26-000021", 51),
        ("0000050863-26-000011", 72),
    ]


def test_parse_abstain_tolerates_punctuation():
    for raw in ("ABSTAIN", "abstain.", "ABSTAIN\n"):
        r = parse_answer(raw)
        assert r.abstained and r.citations == []


def test_answer_with_no_pages_abstains_without_api_call():
    a = Answerer(client=_FakeClient("should never be called"))
    r = a.answer("anything", pages=[])
    assert r.abstained
    assert a._client.messages.last_kwargs is None  # no tokens spent


def test_answer_end_to_end_with_fake_client():
    page = _Page("0001045810-26-000021", 51, "Revenue $215,938")
    fake = _FakeClient("Revenue was $215,938 million [p:0001045810-26-000021#51].")
    r = Answerer(model="fake-model", client=fake).answer("What was revenue?", [page])
    assert not r.abstained
    assert r.citations[0].page_no == 51
    # The prompt actually contained our page and its citation tag:
    sent = fake.messages.last_kwargs["messages"][0]["content"]
    assert "[p:0001045810-26-000021#51]" in sent and "Revenue $215,938" in sent


def test_build_prompt_truncates_huge_pages():
    page = _Page("x-1", 1, "A" * 100_000)
    prompt = build_prompt("q", [page])
    assert len(prompt) < 20_000  # the 6k/page cap held
