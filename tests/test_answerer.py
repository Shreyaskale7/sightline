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


# --- visual answer path (M2) --------------------------------------------------

@dataclass
class _ImagePage:
    accession: str
    page_no: int
    image_path: object
    ticker: str = "NVDA"
    form: str = "10-K"


def _png(tmp_path, name="page.png", size=(1400, 1800)):
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", size, "white").save(p)
    return p


def test_build_visual_content_labels_then_images(tmp_path):
    from sightline.answerer import build_visual_content

    page = _ImagePage("0001045810-26-000021", 51, _png(tmp_path))
    content = build_visual_content("What was revenue?", [page])
    kinds = [c["type"] for c in content]
    assert kinds == ["text", "text", "image_url", "text"]  # instructions, label, image, question
    assert "[p:0001045810-26-000021#51]" in content[1]["text"]
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1]["text"].endswith("What was revenue?")


def test_visual_images_are_downscaled(tmp_path):
    import base64
    import io

    from PIL import Image

    from sightline.answerer import build_visual_content

    page = _ImagePage("a", 1, _png(tmp_path, size=(2048, 2600)))
    content = build_visual_content("q", [page])
    b64 = content[2]["image_url"]["url"].split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.width == 1024  # capped


def test_answer_from_images_end_to_end_and_cap(tmp_path):
    acc = "0001045810-26-000021"  # citation tags only match real accession formats
    pages = [_ImagePage(acc, i, _png(tmp_path, f"p{i}.png")) for i in range(1, 6)]
    fake = _FakeClient(f"Revenue was $215,938 million [p:{acc}#1].")
    r = Answerer(model="fake-vlm", client=fake).answer_from_images("q", pages, max_images=3)
    assert not r.abstained and r.citations[0].page_no == 1
    sent = fake.messages.last_kwargs["messages"][0]["content"]
    assert sum(1 for c in sent if c["type"] == "image_url") == 3  # cap held


def test_answer_from_images_no_pages_abstains():
    a = Answerer(client=_FakeClient("never called"))
    r = a.answer_from_images("q", [])
    assert r.abstained and a._client.messages.last_kwargs is None
