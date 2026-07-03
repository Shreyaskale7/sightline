"""Answerer: turn retrieved pages + a question into a cited answer (or an abstention).

Two answer modes share one citation protocol:
  - answer(...)             — M1 text baseline: pastes page TEXT into the prompt.
  - answer_from_images(...) — M2 visual path: sends the page IMAGES themselves; the model
    reads tables/charts/layout directly. Each image is preceded by a text label carrying its
    [p:accession#page] tag (a model can't read an accession out of pixels), so citations work
    identically in both modes.

Rules in both: answer ONLY from the provided pages, tag every claim, say ABSTAIN otherwise.
Abstention is a feature, not a failure: for questions the corpus can't support, the correct
behavior is a refusal. The eval set contains deliberately-unanswerable questions to check it.

Cost note for the visual mode: images are token-heavy, so pages are downscaled to ~1024px
width and capped at `max_images` (default 3). The response cache in llm.py applies to both
modes (image bytes are part of the cache key via the message hash).

The LLM client is injected (constructor arg) so tests can pass a fake — parsing, prompt
assembly, and abstention logic are all tested without network calls or API spend.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import settings
from .observability import span

# The model must tag claims like: [p:0001045810-26-000021#51]
_CITE_RE = re.compile(r"\[p:([0-9-]+)#(\d+)\]")
_ABSTAIN_TOKEN = "ABSTAIN"

# Cap how much of a page we paste into the prompt. Filing pages are ~2-5k chars; the cap
# guards against pathological pages blowing up cost. (Cost discipline is a hard constraint.)
_MAX_PAGE_CHARS = 6000

_PROMPT = """\
You are a careful financial-filings analyst. Answer the question using ONLY the numbered
filing pages provided below. Rules:

1. Every factual claim in your answer MUST be followed by a citation tag of the exact form
   [p:ACCESSION#PAGE] copied from the page header it came from.
2. Use only what the pages say. Do not use outside knowledge, and do not extrapolate.
3. If the provided pages do not contain the information needed to answer, reply with the
   single word {abstain} and nothing else.
4. Be concise: one to three sentences.

{pages}

Question: {question}
"""

_VISUAL_INSTRUCTIONS = """\
You are a careful financial-filings analyst. Answer the question using ONLY the filing page
IMAGES provided below. Each image is preceded by a label of the form [p:ACCESSION#PAGE].
Rules:

1. Every factual claim in your answer MUST be followed by the citation tag (exact form
   [p:ACCESSION#PAGE]) of the page image it came from.
2. Use only what the page images show. Do not use outside knowledge, and do not extrapolate.
3. If the page images do not contain the information needed to answer, reply with the single
   word {abstain} and nothing else.
4. Be concise: one to three sentences.
"""

_MAX_IMAGE_WIDTH = 1024  # downscale before sending: filings stay legible, tokens stay sane


class PageLike(Protocol):
    """Anything with page identity + text (e.g. ingest.store.StoredPage)."""

    accession: str
    page_no: int
    text: str
    ticker: str
    form: str


class ImagePageLike(Protocol):
    """A page with an image on disk (e.g. ingest.store.StoredPage)."""

    accession: str
    page_no: int
    image_path: Path
    ticker: str
    form: str


@dataclass
class Citation:
    accession: str
    page_no: int


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False


def build_prompt(question: str, pages: Sequence[PageLike]) -> str:
    """Assemble the prompt. Pure function -> easy to unit-test and to version later (M4)."""
    blocks = []
    for p in pages:
        header = f"--- PAGE [p:{p.accession}#{p.page_no}] ({p.ticker} {p.form}) ---"
        blocks.append(f"{header}\n{p.text[:_MAX_PAGE_CHARS]}")
    return _PROMPT.format(abstain=_ABSTAIN_TOKEN, pages="\n\n".join(blocks), question=question)


def _encode_image(path: Path, max_width: int = _MAX_IMAGE_WIDTH) -> str:
    """Load a page PNG, downscale to max_width, return a base64 data URL."""
    from PIL import Image

    img = Image.open(path)
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_visual_content(question: str, pages: Sequence[ImagePageLike]) -> list[dict[str, Any]]:
    """Assemble the multimodal message content (OpenAI-compatible format): instructions,
    then per page a citation label + the image, then the question. Pure-ish (reads images
    from disk) and unit-testable with tiny PNGs."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _VISUAL_INSTRUCTIONS.format(abstain=_ABSTAIN_TOKEN)}
    ]
    for p in pages:
        content.append({
            "type": "text",
            "text": f"[p:{p.accession}#{p.page_no}] ({p.ticker} {p.form})",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image(Path(p.image_path))},
        })
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


def parse_answer(raw: str) -> AnswerResult:
    """Turn the model's reply into a structured result. Pure function, unit-tested."""
    text = raw.strip()
    # Tolerate minor deviation ("ABSTAIN." etc.) — an abstention should never be lost to
    # formatting noise, because a lost abstention becomes a silent wrong answer downstream.
    if text.upper().startswith(_ABSTAIN_TOKEN):
        return AnswerResult(answer="I don't know — the indexed filings don't contain this.",
                            citations=[], abstained=True)
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for acc, page in _CITE_RE.findall(text):
        key = (acc, int(page))
        if key not in seen:  # dedupe, keep first-mention order
            seen.add(key)
            citations.append(Citation(accession=acc, page_no=int(page)))
    return AnswerResult(answer=text, citations=citations, abstained=False)


class Answerer:
    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        self.model = model or settings.llm_model
        self._client = client  # injected fake in tests; real Anthropic client otherwise

    def _ensure_client(self) -> None:
        if self._client is None:
            from .llm import make_client  # lazy: keeps import-time light

            self._client = make_client()

    def answer(self, question: str, pages: Sequence[PageLike]) -> AnswerResult:
        """One retrieval-grounded answer from page TEXT (M1 baseline mode)."""
        if not pages:
            return AnswerResult(answer="I don't know — nothing relevant was retrieved.",
                                citations=[], abstained=True)
        self._ensure_client()
        prompt = build_prompt(question, pages)
        with span("answer", model=self.model, n_pages=len(pages)) as s:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            result = parse_answer(resp.content[0].text)
            s["abstained"] = result.abstained
            s["n_citations"] = len(result.citations)
        return result

    def answer_from_images(
        self, question: str, pages: Sequence[ImagePageLike], max_images: int = 3
    ) -> AnswerResult:
        """One retrieval-grounded answer from page IMAGES (M2 mode — the point of Sightline).

        Uses the VLM model from config and the OpenAI-compatible multimodal format (works via
        OpenRouter/Gemini gateways; direct-Anthropic support lands with M4's provider work).
        Caps images because vision tokens are the expensive part.
        """
        if not pages:
            return AnswerResult(answer="I don't know — nothing relevant was retrieved.",
                                citations=[], abstained=True)
        self._ensure_client()
        content = build_visual_content(question, pages[:max_images])
        model = settings.vlm_model or self.model
        with span("answer_visual", model=model, n_images=min(len(pages), max_images)) as s:
            resp = self._client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": content}],
            )
            result = parse_answer(resp.content[0].text)
            s["abstained"] = result.abstained
            s["n_citations"] = len(result.citations)
        return result
