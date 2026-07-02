"""M1 answerer: turn retrieved pages + a question into a cited answer (or an abstention).

How it works, in plain terms:
  1. The retriever finds the top-k most relevant pages.
  2. We paste those pages' text into a prompt for Claude, with strict rules:
     answer ONLY from the provided pages, tag every claim with the page it came from,
     and if the pages don't contain the answer, say ABSTAIN — never guess.
  3. We parse the citation tags out of the reply so the rest of the system (eval, API, UI)
     gets structured (accession, page_no) citations, not just prose.

M1 scope note: this feeds Claude the extracted page TEXT, because M1 is the deliberately-dumb
text baseline we measure everything against. In M2 the answer path switches to page IMAGES
(the whole point of Sightline); the hard constraint "text never feeds answers" applies M2+.

Abstention is a feature, not a failure: for questions the corpus can't support, the correct
behavior is a refusal. The eval set contains deliberately-unanswerable questions to check this.

The Anthropic client is injected (constructor arg) so tests can pass a fake — the parsing and
abstention logic get tested without network calls or API spend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

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


class PageLike(Protocol):
    """Anything with page identity + text (e.g. ingest.store.StoredPage)."""

    accession: str
    page_no: int
    text: str
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
            from anthropic import Anthropic  # lazy: keeps import-time light

            if not settings.llm_api_key:
                raise RuntimeError(
                    "LLM_API_KEY is empty. Get a key at console.anthropic.com and put it in .env."
                )
            self._client = Anthropic(api_key=settings.llm_api_key)

    def answer(self, question: str, pages: Sequence[PageLike]) -> AnswerResult:
        """One retrieval-grounded answer. No pages -> abstain without spending a token."""
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
