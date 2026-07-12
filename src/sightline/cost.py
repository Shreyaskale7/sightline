"""Cost accounting + the money-shot metric.

Sightline runs on free models, so its real bill is ~$0. But the *architecture* is what a
hiring manager buys: the two-stage retrieve (cheap text prefilter narrows 1,329 pages to the
top ~5, then the VLM reads only those) is what makes visual QA affordable at all. This module
quantifies that — "our design costs $X/query; naive send-every-page-to-a-VLM costs $Y — a Z%
reduction" — using published prices for a reference PAID model, applied to real token counts.

Two uses:
  - estimate_cost(): score any actual LLM call from its usage dict (spans already capture it).
  - money_shot(): the architectural comparison, independent of which model you actually run.

Prices are USD per 1M tokens; verify against the provider's current pricing page before quoting.
Image-token-per-page is a model-specific estimate for a ~1024px-wide page (tile-based vision).
"""
from __future__ import annotations

from dataclasses import dataclass

# Reference paid vision models (approx list prices, mid-2026 — verify before quoting).
PRICING: dict[str, dict[str, float]] = {
    # per 1M tokens in / out, and estimated vision tokens for one ~1024px page image
    "gpt-4o":          {"in": 2.50, "out": 10.00, "img_per_page": 1105},
    "claude-sonnet":   {"in": 3.00, "out": 15.00, "img_per_page": 1200},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50, "img_per_page": 560},
}

_CHARS_PER_TOKEN = 4  # rough English heuristic for text without a tokenizer


def text_tokens(chars: int) -> int:
    return max(1, chars // _CHARS_PER_TOKEN)


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """USD for one call given token counts. Unknown model -> 0.0 (e.g. our free stack)."""
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (in_tokens * p["in"] + out_tokens * p["out"]) / 1_000_000


@dataclass
class MoneyShot:
    model: str
    corpus_pages: int
    pages_read_naive: int      # naive: VLM reads every candidate page image
    pages_read_sightline: int  # ours: only the top-k the retriever selected
    naive_usd: float
    sightline_usd: float

    @property
    def reduction_pct(self) -> float:
        if self.naive_usd == 0:
            return 0.0
        return 100.0 * (1 - self.sightline_usd / self.naive_usd)


def money_shot(
    model: str = "gpt-4o",
    corpus_pages: int = 1329,
    k: int = 5,
    prompt_text_tokens: int = 400,
    answer_tokens: int = 250,
) -> MoneyShot:
    """Cost of answering ONE question, naive-VLM-over-everything vs two-stage retrieve.

    Naive: with no retriever, a visual system must feed the VLM every page image to be sure it
    saw the answer. Sightline: retrieval (cheap/free text embeddings) picks the top-k, and the
    VLM reads only those k page images. Same VLM, same per-image price — the saving is entirely
    from reading k pages instead of `corpus_pages`.
    """
    p = PRICING[model]
    img = p["img_per_page"]

    def one(pages_read: int) -> float:
        in_tok = prompt_text_tokens + pages_read * img
        return estimate_cost(model, in_tok, answer_tokens)

    return MoneyShot(
        model=model,
        corpus_pages=corpus_pages,
        pages_read_naive=corpus_pages,
        pages_read_sightline=k,
        naive_usd=one(corpus_pages),
        sightline_usd=one(k),
    )
