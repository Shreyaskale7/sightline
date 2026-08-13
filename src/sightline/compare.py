"""Cross-company comparison: one cited cell per company, assembled into a table.

This is the capability a chat window structurally cannot match. "Compare R&D spend across all
15 companies" spans ~2,353 pages — far past what fits in a context window — and if a model
guessed at 15 figures you would have no way to tell which ones were wrong.

So the work is decomposed the whole way down, not just at retrieval:

    for each company:
        retrieve from THAT company's filings only   (metadata filter, so no cross-talk)
        answer one cell from those pages            (a value, not a paragraph)
        verify the citation against what we retrieved
    assemble the rows

Two properties fall out of that shape, and they are the point:
  - **Scale.** Nothing has to fit in one context window; adding companies adds rows, not tokens
    per call. The corpus can grow past any model's context limit.
  - **Checkability.** Every cell carries its own page citation, so a wrong figure is findable
    and refutable instead of buried mid-paragraph. A cell with no support abstains on its own,
    rather than dragging the whole answer down.

Cost is the honest trade: this is one LLM call per company, so it is deliberately capped and the
caller is told what it will spend before it spends it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .answerer import Answerer, Citation
from .observability import span
from .retrieval.filters import QueryFilters
from .verify import verify

# A table is a comparison, not a corpus scan: past a dozen or so companies the LLM cost stops
# being worth it and the table stops being readable. Callers can lower this, not raise it.
MAX_COMPANIES = 12


@dataclass
class ComparisonRow:
    ticker: str
    value: str = ""                                   # the cell text, e.g. "$8,675 million"
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False                           # corpus can't support this cell
    error: str = ""                                   # per-row failure (quota, network)

    @property
    def ok(self) -> bool:
        return bool(self.value) and not self.abstained and not self.error


@dataclass
class ComparisonResult:
    question: str
    rows: list[ComparisonRow] = field(default_factory=list)
    truncated_to: int = 0        # >0 if the company list was capped

    @property
    def answered(self) -> int:
        return sum(1 for r in self.rows if r.ok)


def resolve_companies(
    question: str,
    all_tickers: list[str],
    named: list[str] | None = None,
    limit: int = MAX_COMPANIES,
) -> tuple[list[str], int]:
    """Decide which companies a comparison covers.

    If the question names companies, compare exactly those. If it names none ("compare R&D
    across every company"), fall back to the whole corpus — the set has to come from what is
    actually indexed, since the question didn't say. Returns (tickers, truncated_to).
    """
    chosen = list(named or [])
    if len(chosen) < 2:
        chosen = list(all_tickers)
    if len(chosen) > limit:
        return chosen[:limit], len(chosen)
    return chosen, 0


def compare(
    question: str,
    tickers: list[str],
    retriever,
    store,
    answerer: Answerer | None = None,
    k: int = 5,
) -> ComparisonResult:
    """Answer `question` once per company and return the assembled rows.

    `retriever` needs .retrieve(query, k, filter_override); `store` needs .get_page(). Both are
    injected so this is testable with fakes — no Qdrant, no models, no API spend.
    """
    answerer = answerer or Answerer()
    result = ComparisonResult(question=question)

    with span("compare", n_companies=len(tickers)) as s:
        for ticker in tickers:
            row = ComparisonRow(ticker=ticker)
            try:
                # Scope retrieval to this company only: the filter is what stops one company's
                # statement page from answering another company's cell.
                hits = retriever.retrieve(
                    question, k=k, filter_override=QueryFilters(tickers=[ticker])
                )
                pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
                cell = answerer.answer_cell(question, ticker, pages)
                # Same guarantee as the single-answer path: a citation pointing at a page we
                # never retrieved is fabricated, and a cell with nothing valid left abstains.
                verdict = verify(cell, {(p.accession, p.page_no) for p in pages})
                cell = verdict.result
                if cell.abstained:
                    row.abstained = True
                else:
                    row.value = _strip_tags(cell.answer)
                    row.citations = cell.citations
            except Exception as e:  # one company's failure must not lose the whole table
                row.error = f"{type(e).__name__}"
            result.rows.append(row)
        s["answered"] = result.answered
    return result


def _strip_tags(text: str) -> str:
    """Remove the inline [p:acc#page] tags from a cell — the citation is rendered as a chip
    next to the value, so repeating it in the text would just be noise."""
    from .answerer import _CITE_RE

    return _CITE_RE.sub("", text).strip().rstrip(".").strip()
