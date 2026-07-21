"""Deterministic query -> metadata filters (ticker, form type).

Attacks a measured failure, not a hypothetical one: when the corpus grew from 10-Ks only to
10-Ks + 10-Qs, dense Recall@5 fell 0.567 -> 0.313 because a 10-Q income statement embeds
almost identically to the 10-K's. The embedding can't tell WHICH filing a page belongs to —
but the question usually says ("...in its most recent 10-K", "...across its last three
10-Qs", "NVIDIA's..."), and every indexed page carries `ticker` and `form` payload fields.

So: parse company mentions and form mentions out of the question with plain string matching,
and restrict vector search to matching pages. No model, no cost, fully explainable. The M3
upgrade replaces this with proper NER; it has to beat this baseline to ship.

Scope guard: only companies in the corpus are recognized. A mention of an out-of-corpus
company (e.g. Broadcom) yields no filter — retrieval stays corpus-wide and the
answerer/verifier abstains downstream, which is the correct behavior for out-of-domain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Corpus companies: every alias is lowercase; word-boundary matched. Avoid aliases that are
# common English words — e.g. onsemi's ticker "ON" is deliberately NOT an alias (it would match
# "on" in every sentence); its full names are used instead.
_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("nvidia", "nvda"),
    "AMD": ("amd", "advanced micro devices"),
    "INTC": ("intel", "intc"),
    "MU": ("micron", "mu"),
    "QCOM": ("qualcomm", "qcom"),
    "TXN": ("texas instruments", "txn"),
    "ADI": ("analog devices", "adi"),
    "AMAT": ("applied materials", "amat"),
    "LRCX": ("lam research", "lrcx"),
    "KLAC": ("kla", "klac"),
    "NXPI": ("nxp semiconductors", "nxp", "nxpi"),
    "ON": ("onsemi", "on semiconductor", "on semiconductors"),
    "MRVL": ("marvell", "mrvl"),
    "MCHP": ("microchip", "mchp"),
    "SWKS": ("skyworks", "swks"),
}

_FORM_10K = re.compile(r"\b10-?k\b|annual report", re.IGNORECASE)
_FORM_10Q = re.compile(r"\b10-?q\b|quarterly (filing|report)", re.IGNORECASE)
# User-uploaded documents (form="UPLOAD"). Must take precedence: "the uploaded annual report"
# would otherwise trip the 10-K filter and EXCLUDE the very document being asked about.
_FORM_UPLOAD = re.compile(r"\bupload(ed)?\b|\bmy (document|report|file|pdf)\b|\battached\b",
                          re.IGNORECASE)


@dataclass
class QueryFilters:
    tickers: list[str] = field(default_factory=list)
    form: str | None = None  # "10-K" | "10-Q" | None

    @property
    def empty(self) -> bool:
        return not self.tickers and self.form is None


def parse_query_filters(question: str) -> QueryFilters:
    q = question.lower()
    tickers = [
        t for t, aliases in _TICKER_ALIASES.items()
        if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases)
    ]
    form: str | None = None
    if _FORM_UPLOAD.search(question):
        form = "UPLOAD"  # wins over 10-K/10-Q phrasing — see _FORM_UPLOAD comment
    else:
        # If both forms are mentioned, filtering on either would exclude needed pages -> no filter.
        k, qq = bool(_FORM_10K.search(question)), bool(_FORM_10Q.search(question))
        if k != qq:
            form = "10-K" if k else "10-Q"
    return QueryFilters(tickers=tickers, form=form)


def to_qdrant_filter(f: QueryFilters):
    """Build the Qdrant payload filter (None if nothing to filter on)."""
    if f.empty:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must = []
    if f.tickers:
        must.append(FieldCondition(key="ticker", match=MatchAny(any=f.tickers)))
    if f.form:
        must.append(FieldCondition(key="form", match=MatchValue(value=f.form)))
    return Filter(must=must)
