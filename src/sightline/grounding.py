"""Numeric grounding: does the cited page actually contain the figure the answer states?

The existing verifier (verify.py) proves a citation points at a page we retrieved. That closes
one hallucination mode but leaves a nastier one open: **right page, wrong number.** The model
cites the income statement — genuinely the correct source — and states a figure that isn't on
it. Every existing check passes, and the answer looks maximally trustworthy precisely because
the citation is real.

So each figure in the answer is looked for in the pages it cites. Matching is
formatting-tolerant, because a faithful answer legitimately rephrases: "60,922" may be written
$60,922 million, 60922, or ~$60.9 billion, and none of those is a hallucination.

Two findings come out of it:
  - **unsupported** — a figure appears in no cited page. Grounds for distrust.
  - **disagreement** — the cited pages carry conflicting values for the same quantity, which an
    analyst should be told about rather than have silently resolved.

Deliberately advisory, not enforcing: this reports, it does not rewrite answers. Numeric
extraction from prose is heuristic, and silently deleting a correct claim would be worse than
flagging a suspicious one. verify.py stays the hard gate; this is the microscope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Figures worth checking: 3+ significant digits, or any decimal. Small integers ("three
# segments", "38 countries") are prose, not claims a citation is expected to carry.
_FIGURE = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,})")
_UNIT = re.compile(r"\b(billion|million|thousand|bn|mm?)\b", re.IGNORECASE)


@dataclass
class FigureCheck:
    text: str                      # the figure as written in the answer, e.g. "60,922"
    value: float
    supported_by: list[tuple[str, int]] = field(default_factory=list)  # (accession, page)

    @property
    def supported(self) -> bool:
        return bool(self.supported_by)


@dataclass
class GroundingReport:
    figures: list[FigureCheck] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)

    @property
    def unsupported(self) -> list[FigureCheck]:
        return [f for f in self.figures if not f.supported]

    @property
    def all_supported(self) -> bool:
        return all(f.supported for f in self.figures)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def extract_figures(text: str) -> list[tuple[str, float]]:
    """Figures *claimed* in a piece of text, as (as-written, numeric value).

    Citation tags are stripped first. An accession is a long digit string — "[p:0001045810-26-
    000021#51]" reads as the figures 0001045810, 000021 and 51 — so leaving tags in makes every
    correctly-cited answer look like it states numbers no page supports. The tag is provenance
    metadata, not a claim about the world.
    """
    from .answerer import _CITE_RE

    text = _CITE_RE.sub(" ", text or "")
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for m in _FIGURE.finditer(text):
        raw = m.group(1)
        v = _to_float(raw)
        if v is None or raw in seen:
            continue
        seen.add(raw)
        out.append((raw, v))
    return out


def _page_contains(value: float, page_text: str) -> bool:
    """Is `value` present in the page, allowing the formatting a faithful answer may use?

    Accepts the plain and comma forms, and unit rescaling in both directions — an answer saying
    "$60.9 billion" is grounded by a statement line reading "60,922" (millions), and vice versa.
    """
    candidates = {round(value, 2)}
    for factor in (1_000, 1_000_000):
        candidates.add(round(value * factor, 2))
        candidates.add(round(value / factor, 2))

    page_values = {v for _, v in extract_figures(page_text)}
    for c in candidates:
        for pv in page_values:
            if pv == c:
                return True
            # Tolerate rounding in the answer: 60,922 -> "60.9" (0.1% of the larger magnitude).
            if c and pv and abs(pv - c) <= max(abs(pv), abs(c)) * 0.001:
                return True
    return False


def check_answer(answer: str, cited_pages: dict[tuple[str, int], str]) -> GroundingReport:
    """Check every figure in `answer` against the text of the pages it cites.

    `cited_pages` maps (accession, page_no) -> page text, and should contain ONLY the pages the
    answer actually cited — checking against pages it didn't cite would credit it for evidence
    it never pointed at.
    """
    report = GroundingReport()
    for raw, value in extract_figures(answer):
        check = FigureCheck(text=raw, value=value)
        for key, text in cited_pages.items():
            if _page_contains(value, text):
                check.supported_by.append(key)
        report.figures.append(check)

    report.disagreements = _find_disagreements(cited_pages)
    return report


def _find_disagreements(cited_pages: dict[tuple[str, int], str]) -> list[str]:
    """Flag a metric whose cited pages carry conflicting values.

    Scoped to labelled statement lines rather than every number on the page: two pages listing
    different figures under the same label is a real conflict worth surfacing, whereas two pages
    merely containing different numbers is just two pages.
    """
    labels = ("total revenue", "net revenue", "net sales", "net income",
              "research and development", "gross profit", "operating income")
    by_label: dict[str, set[float]] = {}
    where: dict[str, list[tuple[str, int]]] = {}

    for key, text in cited_pages.items():
        low = (text or "").lower()
        for label in labels:
            i = low.find(label)
            if i == -1:
                continue
            figs = extract_figures(low[i : i + 120])   # the value sits just after its label
            if not figs:
                continue
            by_label.setdefault(label, set()).add(figs[0][1])
            where.setdefault(label, []).append(key)

    out = []
    for label, values in by_label.items():
        if len(values) > 1 and len(where[label]) > 1:
            shown = ", ".join(f"{v:,.0f}" for v in sorted(values))
            pages = ", ".join(f"{a}#{p}" for a, p in where[label])
            out.append(f"cited pages disagree on '{label}': {shown} (from {pages})")
    return out
