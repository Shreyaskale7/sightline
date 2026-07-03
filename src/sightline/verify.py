"""Verifier (M3 floor): no uncited answer leaves the system.

The deterministic half of verification — no LLM, no cost, no ambiguity:
  1. A non-abstained answer with ZERO citations is worthless in this domain -> force abstain.
  2. A citation pointing at a page we never retrieved is fabricated (the model can only have
     read the pages we gave it) -> strip it; if nothing valid remains -> force abstain.

This is what turns "abstention is a feature" from a prompt suggestion into a guarantee.
The M3 upgrade adds an LLM pass on top (does the cited page actually SUPPORT each claim?);
that lands with judge calibration, because an uncalibrated support-checker is just vibes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .answerer import AnswerResult

_ABSTAIN_ANSWER = "I don't know — the answer could not be verified against the retrieved pages."


@dataclass
class Verdict:
    result: AnswerResult          # possibly downgraded to an abstention
    forced_abstain: bool = False
    dropped_citations: int = 0
    reasons: list[str] = field(default_factory=list)


def verify(result: AnswerResult, retrieved_pages: set[tuple[str, int]]) -> Verdict:
    """Check an answer against the set of (accession, page_no) pages actually retrieved."""
    if result.abstained:
        return Verdict(result=result)  # an abstention needs no defense

    valid = [c for c in result.citations if (c.accession, c.page_no) in retrieved_pages]
    dropped = len(result.citations) - len(valid)
    reasons = []
    if dropped:
        reasons.append(f"{dropped} citation(s) referenced pages that were never retrieved")

    if not valid:
        reasons.append("no valid citations — uncited claims don't ship")
        return Verdict(
            result=AnswerResult(answer=_ABSTAIN_ANSWER, citations=[], abstained=True),
            forced_abstain=True,
            dropped_citations=dropped,
            reasons=reasons,
        )

    if dropped:
        return Verdict(
            result=AnswerResult(answer=result.answer, citations=valid, abstained=False),
            dropped_citations=dropped,
            reasons=reasons,
        )
    return Verdict(result=result)
