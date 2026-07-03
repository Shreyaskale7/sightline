"""Golden-set schema + loader.

A case pairs a question with (a) the page(s) that should be retrieved and (b) a gold answer,
plus a flag for deliberately-unanswerable questions (to test abstention).

Start by hand-labeling ~30 cases, grow to ~150. Include adversarial slices:
multi-hop, cross-company, chart-only, and unanswerable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RelevantPage:
    accession: str
    page_no: int


@dataclass
class EvalCase:
    id: str
    question: str
    relevant_pages: list[RelevantPage] = field(default_factory=list)
    # Other pages that ALSO state the fact (MD&A repeats the statement figure, etc.).
    # Used only for citation-accuracy grading — citing any of these is a correct citation.
    # Retrieval metrics keep scoring against relevant_pages alone (the canonical target),
    # so recall isn't diluted by alternates.
    also_valid_pages: list[RelevantPage] = field(default_factory=list)
    gold_answer: str | None = None
    answerable: bool = True
    slice: str = "basic"  # basic | multi_hop | cross_company | chart_only | unanswerable


def load_golden_set(path: str | Path) -> list[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text())
    cases: list[EvalCase] = []
    for item in raw["cases"]:
        pages = [RelevantPage(**p) for p in item.get("relevant_pages", [])]
        also = [RelevantPage(**p) for p in item.get("also_valid_pages", [])]
        cases.append(
            EvalCase(
                id=item["id"],
                question=item["question"],
                relevant_pages=pages,
                also_valid_pages=also,
                gold_answer=item.get("gold_answer"),
                answerable=item.get("answerable", True),
                slice=item.get("slice", "basic"),
            )
        )
    return cases
