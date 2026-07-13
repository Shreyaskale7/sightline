"""LLM-judge calibration: measure judge–human agreement with Cohen's kappa.

An LLM judge's verdicts are only as trustworthy as their agreement with a human's. The
discipline (rarely done, high signal): sample the judge's verdicts, have a human label the
same items blind, and report Cohen's κ — agreement corrected for chance. κ ≥ 0.8 = strong;
0.6–0.8 = substantial; below that, fix the judge prompt before believing its numbers.

Workflow:
  1. `python -m sightline.eval.calibration --export labels.csv`
       runs the (cached) answer pipeline over prose-judged golden cases and writes one row per
       verdict: question, gold answer, system answer, judge verdict, and an EMPTY human column.
  2. A human fills the `human` column (1 = correct, 0 = incorrect) without peeking at `judge`.
  3. `python -m sightline.eval.calibration --score labels.csv`
       prints raw agreement and Cohen's κ.

Numeric-graded cases are excluded: they're deterministic string math, not judgment — κ only
measures the part where the LLM actually judges.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two binary raters. 1.0 = perfect, 0.0 = chance-level agreement.

    κ = (p_o − p_e) / (1 − p_e), where p_o is observed agreement and p_e is the agreement
    expected by chance from each rater's label marginals.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if p_e == 1.0:  # both raters constant and identical — agreement is trivially perfect
        return 1.0
    return (p_o - p_e) / (1 - p_e)


@dataclass
class KappaReport:
    n: int
    raw_agreement: float
    kappa: float

    @property
    def reading(self) -> str:
        k = self.kappa
        if k >= 0.8:
            return "strong — judge verdicts are trustworthy"
        if k >= 0.6:
            return "substantial — usable, note the caveat"
        if k >= 0.4:
            return "moderate — improve the judge prompt before trusting"
        return "weak — do NOT trust this judge's numbers"


def score_labels(csv_path: str | Path) -> KappaReport:
    """Compute κ from a filled labels CSV (columns: judge, human; 0/1)."""
    judge: list[int] = []
    human: list[int] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = (row.get("human") or "").strip()
            if h not in ("0", "1"):
                continue  # unlabeled row
            judge.append(int(row["judge"]))
            human.append(int(h))
    if not judge:
        raise SystemExit("no labeled rows — fill the `human` column with 0/1 first")
    n = len(judge)
    raw = sum(1 for j, h in zip(judge, human) if j == h) / n
    return KappaReport(n=n, raw_agreement=raw, kappa=cohens_kappa(judge, human))


def export_for_labeling(csv_path: str | Path, k: int = 5) -> int:
    """Run the pipeline over prose-judged cases; write verdict rows for blind human labeling."""
    from sightline.answerer import Answerer
    from sightline.config import settings
    from sightline.ingest.store import MetadataStore
    from sightline.retrieval.routed import RoutedRetriever

    from .dataset import load_golden_set
    from .judge import Judge, numeric_match
    from .run import GOLDEN

    cases = [c for c in load_golden_set(GOLDEN) if c.answerable]
    rows: list[dict] = []
    retriever = RoutedRetriever()
    store = MetadataStore(settings.data_dir / "sightline.db")
    answerer = Answerer()
    judge = Judge()
    try:
        for c in cases:
            hits = retriever.retrieve(c.question, k=k)
            pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
            result = answerer.answer(c.question, pages)  # cache makes re-runs free
            if result.abstained:
                continue
            if numeric_match(c.gold_answer or "", result.answer) is not None:
                continue  # deterministic grade — not the judge's judgment
            verdict = judge.is_correct(c.question, c.gold_answer or "", result.answer)
            rows.append({
                "id": c.id, "question": c.question, "gold": c.gold_answer,
                "answer": result.answer, "judge": int(verdict), "human": "",
            })
    finally:
        retriever.close()
        store.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "gold", "answer", "judge", "human"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    import argparse

    from rich.console import Console

    console = Console()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", metavar="CSV", help="write judge verdicts for blind labeling")
    g.add_argument("--score", metavar="CSV", help="compute kappa from a filled CSV")
    args = ap.parse_args()

    if args.export:
        n = export_for_labeling(args.export)
        console.print(f"[green]wrote {n} verdicts[/green] -> {args.export}\n"
                      "Fill the `human` column (1=correct, 0=incorrect) WITHOUT looking at "
                      "`judge`, then run --score.")
    else:
        r = score_labels(args.score)
        console.print(f"n={r.n}  raw agreement={r.raw_agreement:.2f}  "
                      f"[bold]Cohen's κ={r.kappa:.3f}[/bold]  ({r.reading})")
