"""CI eval gate: fail the build if a retrieval metric regresses past tolerance.

"CI for model behavior" — the same idea as unit tests, applied to eval numbers. A committed
`baseline_metrics.json` is the contract; a code change that drops Recall@5 (or any tracked
metric) by more than `tolerance` below the baseline fails the check, so a silent quality
regression can't merge. Small improvements are free; the baseline is bumped deliberately when
a real gain lands (that commit is the "we got better" record).

Tolerance (default 0.02) absorbs the noise of a 39-case exam without letting real drops through.
Usage:
    python -m sightline.eval.gate --current metrics.json      # compare to committed baseline
    python -m sightline.eval.gate --current m.json --update   # accept current as new baseline
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BASELINE = Path(__file__).parent / "baseline_metrics.json"
_TRACKED = ("recall@5", "ndcg@10", "mrr")
_DEFAULT_TOLERANCE = 0.02


@dataclass
class Regression:
    metric: str
    baseline: float
    current: float

    @property
    def drop(self) -> float:
        return self.baseline - self.current


def check_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    tolerance: float = _DEFAULT_TOLERANCE,
) -> list[Regression]:
    """Return the metrics that fell more than `tolerance` below baseline (empty = gate passes)."""
    out: list[Regression] = []
    for m in _TRACKED:
        if m in baseline and m in current and current[m] < baseline[m] - tolerance:
            out.append(Regression(metric=m, baseline=baseline[m], current=current[m]))
    return out


def load_metrics(path: str | Path) -> dict[str, float]:
    return json.loads(Path(path).read_text())


def _main() -> int:
    import argparse

    from rich.console import Console

    console = Console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="metrics JSON from `eval.run --json`")
    ap.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    ap.add_argument("--update", action="store_true", help="accept current as the new baseline")
    args = ap.parse_args()

    current = load_metrics(args.current)
    if args.update:
        BASELINE.write_text(json.dumps(current, indent=2) + "\n")
        console.print(f"[green]baseline updated[/green] -> {BASELINE.name}: {current}")
        return 0

    baseline = load_metrics(BASELINE)
    regressions = check_regression(current, baseline, args.tolerance)
    if regressions:
        console.print("[bold red]EVAL GATE FAILED — metric regressed past tolerance:[/bold red]")
        for r in regressions:
            console.print(f"  {r.metric}: {r.baseline:.3f} -> {r.current:.3f}  (−{r.drop:.3f})")
        return 1
    console.print("[green]eval gate passed[/green] — no metric regressed past "
                  f"{args.tolerance:.3f} vs baseline {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
