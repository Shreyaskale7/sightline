"""Corpus freshness: which filings exist at the SEC that we haven't ingested yet.

A chat session forgets your documents the moment it ends. An indexed corpus doesn't — but it
does go stale, silently, which is worse: the system keeps answering confidently from last
quarter's numbers with no indication anything newer exists.

So freshness is made explicit and checkable. This module answers one question — "what has been
filed that we don't have?" — by diffing the SEC's filing list against what the store has
ingested. It never guesses from dates; a filing counts as held only if its accession is actually
in the store, which is the same accession-keyed idempotency the ingest path uses.

Detection is deliberately separate from ingestion: knowing you're stale is cheap and safe to run
often, while ingesting is slow and side-effectful. Scheduling belongs to the deployment (a cron
hitting the endpoint), not in here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ingest.edgar import Filing
from .observability import span


@dataclass
class NewFiling:
    ticker: str
    form: str
    accession: str
    filing_date: str

    @classmethod
    def of(cls, f: Filing) -> "NewFiling":
        return cls(ticker=f.ticker, form=f.form, accession=f.accession,
                   filing_date=f.filing_date)


@dataclass
class FreshnessReport:
    checked: list[str] = field(default_factory=list)      # tickers actually checked
    new_filings: list[NewFiling] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # ticker -> failure reason

    @property
    def is_stale(self) -> bool:
        return bool(self.new_filings)


def find_new_filings(
    client,
    store,
    tickers: list[str],
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    limit_per_ticker: int = 4,
) -> FreshnessReport:
    """Filings the SEC lists for these tickers that the store hasn't ingested.

    `client` needs .ticker_to_cik() and .get_filings(); `store` needs .is_ingested() — both
    injected so this is testable without touching sec.gov.

    One ticker failing (unknown symbol, network blip) must not hide the rest of the report, so
    failures are collected per ticker rather than raised.
    """
    report = FreshnessReport()
    with span("watch.check", n_tickers=len(tickers)) as s:
        try:
            cik_map = client.ticker_to_cik()
        except Exception as e:
            report.errors["*"] = f"{type(e).__name__}"
            s["error"] = report.errors["*"]
            return report

        for ticker in tickers:
            cik = cik_map.get(ticker.upper())
            if cik is None:
                report.errors[ticker] = "unknown ticker at SEC"
                continue
            try:
                filings = client.get_filings(
                    cik, ticker.upper(), forms=forms, limit=limit_per_ticker
                )
            except Exception as e:
                report.errors[ticker] = f"{type(e).__name__}"
                continue
            report.checked.append(ticker.upper())
            for f in filings:
                # Accession is the idempotency key everywhere else, so it decides here too.
                if not store.is_ingested(f.accession):
                    report.new_filings.append(NewFiling.of(f))

        s["new"] = len(report.new_filings)
        s["errors"] = len(report.errors)
    return report
