"""SEC EDGAR client.

EDGAR is free and needs no API key, but it has two hard rules:
  1. Every request must send a User-Agent header with a real name + email.
  2. Stay under 10 requests/second across all sec.gov domains.
This module enforces both. Do not remove the rate limiter.

Key endpoints (all under https):
  - Ticker -> CIK map:   www.sec.gov/files/company_tickers.json
  - All filings for CIK: data.sec.gov/submissions/CIK##########.json  (10-digit zero-padded)
  - Filing documents:    www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx

from ..config import settings

_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_DIR = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"


class _RateLimiter:
    """Simple global limiter: never exceed `max_per_sec` requests. Thread-safe."""

    def __init__(self, max_per_sec: float = 8.0) -> None:  # 8 < 10 for safety margin
        self._min_interval = 1.0 / max_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


@dataclass
class Filing:
    cik: int
    ticker: str
    form: str              # e.g. "10-K", "10-Q"
    filing_date: str       # ISO date
    accession: str         # e.g. "0000320193-24-000123"
    primary_document: str  # filename of the primary doc

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def primary_url(self) -> str:
        return _ARCHIVE_DIR.format(cik=self.cik, acc_nodash=self.accession_nodash) + self.primary_document


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        ua = user_agent or settings.sec_user_agent
        if "@" not in ua:
            raise ValueError(
                "SEC_USER_AGENT must include a contact email, e.g. 'Your Name you@email.com'. "
                "Requests without a proper User-Agent are rejected by the SEC."
            )
        self._limiter = _RateLimiter()
        self._client = httpx.Client(
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
        )

    def _get(self, url: str) -> httpx.Response:
        self._limiter.wait()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    def ticker_to_cik(self) -> dict[str, int]:
        """Return {TICKER: cik} for every SEC filer."""
        data = self._get(_COMPANY_TICKERS).json()
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}

    def get_filings(
        self,
        cik: int,
        ticker: str,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int | None = None,
        since: str | None = None,  # ISO date, e.g. "2022-01-01"
    ) -> list[Filing]:
        """List a company's filings, filtered by form type and (optionally) date."""
        data = self._get(_SUBMISSIONS.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        out: list[Filing] = []
        for form, date, acc, doc in zip(
            recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"]
        ):
            if form not in forms:
                continue
            if since and date < since:
                continue
            out.append(
                Filing(
                    cik=cik, ticker=ticker, form=form, filing_date=date,
                    accession=acc, primary_document=doc,
                )
            )
            if limit and len(out) >= limit:
                break
        return out

    def download(self, url: str) -> bytes:
        """Download a filing document (HTML or PDF)."""
        return self._get(url).content

    def close(self) -> None:
        self._client.close()
