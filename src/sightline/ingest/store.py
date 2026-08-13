"""SQLite metadata store for ingested filings and their page images.

Why SQLite: page-level metadata (which company / form / period / page a PNG belongs to)
is small, relational, and needs *idempotent* upserts. `sqlite3` is in the stdlib — no extra
dependency — and is plenty for a v1 corpus of a few thousand pages. Graduate to Postgres only
if the corpus outgrows a single file.

Idempotency is a hard constraint: re-running ingestion must never duplicate a
filing. We key filings on the SEC `accession` number (PRIMARY KEY) and pages on
(accession, page_no), and re-ingesting a filing replaces its rows transactionally rather than
appending. The image files on disk are likewise overwritten in place, keyed by the same ids.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .edgar import Filing
from .rasterize import Page

# Schema is created on connect; `IF NOT EXISTS` keeps `MetadataStore(...)` cheap and idempotent.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession        TEXT PRIMARY KEY,
    cik              INTEGER NOT NULL,
    ticker           TEXT NOT NULL,
    form             TEXT NOT NULL,
    filing_date      TEXT NOT NULL,
    primary_document TEXT NOT NULL,
    n_pages          INTEGER,
    ingested_at      TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    accession   TEXT NOT NULL,
    page_no     INTEGER NOT NULL,   -- 1-based, matches how a human cites "page N"
    image_path  TEXT NOT NULL,
    text        TEXT NOT NULL,      -- extracted text: prefilter side of hybrid retrieval ONLY
    char_count  INTEGER NOT NULL,
    PRIMARY KEY (accession, page_no),
    FOREIGN KEY (accession) REFERENCES filings(accession) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pages_accession ON pages(accession);
"""


def _norm_path(raw: str) -> Path:
    """Read a stored image path in an OS-portable way.

    Rows written on Windows store separators as backslashes ("data\\pages\\acc\\p0001.png").
    On POSIX those are *not* separators — the whole string becomes one filename, so .exists()
    fails and .parent collapses to ".". That breaks page serving whenever an index built on one
    OS is deployed on another (exactly what happens when a Windows-built corpus ships to a Linux
    container). Normalizing on read keeps the stored rows untouched and fixes every consumer.
    """
    return Path(raw.replace("\\", "/"))


@dataclass
class StoredPage:
    """A page row read back out of the store (image + prefilter text + provenance)."""

    accession: str
    page_no: int
    image_path: Path
    text: str
    ticker: str
    form: str
    filing_date: str


class MetadataStore:
    """Thin, idempotent wrapper over a SQLite file. Use as a context manager."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the API keeps one long-lived store on the shared retriever
        # singleton and serves requests from a threadpool, so the connection is created in one
        # thread and used in another. Safe here because the API serializes queries under a lock
        # (main.py: _retriever_lock) — no concurrent access to this connection.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")  # so ON DELETE CASCADE actually fires
        self._conn.executescript(_SCHEMA)

    # --- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes --------------------------------------------------------------
    def is_ingested(self, accession: str) -> bool:
        """True if this filing was already fully ingested (has a page count)."""
        row = self._conn.execute(
            "SELECT 1 FROM filings WHERE accession = ? AND n_pages IS NOT NULL",
            (accession,),
        ).fetchone()
        return row is not None

    def save_filing_with_pages(self, filing: Filing, pages: list[Page]) -> None:
        """Persist one filing + all its pages in a single transaction.

        Re-running for the same accession replaces the prior rows (delete-then-insert),
        which is what makes ingestion safe to re-run. The `with self._conn` block is an
        atomic transaction: either the whole filing lands or none of it does.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn:  # transaction
            # Replace any prior pages for this accession (idempotent re-ingest).
            self._conn.execute("DELETE FROM pages WHERE accession = ?", (filing.accession,))
            self._conn.execute(
                """
                INSERT INTO filings
                    (accession, cik, ticker, form, filing_date, primary_document, n_pages, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(accession) DO UPDATE SET
                    cik=excluded.cik, ticker=excluded.ticker, form=excluded.form,
                    filing_date=excluded.filing_date, primary_document=excluded.primary_document,
                    n_pages=excluded.n_pages, ingested_at=excluded.ingested_at
                """,
                (
                    filing.accession, filing.cik, filing.ticker, filing.form,
                    filing.filing_date, filing.primary_document, len(pages), now,
                ),
            )
            self._conn.executemany(
                "INSERT INTO pages (accession, page_no, image_path, text, char_count) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (filing.accession, p.page_no, str(p.image_path), p.text, len(p.text))
                    for p in pages
                ],
            )

    # --- reads ---------------------------------------------------------------
    def count_filings(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

    def count_pages(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    def has_uploads(self) -> bool:
        """True if any user-uploaded document (form='UPLOAD') is indexed. Drives the product
        default: once you've uploaded, your questions search YOUR documents, not the sample
        SEC corpus. Cheap enough to check per query (indexed COUNT with a LIMIT)."""
        row = self._conn.execute(
            "SELECT 1 FROM filings WHERE form = 'UPLOAD' LIMIT 1"
        ).fetchone()
        return row is not None

    def list_tickers(self, exclude_uploads: bool = True) -> list[str]:
        """Distinct company tickers in the corpus, alphabetical.

        Used by cross-corpus comparison ("compare R&D across every company"): the question
        names no companies, so the set has to come from what's actually indexed. Uploads are
        excluded by default — a user's own documents aren't part of the sample peer group.
        """
        q = "SELECT DISTINCT ticker FROM filings"
        if exclude_uploads:
            q += " WHERE form != 'UPLOAD'"
        q += " ORDER BY ticker"
        return [r["ticker"] for r in self._conn.execute(q)]

    def list_accessions(self, ticker: str, form: str | None = None, limit: int = 3) -> list[str]:
        """Most-recent-first accession numbers for a ticker (optionally one form type).
        Used by decomposed retrieval: 'across the last three 10-Qs' -> one search per filing."""
        q = "SELECT accession FROM filings WHERE ticker = ?"
        args: list[object] = [ticker]
        if form:
            q += " AND form = ?"
            args.append(form)
        q += " ORDER BY filing_date DESC LIMIT ?"
        args.append(limit)
        return [r["accession"] for r in self._conn.execute(q, args)]

    def get_page(self, accession: str, page_no: int) -> StoredPage | None:
        """Fetch one page by its identity — used to hand retrieved pages to the answerer."""
        r = self._conn.execute(
            """
            SELECT p.accession, p.page_no, p.image_path, p.text,
                   f.ticker, f.form, f.filing_date
            FROM pages p JOIN filings f ON f.accession = p.accession
            WHERE p.accession = ? AND p.page_no = ?
            """,
            (accession, page_no),
        ).fetchone()
        if r is None:
            return None
        return StoredPage(
            accession=r["accession"],
            page_no=r["page_no"],
            image_path=_norm_path(r["image_path"]),
            text=r["text"],
            ticker=r["ticker"],
            form=r["form"],
            filing_date=r["filing_date"],
        )

    def iter_pages_for(self, accession: str) -> Iterator[StoredPage]:
        """Pages of ONE filing, with provenance — what indexing an upload needs."""
        cur = self._conn.execute(
            """
            SELECT p.accession, p.page_no, p.image_path, p.text,
                   f.ticker, f.form, f.filing_date
            FROM pages p JOIN filings f ON f.accession = p.accession
            WHERE p.accession = ? ORDER BY p.page_no
            """,
            (accession,),
        )
        for r in cur:
            yield StoredPage(
                accession=r["accession"], page_no=r["page_no"],
                image_path=_norm_path(r["image_path"]), text=r["text"],
                ticker=r["ticker"], form=r["form"], filing_date=r["filing_date"],
            )

    def iter_pages(self) -> Iterator[StoredPage]:
        """Stream every page with its filing provenance — the input to indexing (M1/M2)."""
        cur = self._conn.execute(
            """
            SELECT p.accession, p.page_no, p.image_path, p.text,
                   f.ticker, f.form, f.filing_date
            FROM pages p JOIN filings f ON f.accession = p.accession
            ORDER BY p.accession, p.page_no
            """
        )
        for r in cur:
            yield StoredPage(
                accession=r["accession"],
                page_no=r["page_no"],
                image_path=_norm_path(r["image_path"]),
                text=r["text"],
                ticker=r["ticker"],
                form=r["form"],
                filing_date=r["filing_date"],
            )
