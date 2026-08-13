"""A corpus indexed on one OS must serve on another.

The index is built on a workstation and deployed into a Linux container. Rows written on
Windows store backslash separators; read literally on POSIX the whole string becomes a single
filename, so .exists() fails and .parent collapses to "." — page images 404 and the citation
UI silently shows nothing. This guards the read-side normalization that prevents it.
"""
from __future__ import annotations


from sightline.ingest.store import MetadataStore, _norm_path


def test_windows_separators_normalize_to_a_real_path():
    p = _norm_path(r"data\pages\000104581026000021\p0051.png")
    # Parent must be a directory chain, not "." — that's what lets us find filing.pdf beside it.
    assert p.name == "p0051.png"
    assert p.parent.name == "000104581026000021"
    assert p.parent.parent.name == "pages"


def test_posix_separators_are_unchanged():
    p = _norm_path("data/pages/000104581026000021/p0051.png")
    assert p.parent.name == "000104581026000021"


def test_store_returns_usable_paths_for_windows_written_rows(tmp_path):
    """Simulate a DB written on Windows, then read it the way the container does."""
    db = tmp_path / "sightline.db"
    store = MetadataStore(db)
    store._conn.execute(
        "INSERT INTO filings (accession, cik, ticker, form, filing_date, primary_document,"
        " n_pages, ingested_at) VALUES (?,?,?,?,?,?,?,?)",
        ("0000000000-99-000001", 0, "TEST", "10-K", "2026-01-01", "d.htm", 1, "2026-01-01"),
    )
    store._conn.execute(
        "INSERT INTO pages (accession, page_no, image_path, text, char_count) VALUES (?,?,?,?,?)",
        ("0000000000-99-000001", 1, r"data\pages\000000000099000001\p0001.png", "x", 1),
    )
    store._conn.commit()

    page = store.get_page("0000000000-99-000001", 1)
    assert page is not None
    # The PDF that a page renders from is resolved as a sibling of the image.
    assert page.image_path.parent.name == "000000000099000001"
    assert (page.image_path.parent / "filing.pdf").name == "filing.pdf"
    store.close()
