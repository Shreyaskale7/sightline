"""Split page text into embedding-sized windows (fixes a measured blind spot).

The problem: BGE-small reads ~512 tokens, but a filing page holds 2-5x that. The embedder
silently truncates, so a fact in the BOTTOM half of a page is invisible to dense retrieval.

The fix: embed each page as several overlapping windows that all point back to the same
(accession, page_no). Retrieval over-fetches windows and collapses them to page-level hits,
so the rest of the system (eval, citations, fusion) never notices chunks exist — the page
stays the unit of retrieval and citation, as Sightline's design demands.

Window size ~1800 chars ≈ 450 tokens (safely under the 512 limit); 200-char overlap so a
sentence straddling a boundary appears whole in at least one window. Splits prefer paragraph
breaks, then line breaks, so windows stay semantically coherent.
"""
from __future__ import annotations

_TARGET_CHARS = 1800
_OVERLAP_CHARS = 200


def chunk_text(text: str, target: int = _TARGET_CHARS, overlap: int = _OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping windows, preferring paragraph/line boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            # Prefer to break at a paragraph, then a newline, then a space — searching
            # backwards from the target size so windows stay near-full.
            window = text[start:end]
            for sep in ("\n\n", "\n", " "):
                cut = window.rfind(sep, target // 2)
                if cut != -1:
                    end = start + cut
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]
