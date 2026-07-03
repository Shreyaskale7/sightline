"""Tests for the chunker and decomposed retrieval — both are pure logic, test them hard."""
from __future__ import annotations

from sightline.retrieval.chunking import chunk_text
from sightline.retrieval.decompose import decomposed_retrieve, interleave
from sightline.retrieval.text_baseline import Hit


# --- chunking ------------------------------------------------------------------

def test_short_text_is_one_chunk():
    assert chunk_text("short page") == ["short page"]


def test_long_text_splits_with_overlap():
    paras = "\n\n".join(f"Paragraph {i}. " + "x" * 300 for i in range(12))
    chunks = chunk_text(paras)
    assert len(chunks) > 1
    assert all(len(c) <= 1800 for c in chunks)
    # every character of the original appears in some chunk (nothing silently dropped)
    joined = "".join(chunks)
    assert "Paragraph 11" in joined and "Paragraph 0" in joined


def test_empty_text_yields_nothing():
    assert chunk_text("   ") == []


def test_prefers_paragraph_boundaries():
    text = ("A" * 1000) + "\n\n" + ("B" * 1500)
    chunks = chunk_text(text)
    # First chunk should end at the paragraph break, not mid-B
    assert chunks[0].rstrip().endswith("A")


# --- interleave ----------------------------------------------------------------

def _h(acc: str, page: int) -> Hit:
    return Hit(accession=acc, page_no=page, score=1.0)


def test_interleave_guarantees_representation():
    a = [_h("A", 1), _h("A", 2), _h("A", 3)]
    b = [_h("B", 1), _h("B", 2), _h("B", 3)]
    merged = interleave([a, b], top_n=4)
    accs = [h.accession for h in merged]
    assert accs == ["A", "B", "A", "B"]  # both companies in the top-4, alternating


def test_interleave_dedupes():
    a = [_h("A", 1)]
    b = [_h("A", 1), _h("B", 1)]
    assert len(interleave([a, b], top_n=5)) == 2


# --- decomposed_retrieve --------------------------------------------------------

class _FakeRetriever:
    """Records the filters it was called with; returns hits tagged by call order."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def retrieve(self, query: str, k: int = 5, query_filter: object = None) -> list[Hit]:
        self.calls.append(query_filter)
        n = len(self.calls)
        return [_h(f"call{n}", i) for i in range(1, k + 1)]


def test_comparison_fans_out_per_company():
    fake = _FakeRetriever()
    hits = decomposed_retrieve(
        "Which spent more on R&D in its most recent 10-K, NVIDIA or Intel?",
        k=4, retrieve_fn=fake.retrieve, list_accessions=lambda *a, **kw: [],
    )
    assert len(fake.calls) == 2  # one retrieval per company
    assert {h.accession for h in hits[:2]} == {"call1", "call2"}  # both represented up top


def test_multi_hop_fans_out_per_filing():
    fake = _FakeRetriever()
    hits = decomposed_retrieve(
        "How did NVIDIA's quarterly revenue change across its last three 10-Q filings?",
        k=6, retrieve_fn=fake.retrieve,
        list_accessions=lambda t, f, n: ["acc-1", "acc-2", "acc-3"],
    )
    assert len(fake.calls) == 3  # one retrieval per filing
    assert {h.accession for h in hits[:3]} == {"call1", "call2", "call3"}


def test_simple_takes_single_filtered_path():
    fake = _FakeRetriever()
    decomposed_retrieve(
        "What was Micron's total revenue in its most recent 10-K?",
        k=5, retrieve_fn=fake.retrieve, list_accessions=lambda *a, **kw: [],
    )
    assert len(fake.calls) == 1
