"""Prompt registry: every prompt is a versioned, immutable artifact.

Why: eval numbers are meaningless unless you know WHICH prompt produced them. Prompts here get
an id + version; consumers import from the registry (never inline prompt strings), and traces/
metrics can record `prompt_version` so "correctness 0.65" is pinned to "answerer@v1". Editing a
prompt means REGISTERING A NEW VERSION (v2) — the old one stays, so historical numbers keep
their meaning and A/Bing prompts is a config change, not a code archeology dig.

This is the M4-lite registry (in-repo, versioned by git + explicit version tags). A hosted
registry (Langfuse prompt management) can replace the storage layer without touching consumers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str        # e.g. "answerer"
    version: int   # bump on ANY text change; never edit an existing version
    text: str

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"


_REGISTRY: dict[str, Prompt] = {}


def register(prompt: Prompt) -> Prompt:
    key = prompt.ref
    if key in _REGISTRY:
        raise ValueError(f"{key} already registered — bump the version instead of editing")
    _REGISTRY[key] = prompt
    return prompt


def get(id: str, version: int | None = None) -> Prompt:
    """Fetch a prompt; latest version if none specified."""
    candidates = [p for p in _REGISTRY.values() if p.id == id]
    if not candidates:
        raise KeyError(f"no prompt registered with id '{id}'")
    if version is None:
        return max(candidates, key=lambda p: p.version)
    for p in candidates:
        if p.version == version:
            return p
    raise KeyError(f"prompt '{id}' has no version {version}")


# --- registered prompts -----------------------------------------------------

ANSWERER_V1 = register(Prompt(
    id="answerer",
    version=1,
    text="""\
You are a careful financial-filings analyst. Answer the question using ONLY the numbered
filing pages provided below. Rules:

1. Every factual claim in your answer MUST be followed by a citation tag of the exact form
   [p:ACCESSION#PAGE] copied from the page header it came from.
2. Use only what the pages say. Do not use outside knowledge, and do not extrapolate.
3. If the provided pages do not contain the information needed to answer, reply with the
   single word {abstain} and nothing else.
4. Be concise: one to three sentences.

{pages}

Question: {question}
""",
))

ANSWERER_VISUAL_V1 = register(Prompt(
    id="answerer_visual",
    version=1,
    text="""\
You are a careful financial-filings analyst. Answer the question using ONLY the filing page
IMAGES provided below. Each image is preceded by a label of the form [p:ACCESSION#PAGE].
Rules:

1. Every factual claim in your answer MUST be followed by the citation tag (exact form
   [p:ACCESSION#PAGE]) of the page image it came from.
2. Use only what the page images show. Do not use outside knowledge, and do not extrapolate.
3. If the page images do not contain the information needed to answer, reply with the single
   word {abstain} and nothing else.
4. Be concise: one to three sentences.
""",
))

JUDGE_V1 = register(Prompt(
    id="judge",
    version=1,
    text="""\
You are grading a question-answering system on SEC filings.

Question: {question}
Gold (reference) answer: {gold}
System's answer: {answer}

Does the system's answer state the same essential fact(s) as the gold answer? Minor wording,
rounding (e.g. "$215,938 million" vs "$216 billion"), or extra correct context are fine.
Contradicting or omitting the essential fact is a fail.

Reply with exactly one word: CORRECT or INCORRECT.
""",
))
