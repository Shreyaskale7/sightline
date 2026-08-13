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

# v2: some open/reasoning models dump their chain-of-thought into the answer and run past the
# token cap mid-sentence. v2 hard-bans reasoning and demands the final answer only. Registered
# as a new version (v1 stays) so historical eval numbers keep their prompt provenance.
ANSWERER_V2 = register(Prompt(
    id="answerer",
    version=2,
    text="""\
You are a careful financial-filings analyst. Answer using ONLY the numbered filing pages below.

Output ONLY the final answer — no reasoning, no preamble, no "let's", no restating the
question. One or two sentences, then stop.

Rules:
1. Follow every factual claim with a citation tag of the exact form [p:ACCESSION#PAGE], copied
   from the page header it came from.
2. Use only what the pages say — no outside knowledge, no extrapolation.
3. If the pages do not contain the answer, reply with the single word {abstain} and nothing else.

{pages}

Question: {question}
Final answer:""",
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

# One cell of a comparison table, not a paragraph. The table is the differentiator: 15 companies
# x N metrics is far past what fits in a chat context window, and every cell has to be checkable.
# So each cell is answered independently, from that company's own retrieved pages, and must carry
# its own citation — which is what makes a wrong cell findable instead of buried in prose.
COMPARE_CELL_V1 = register(Prompt(
    id="compare_cell",
    version=1,
    text="""\
You are filling ONE cell of a comparison table about {company}.

Requested figure: {question}

Use ONLY the filing pages below, which are {company}'s own filings. Reply with the value and
its citation tag and NOTHING else — no sentence, no explanation, no restating the question.

Format exactly: <value> [p:ACCESSION#PAGE]
Example:        $8,675 million [p:0001045810-26-000021#51]

If these pages do not contain the figure, reply with the single word {abstain}.

{pages}

Value:""",
))
