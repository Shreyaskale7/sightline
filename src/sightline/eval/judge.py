"""LLM-as-judge for answer correctness (M1: basic version).

"LLM-as-judge" = using a model to grade another model's answers against a gold answer. It's
how you score free-text answers at scale ("$215,938 million" vs "about $216 billion" should
both pass — exact string match can't do that).

Two disciplines, per the design doc:
  - Use a CHEAP model (Haiku) — grading is high-volume, so cost matters more than brilliance.
  - Don't trust it blindly: in M3 we hand-label a sample of its verdicts and report Cohen's
    kappa (judge-human agreement). Until then, treat correctness numbers as approximate.

Client is injectable so tests run without network or spend.
"""
from __future__ import annotations

from ..config import settings

_JUDGE_PROMPT = """\
You are grading a question-answering system on SEC filings.

Question: {question}
Gold (reference) answer: {gold}
System's answer: {answer}

Does the system's answer state the same essential fact(s) as the gold answer? Minor wording,
rounding (e.g. "$215,938 million" vs "$216 billion"), or extra correct context are fine.
Contradicting or omitting the essential fact is a fail.

Reply with exactly one word: CORRECT or INCORRECT.
"""


class Judge:
    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        self.model = model or settings.router_model  # the cheap model (Haiku)
        self._client = client

    def _ensure_client(self) -> None:
        if self._client is None:
            from anthropic import Anthropic

            if not settings.llm_api_key:
                raise RuntimeError("LLM_API_KEY is empty — set it in .env.")
            self._client = Anthropic(api_key=settings.llm_api_key)

    def is_correct(self, question: str, gold: str, answer: str) -> bool:
        self._ensure_client()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": _JUDGE_PROMPT.format(question=question, gold=gold, answer=answer),
            }],
        )
        return resp.content[0].text.strip().upper().startswith("CORRECT")
