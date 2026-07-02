"""Answer-correctness judge: deterministic numeric check first, LLM only as fallback.

Most Sightline gold answers state an exact figure ("$215,938 million"). For those, grading is
string arithmetic, not judgment — so we check numerically, for free, with zero API calls, and
it's MORE trustworthy than a model's opinion. Only prose answers ("TSMC and Samsung") fall
through to the LLM judge.

"LLM-as-judge" = using a model to grade another model's answers against a gold answer — the
standard way to score free-text at scale. Two disciplines, per the design doc:
  - Use a CHEAP model — grading is high-volume, so cost matters more than brilliance.
  - Don't trust it blindly: in M3 we hand-label a sample of its verdicts and report Cohen's
    kappa (judge-human agreement). Until then, treat LLM-judged correctness as approximate.

Client is injectable so tests run without network or spend.
"""
from __future__ import annotations

import re

from ..config import settings

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(text)]


def numeric_match(gold: str, answer: str) -> bool | None:
    """Deterministic correctness check for figure-style gold answers.

    Rule: the LARGEST number in the gold answer is its essential fact (the revenue figure,
    the employee count — support numbers like "in 38 countries" are smaller). The answer
    passes if that number appears, allowing comma/plain forms and billions-rounding of
    millions figures ("$215,938 million" ≈ "$215.9 billion" ≈ "$216 billion").

    Returns None ("can't tell — use the LLM") when the gold answer has no numbers.
    """
    gold_nums = _numbers(gold)
    if not gold_nums:
        return None
    key = max(gold_nums)
    acceptable = {round(key, 2)}
    if key >= 1_000:  # a millions figure may be quoted in billions
        acceptable |= {round(key / 1_000, 1), round(key / 1_000)}  # 215.9, 216
    ans = {round(n, 2) for n in _numbers(answer)}
    ans |= {round(n, 1) for n in _numbers(answer)}
    return bool(acceptable & ans)

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
            from ..llm import make_client

            self._client = make_client()

    def is_correct(self, question: str, gold: str, answer: str) -> bool:
        verdict = numeric_match(gold, answer)
        if verdict is not None:  # numeric gold answer: graded deterministically, no API call
            return verdict
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
