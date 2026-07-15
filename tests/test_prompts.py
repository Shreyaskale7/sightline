"""Prompt-registry tests — immutability is the whole point."""
from __future__ import annotations

import pytest

from sightline.prompts import ANSWERER_V1, JUDGE_V1, Prompt, get, register


def test_get_latest_and_specific():
    assert get("answerer").version >= 1
    assert get("answerer", 1) is ANSWERER_V1
    assert get("judge", 1) is JUDGE_V1


def test_reregistering_same_version_is_refused():
    with pytest.raises(ValueError, match="bump the version"):
        register(Prompt(id="answerer", version=1, text="edited!"))


def test_unknown_prompt_raises():
    with pytest.raises(KeyError):
        get("nonexistent")
    with pytest.raises(KeyError):
        get("answerer", 999)


def test_consumers_use_registry_text():
    from sightline.answerer import _PROMPT
    from sightline.eval.judge import _JUDGE_PROMPT

    from sightline.prompts import get

    # The answerer uses the LATEST registered version (v2 curbs reasoning-model rambling).
    assert _PROMPT == get("answerer").text
    assert get("answerer").version >= 2
    assert "no reasoning" in _PROMPT.lower()
    assert _JUDGE_PROMPT == JUDGE_V1.text


def test_v1_still_registered_for_provenance():
    assert ANSWERER_V1.ref == "answerer@v1"  # old version never mutated, historical numbers hold
