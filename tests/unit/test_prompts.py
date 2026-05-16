"""Prompt-template integrity tests.

For every shipped prompt:
  • it must be parseable by ``str.format`` (no stray ``{`` etc.)
  • its placeholders must match what its agent actually passes in
"""
from __future__ import annotations

import pytest

from longvideoagent.agents.base import extract_placeholders
from longvideoagent.config import load_prompt


PROMPT_AGENT_EXPECTED = {
    "screenwriter":           {"user_prompt", "memory_summary", "bpm", "music_sections"},
    "director_query":         {"section_plan", "history"},
    "director_heuristic":     {"preset_descriptions", "semantic_query", "energy_level"},
    "director_pacing":        {"section_start", "section_end", "section_beats", "energy_level"},
    "orchestrator_validate":  {"global_plan", "segment_guidances", "memory_summary", "checks"},
    "editor_summary":         {"max_steps", "guidance", "candidates", "neighbor_context",
                               "feasibility_threshold"},
    "reward_judge":           {"semantic_query", "heuristic", "cinematography"},
}


@pytest.mark.parametrize("name,expected", list(PROMPT_AGENT_EXPECTED.items()))
def test_prompt_placeholders_match_callers(name, expected):
    template = load_prompt(name)
    actual = extract_placeholders(template)
    assert actual == expected, (
        f"Prompt {name}.txt has placeholders {actual!r} "
        f"but caller expects {expected!r}"
    )


def test_render_prompt_raises_with_clear_message():
    from longvideoagent.agents import ScreenwriterAgent
    from longvideoagent.models.llm import MockLLMClient

    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))
    with pytest.raises(KeyError) as ei:
        sw.render_prompt(user_prompt="x")        # missing memory_summary, bpm, music_sections
    assert "missing placeholder" in str(ei.value)
