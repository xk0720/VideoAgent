"""CriticAgent — post-hoc meta-reviewer.

Runs *after* an entire pipeline run finishes. Reads the trajectory log,
identifies suspect decision points, and writes structured Lesson entries
to the LessonBook so the *next* run can avoid the same mistakes.

This is the cross-run analog of Reflexion's verbal episodic memory
(Shinn et al., NeurIPS 2023): the Critic distills failures into short
imperative lessons. Unlike Reflexion's same-agent self-critique, our
CriticAgent is a separate role — preventing self-grading bias the same
way ValidatorAgent is separate from EditorAgent.

Triggers we detect (cheap rules, no LLM call needed — keeps Critic free
to call after every run):

    • low_reward         — a segment_finalized step's reward < threshold
    • duplicate_query    — Director emitted two identical semantic_queries
    • disagreement       — judge score and Σ(weighted m1..m6) differ > δ
    • fallback_taken     — Editor took the "fallback" action
    • validation_failed  — Orchestrator's first validate() returned passed=False

For each trigger we synthesise a short imperative lesson keyed on the
relevant context (memory size, user-prompt keywords, music BPM, ...) so
LessonBook.retrieve_relevant can surface it later.

Open-source background (older + 2024-2025 successors):
    • **Reflexion**     (Shinn et al., NeurIPS 2023) — canonical verbal self-reflection.
    • **Self-Discover** (Zhou et al., 2024) — separation of meta-agent from
                          base agents; the architectural precedent for keeping
                          CriticAgent out of the execution loop.
    • **Trace**         (Microsoft, 2024) — gradient-style optimization over
                          agent traces; treats critique as a learnable signal.
                          https://github.com/microsoft/Trace
    • **rStar**         (Microsoft, 2024) + **rStar-Math** (Microsoft, Jan 2025) —
                          self-play search-augmented reasoning with a separate
                          verifier; analogous to our Editor + Critic split.
    • **AFlow**         (Zhang et al., 2024) — meta-agent that designs agent
                          workflows from past lesson sets; the v0.3 destination
                          when our LessonBook is rich enough.
    • **G-Eval**        (Liu et al. 2023) — structured critique prompts (we
                          use static rules in v0.1; LLM critique slot in v0.2).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..memory.lessons import LessonBook
from ..logging import logger


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "with", "make", "create",
    "video", "edit", "edits", "editing", "shot", "shots", "clip", "clips",
    "very", "high", "low",
}


def _keywords(text: str, k: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2][:k]


class CriticAgent:
    """Reads a trajectory JSONL and emits Lessons.

    Intentionally *not* a BaseAgent subclass — it has no LLM dependency
    in v0.1, no prompt template, and is invoked from the pipeline driver
    rather than the orchestration graph.
    """

    name = "critic"

    def __init__(
        self,
        lesson_book: LessonBook,
        low_reward_threshold: float = 6.0,
        disagreement_threshold: float = 2.0,
    ) -> None:
        self.book = lesson_book
        self.low_reward_threshold = low_reward_threshold
        self.disagreement_threshold = disagreement_threshold

    def review(
        self,
        trajectory_path: Path | str,
        user_prompt: str = "",
        run_context: dict[str, Any] | None = None,
    ) -> list:
        """Scan a finished run's trajectory JSONL and write Lessons.

        Returns the list of Lessons produced this run (for caller logging).
        """
        run_context = dict(run_context or {})
        run_context.setdefault("user_prompt_keywords", _keywords(user_prompt))

        records = _load_jsonl(Path(trajectory_path))
        if not records:
            return []

        lessons = []
        lessons += self._scan_low_reward(records, run_context)
        lessons += self._scan_duplicate_queries(records, run_context)
        lessons += self._scan_disagreement(records, run_context)
        lessons += self._scan_fallbacks(records, run_context)
        lessons += self._scan_validation_failed(records, run_context)
        return lessons

    # ─────── scanners ───────

    def _scan_low_reward(self, records, ctx):
        out = []
        for rec in records:
            if rec.get("action") != "segment_finalized":
                continue
            reward = rec.get("reward")
            if reward is not None and reward < self.low_reward_threshold:
                msg = (
                    f"Segment with query "
                    f"{rec['action_input'].get('semantic_query')!r} scored low ({reward:.2f}); "
                    f"prefer queries grounded by ≥1 strong CLIP match before "
                    f"committing to a SegmentGuidance."
                )
                out.append(self.book.add_simple(
                    trigger="low_reward", scope="editor",
                    lesson_text=msg, context=ctx,
                    evidence={"reward": reward,
                              "segment_idx": rec["action_input"].get("segment_idx"),
                              "heuristic": rec["action_input"].get("heuristic")},
                ))
        return out

    def _scan_duplicate_queries(self, records, ctx):
        out = []
        seen: dict[str, int] = {}
        for rec in records:
            if rec.get("action") != "segment_finalized":
                continue
            q = rec["action_input"].get("semantic_query", "")
            seen[q] = seen.get(q, 0) + 1
        for q, n in seen.items():
            if n >= 2:
                out.append(self.book.add_simple(
                    trigger="duplicate_query", scope="director",
                    lesson_text=(
                        f"Avoid repeating the semantic_query {q!r} across "
                        f"segments — DirectorAgent should diversify."
                    ),
                    context=ctx, evidence={"count": n, "query": q},
                ))
        return out

    def _scan_disagreement(self, records, ctx):
        out = []
        for rec in records:
            if rec.get("action") != "segment_finalized":
                continue
            scores = rec.get("observation", {}).get("metric_scores", {}) or {}
            judge = scores.get("validator")
            if judge is None:
                continue
            m_sum = sum(float(scores.get(f"m{i}", 0.0)) for i in range(1, 7)) * 10.0 / 6.0
            if abs(m_sum - judge) > self.disagreement_threshold:
                out.append(self.book.add_simple(
                    trigger="disagreement", scope="validator",
                    lesson_text=(
                        f"Judge ({judge:.2f}) and metric-mean ({m_sum:.2f}) "
                        f"disagree by >{self.disagreement_threshold:.1f}; flag for "
                        f"active-learning review."
                    ),
                    context=ctx,
                    evidence={"judge": judge, "metric_mean": m_sum,
                              "segment_idx": rec["action_input"].get("segment_idx")},
                ))
        return out

    def _scan_fallbacks(self, records, ctx):
        out = []
        for rec in records:
            if rec.get("agent_name") == "editor" and rec.get("action") == "fallback":
                out.append(self.book.add_simple(
                    trigger="fallback_taken", scope="editor",
                    lesson_text=(
                        "EditorAgent had to take a fallback — consider lowering "
                        "GenerationCfg.fallback_threshold or broadening retrieval pool."
                    ),
                    context=ctx,
                    evidence={"segment_idx": rec["action_input"].get("segment_idx")},
                ))
        return out

    def _scan_validation_failed(self, records, ctx):
        out = []
        for rec in records:
            if (rec.get("agent_name") == "orchestrator"
                    and rec.get("action") == "validate"
                    and rec.get("reward") == 0.0
                    and rec.get("observation", {}).get("n_feedback", 0) > 0):
                out.append(self.book.add_simple(
                    trigger="validation_failed", scope="orchestrator",
                    lesson_text=(
                        "First-iteration plan validation failed; downstream "
                        "ScreenwriterAgent should emit fuller music-section coverage "
                        "or more grounded queries from the start."
                    ),
                    context=ctx,
                    evidence={"n_feedback": rec["observation"].get("n_feedback")},
                ))
                break  # one is enough
        return out


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                logger.warning(f"CriticAgent: skipped malformed trajectory line in {path}")
    return out


__all__ = ["CriticAgent"]
