"""LessonBook — persistent cross-run reflection memory.

References (older → 2024 → 2025/2026):
    • **Reflexion** (Shinn et al., NeurIPS 2023) — canonical "verbal episodic
      memory" of past failures, retrieved on next attempt. Foundational.
    • **Trace** (Microsoft, 2024) — gradient-style optimization over agent
      execution traces; treats lessons as differentiable signals.
      https://github.com/microsoft/Trace
    • **AFlow** (Zhang et al., 2024) — meta-agent that learns agent workflows
      from past lesson sets; future direction for v0.3 if we want LessonBook
      to drive automatic graph rewrites.
    • **Trajectory-Informed Memory Generation** (arXiv 2603.10600, 2026) —
      generates structured memories from agent trajectories; exact methodology
      precedent for our CriticAgent → LessonBook → next-run-injection flow.
    • **Awesome Self-Evolving Agents** (XMU DeepLIT, 2025-2026 survey
      https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents) — taxonomy of
      Model / Environment / Co-Evolution; our LessonBook + CriticAgent + Editor
      together cover the "Model-Environment Co-Evolution" quadrant.

We extend Reflexion across *runs* (not just episodes within a run):

    run N    →  CriticAgent reads trajectory  →  writes lessons to disk
    run N+1  →  Pipeline loads relevant lessons → injects into prompts

Schema (one JSON object per line in ``lessons.jsonl``):

    {
      "lesson_id":   "lsn_2026-05-16T03:21:00_a1b2",
      "created_at":  1778905036.78,
      "trigger":     "low_reward" | "duplicate_query" | "disagreement"
                     | "fallback_taken" | "validation_failed",
      "scope":       "screenwriter" | "director" | "orchestrator"
                     | "editor" | "validator" | "global",
      "context":     {"user_prompt_keywords": [...], "memory_size": int,
                      "music_bpm": float, "energy_level": str, ...},
      "lesson":      "<short imperative sentence — e.g. 'Avoid generic queries
                       like \"a cinematic shot\" when memory has < 10 shots'>",
      "evidence":    {"reward": float, "agent": "...", "segment_idx": int}
    }

LessonBook is intentionally schema-stable and append-only so that:
  • Same JSONL file is RL training data later (no migrations).
  • Lessons can be retrieved by SQL-like predicates without a real DB.
  • A run that crashes still leaves earlier lessons intact.

Open-source dependency: stdlib only. (Match design-doc §15 "no hidden deps".)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


@dataclass
class Lesson:
    lesson_id: str
    created_at: float
    trigger: str                      # see module docstring for allowed values
    scope: str                        # which agent / "global"
    context: dict[str, Any] = field(default_factory=dict)
    lesson: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def _new_lesson_id() -> str:
    return f"lsn_{int(time.time())}_{uuid.uuid4().hex[:6]}"


class LessonBook:
    """Append-only JSONL store of Lessons, with simple retrieval helpers.

    Stored at ``<cache_root>/lessons.jsonl`` by default. Reading is cheap
    (whole file fits in memory for any realistic v0.2 workload) and writing
    is append-only so concurrent writers can't corrupt earlier records.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the file exists (empty) so callers / tests can always
        # ``assert path.exists()`` even when the book has no lessons yet.
        self.path.touch(exist_ok=True)
        self._cache: Optional[list[Lesson]] = None

    # ─── write side ───

    def add(self, lesson: Lesson) -> Lesson:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(lesson), ensure_ascii=False) + "\n")
        if self._cache is not None:
            self._cache.append(lesson)
        return lesson

    def add_simple(
        self,
        trigger: str,
        scope: str,
        lesson_text: str,
        context: Optional[dict[str, Any]] = None,
        evidence: Optional[dict[str, Any]] = None,
    ) -> Lesson:
        return self.add(Lesson(
            lesson_id=_new_lesson_id(),
            created_at=time.time(),
            trigger=trigger, scope=scope,
            context=context or {}, lesson=lesson_text,
            evidence=evidence or {},
        ))

    # ─── read side ───

    def all(self) -> list[Lesson]:
        if self._cache is None:
            self._cache = self._load()
        return list(self._cache)

    def _load(self) -> list[Lesson]:
        out: list[Lesson] = []
        if not self.path.exists():
            return out
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out.append(Lesson(**rec))
                except Exception:
                    # Skip malformed lines so a half-flushed write doesn't
                    # kill the whole load.
                    continue
        return out

    def filter(
        self,
        scope: Optional[str] = None,
        trigger: Optional[str] = None,
        predicate: Optional[Callable[[Lesson], bool]] = None,
        limit: int = 50,
    ) -> list[Lesson]:
        out: list[Lesson] = []
        for L in self.all():
            if scope and L.scope != scope and L.scope != "global":
                continue
            if trigger and L.trigger != trigger:
                continue
            if predicate and not predicate(L):
                continue
            out.append(L)
            if len(out) >= limit:
                break
        return out

    def retrieve_relevant(
        self,
        scope: str,
        keywords: Iterable[str] = (),
        limit: int = 5,
    ) -> list[Lesson]:
        """Cheap keyword-overlap retrieval.

        v0.1 uses simple token overlap so prompt injection is deterministic;
        v0.2 swap-in: BM25 or CLIP-text embedding similarity on the
        ``user_prompt_keywords`` field of stored context.
        """
        kw = {k.lower() for k in keywords if k}
        scored: list[tuple[int, Lesson]] = []
        for L in self.filter(scope=scope, limit=10_000):
            haystack = (
                (L.lesson or "") + " " +
                " ".join(str(v).lower() for v in L.context.values())
            ).lower()
            score = sum(1 for k in kw if k in haystack)
            if score > 0 or not kw:
                scored.append((score, L))
        scored.sort(key=lambda x: (-x[0], -x[1].created_at))
        return [L for _, L in scored[:limit]]

    # ─── helpers ───

    def reload(self) -> None:
        self._cache = None

    def __len__(self) -> int:
        return len(self.all())


__all__ = ["Lesson", "LessonBook"]
