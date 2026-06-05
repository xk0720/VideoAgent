"""LessonLibrary (C4) — capability-level self-improvement.

Distills each revision round's "failure + successful fix + trigger" into a
retrievable lesson. Future tasks retrieve relevant lessons during planning and
inject them as constraints BEFORE generating — so the system gets better across
tasks, not just within one task. No existing framework does this (空白③).

v0.3 (NEW): A-MEM-style memory evolution — each lesson is a *structured note*
with stable `lesson_id`, `keywords`, `linked_lesson_ids`. Adding a new lesson
auto-links it to keyword-overlapping existing lessons (the lightweight v0.2.2
heuristic stand-in for A-MEM's LLM-driven linker, see arXiv:2502.12110).
The on-disk JSONL format is forward-compatible: old records load with the new
fields defaulted, and we re-hash a stable `lesson_id` so old files become
linkable retroactively.

Persisted as JSONL so it survives across runs.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..types import Lesson, PhysFailureMode

# ── A-MEM-style linker heuristic ─────────────────────────────────────────
_LINK_KEYWORD_OVERLAP = 2          # min shared keywords to auto-link
_LINK_COSINE_FLOOR = 0.4           # OR cosine ≥ this (lighter typed overlap)


def _stable_id(trigger: str, fix: str, mode: Optional[PhysFailureMode]) -> str:
    """Content-hash so the same lesson reloaded from disk gets the same id —
    lets cross-references survive a process restart."""
    m = mode.value if mode is not None else ""
    h = hashlib.md5(f"{trigger}|{fix}|{m}".encode("utf-8")).hexdigest()
    return f"L{h[:12]}"


def _extract_keywords(text: str, k: int = 8) -> list[str]:
    """Cheap keyword extractor — top-k longest tokens by length, deduped.
    Replaces a real LLM tagger; sufficient for the linker heuristic."""
    toks = re.findall(r"\w{4,}", (text or "").lower())
    seen: dict[str, int] = {}
    for t in toks:
        seen[t] = seen.get(t, 0) + 1
    # Stable order: by frequency desc, then alpha.
    ranked = sorted(seen.keys(), key=lambda t: (-seen[t], t))
    return ranked[:k]


class LessonLibrary:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.lessons: list[Lesson] = []
        self._by_id: dict[str, Lesson] = {}
        if self.path and self.path.exists():
            self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            mode = d.get("failure_mode")
            mode_val = PhysFailureMode(mode) if mode else None
            lesson = Lesson(
                trigger=d["trigger"],
                fix=d["fix"],
                failure_mode=mode_val,
                embedding=embed_text(d["trigger"]),
                # v0.3 fields (default-tolerant for old JSONL):
                lesson_id=d.get("lesson_id") or _stable_id(d["trigger"], d["fix"], mode_val),
                keywords=d.get("keywords") or _extract_keywords(d["trigger"] + " " + d["fix"]),
                tags=d.get("tags", []),
                linked_lesson_ids=d.get("linked_lesson_ids", []),
                revised_by=d.get("revised_by", []),
                confidence=float(d.get("confidence", 1.0)),
                uses=int(d.get("uses", 0)),
                born_task_id=d.get("born_task_id", ""),
            )
            self.lessons.append(lesson)
            self._by_id[lesson.lesson_id] = lesson

    def _write_one(self, lesson: Lesson) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "lesson_id": lesson.lesson_id,
                "trigger": lesson.trigger,
                "fix": lesson.fix,
                "failure_mode": lesson.failure_mode.value if lesson.failure_mode else None,
                "keywords": lesson.keywords,
                "tags": lesson.tags,
                "linked_lesson_ids": lesson.linked_lesson_ids,
                "revised_by": lesson.revised_by,
                "confidence": lesson.confidence,
                "uses": lesson.uses,
                "born_task_id": lesson.born_task_id,
            }, ensure_ascii=False) + "\n")

    # ── public API ───────────────────────────────────────────────────────
    def add(
        self,
        trigger: str,
        fix: str,
        failure_mode: Optional[PhysFailureMode] = None,
        tags: Optional[list[str]] = None,
        born_task_id: str = "",
    ) -> Lesson:
        lesson_id = _stable_id(trigger, fix, failure_mode)
        # Idempotence: re-adding an identical lesson just reconfirms it.
        if lesson_id in self._by_id:
            existing = self._by_id[lesson_id]
            existing.confidence = min(1.0, existing.confidence + 0.05)
            return existing
        lesson = Lesson(
            trigger=trigger,
            fix=fix,
            failure_mode=failure_mode,
            embedding=embed_text(trigger),
            lesson_id=lesson_id,
            keywords=_extract_keywords(trigger + " " + fix),
            tags=tags or [],
            born_task_id=born_task_id,
        )
        # A-MEM-style evolution: link to keyword-overlapping or cosine-similar
        # existing lessons BEFORE persisting, so the bidirectional link is on
        # disk immediately.
        for other in self.lessons:
            shared = set(lesson.keywords) & set(other.keywords)
            sim = cosine(lesson.embedding, other.embedding) if other.embedding is not None else 0.0
            if len(shared) >= _LINK_KEYWORD_OVERLAP or sim >= _LINK_COSINE_FLOOR:
                if other.lesson_id not in lesson.linked_lesson_ids:
                    lesson.linked_lesson_ids.append(other.lesson_id)
                if lesson_id not in other.linked_lesson_ids:
                    other.linked_lesson_ids.append(lesson_id)
        self.lessons.append(lesson)
        self._by_id[lesson_id] = lesson
        self._write_one(lesson)
        return lesson

    def get(self, lesson_id: str) -> Optional[Lesson]:
        return self._by_id.get(lesson_id)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list[Lesson]:
        if not self.lessons:
            return []
        q = embed_text(query)
        scored = [(cosine(q, l.embedding) * l.confidence, l) for l in self.lessons]
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [l for s, l in scored[:top_k] if s >= threshold]
        for l in hits:
            l.uses += 1
        return hits

    def neighbors(self, lesson_id: str) -> list[Lesson]:
        """Return the A-MEM links of a lesson (helps an LLM see context)."""
        lesson = self._by_id.get(lesson_id)
        if lesson is None:
            return []
        return [self._by_id[lid] for lid in lesson.linked_lesson_ids
                if lid in self._by_id]

    def __len__(self) -> int:
        return len(self.lessons)
