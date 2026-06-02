"""LessonLibrary (C4) — capability-level self-improvement.

Distills each revision round's "failure + successful fix + trigger" into a
retrievable lesson. Future tasks retrieve relevant lessons during planning and
inject them as constraints BEFORE generating — so the system gets better across
tasks, not just within one task. No existing framework does this (空白③).

Persisted as JSONL so it survives across runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..types import Lesson, PhysFailureMode


class LessonLibrary:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.lessons: list[Lesson] = []
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            mode = d.get("failure_mode")
            self.lessons.append(
                Lesson(
                    trigger=d["trigger"],
                    fix=d["fix"],
                    failure_mode=PhysFailureMode(mode) if mode else None,
                    embedding=embed_text(d["trigger"]),
                )
            )

    def add(
        self, trigger: str, fix: str, failure_mode: Optional[PhysFailureMode] = None
    ) -> Lesson:
        lesson = Lesson(
            trigger=trigger,
            fix=fix,
            failure_mode=failure_mode,
            embedding=embed_text(trigger),
        )
        self.lessons.append(lesson)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "trigger": trigger,
                            "fix": fix,
                            "failure_mode": failure_mode.value if failure_mode else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return lesson

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list[Lesson]:
        if not self.lessons:
            return []
        q = embed_text(query)
        scored = [(cosine(q, l.embedding), l) for l in self.lessons]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [l for s, l in scored[:top_k] if s >= threshold]

    def __len__(self) -> int:
        return len(self.lessons)
