"""PreferenceStore — C8 Tier-5 per-user preferences (v0.3).

Borrowed: Me-Agent (arXiv:2601.20162) hierarchical preference memory.
Maestro keeps it light — a single JSON file per user_id with cinematic
priors, style priors, and a `physics_strictness` multiplier that the
MetricTool can use to weight p1/p2 more or less aggressively per user.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..types import UserPreference


class PreferenceStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.by_user: dict[str, UserPreference] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for uid, d in (data or {}).items():
            self.by_user[uid] = UserPreference(
                user_id=uid,
                cinematic_priors=d.get("cinematic_priors", {}),
                style_priors=d.get("style_priors", []),
                physics_strictness=float(d.get("physics_strictness", 1.0)),
                endorsed_lesson_ids=d.get("endorsed_lesson_ids", []),
                rejected_lesson_ids=d.get("rejected_lesson_ids", []),
            )

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            uid: {
                "cinematic_priors": p.cinematic_priors,
                "style_priors": p.style_priors,
                "physics_strictness": p.physics_strictness,
                "endorsed_lesson_ids": p.endorsed_lesson_ids,
                "rejected_lesson_ids": p.rejected_lesson_ids,
            }
            for uid, p in self.by_user.items()
        }
        self.path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def get(self, user_id: str) -> UserPreference:
        if user_id not in self.by_user:
            self.by_user[user_id] = UserPreference(user_id=user_id)
        return self.by_user[user_id]

    def update_cinematic(self, user_id: str, key: str, value: str) -> None:
        pref = self.get(user_id)
        pref.cinematic_priors[key] = value
        self._persist()

    def endorse_lesson(self, user_id: str, lesson_id: str) -> None:
        pref = self.get(user_id)
        if lesson_id not in pref.endorsed_lesson_ids:
            pref.endorsed_lesson_ids.append(lesson_id)
            self._persist()

    def set_strictness(self, user_id: str, value: float) -> None:
        pref = self.get(user_id)
        pref.physics_strictness = float(value)
        self._persist()

    def __contains__(self, user_id: str) -> bool:
        return user_id in self.by_user
