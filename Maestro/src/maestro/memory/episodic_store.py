"""EpisodicStore — C8 Tier-1 episodic memory with replay (v0.3).

The full trajectory JSONL already exists on disk after each task. What was
missing in v0.2.2 is *replay*: the next task never looks at past tasks.

This store keeps a lightweight `EpisodicTrace` summary per task (prompt +
metrics + path to trajectory) and lets the planner query "show me the K most-
similar past tasks". A-MEM-style: a new entry can be auto-linked to similar
older entries.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..types import EpisodicTrace


class EpisodicStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.traces: list[EpisodicTrace] = []
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tr = EpisodicTrace(
                task_id=d["task_id"],
                user_prompt=d["user_prompt"],
                timestamp=float(d["timestamp"]),
                trajectory_path=d.get("trajectory_path", ""),
                n_shots=int(d.get("n_shots", 0)),
                total_revisions=int(d.get("total_revisions", 0)),
                escalations=int(d.get("escalations", 0)),
                converged=bool(d.get("converged", True)),
                final_weighted_total=float(d.get("final_weighted_total", 0.0)),
                lessons_distilled=d.get("lessons_distilled", []),
                skills_distilled=d.get("skills_distilled", []),
                embedding=embed_text(d["user_prompt"]),
            )
            self.traces.append(tr)

    def append(
        self,
        task_id: str,
        user_prompt: str,
        trajectory_path: str = "",
        report: Optional[dict] = None,
        lessons_distilled: Optional[list[str]] = None,
        skills_distilled: Optional[list[str]] = None,
    ) -> EpisodicTrace:
        report = report or {}
        # Aggregate metrics across shots so we can rank past tasks by quality.
        shots = report.get("shots", [])
        avg_total = (
            sum(s.get("final_metrics", {}).get("weighted_total", 0.0)
                for s in shots) / max(1, len(shots))
        )
        tr = EpisodicTrace(
            task_id=task_id,
            user_prompt=user_prompt,
            timestamp=time.time(),
            trajectory_path=trajectory_path,
            n_shots=int(report.get("n_shots", len(shots))),
            total_revisions=sum(s.get("revisions_used", 0) for s in shots),
            escalations=sum(s.get("escalations", 0) for s in shots),
            converged=all(s.get("converged", True) for s in shots) if shots else True,
            final_weighted_total=round(avg_total, 4),
            lessons_distilled=list(lessons_distilled or []),
            skills_distilled=list(skills_distilled or []),
            embedding=embed_text(user_prompt),
        )
        self.traces.append(tr)
        self._persist_one(tr)
        return tr

    def _persist_one(self, tr: EpisodicTrace) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task_id": tr.task_id,
                "user_prompt": tr.user_prompt,
                "timestamp": tr.timestamp,
                "trajectory_path": tr.trajectory_path,
                "n_shots": tr.n_shots,
                "total_revisions": tr.total_revisions,
                "escalations": tr.escalations,
                "converged": tr.converged,
                "final_weighted_total": tr.final_weighted_total,
                "lessons_distilled": tr.lessons_distilled,
                "skills_distilled": tr.skills_distilled,
            }, ensure_ascii=False) + "\n")

    def similar_tasks(self, prompt: str, top_k: int = 3,
                      threshold: float = 0.2) -> list[EpisodicTrace]:
        """Find past tasks whose prompt is most similar to the current one.

        Down-weight diverged tasks (escalations > 0 OR not converged) so
        precedents the planner sees are the GOOD ones.
        """
        if not self.traces:
            return []
        q = embed_text(prompt)
        scored = []
        for tr in self.traces:
            if tr.embedding is None:
                continue
            sim = cosine(q, tr.embedding)
            quality = 1.0
            if tr.escalations > 0:
                quality *= 0.7
            if not tr.converged:
                quality *= 0.5
            scored.append((sim * quality, tr))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for s, t in scored[:top_k] if s >= threshold]

    def __len__(self) -> int:
        return len(self.traces)
