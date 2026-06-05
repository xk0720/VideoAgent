"""EntityStore — C8 Tier-4 cross-run entity persistence (v0.3).

Borrowed pattern: VideoMemory (arXiv:2601.03655) Dynamic Memory Bank, which
keeps character/prop/background descriptors persistent across SHOTS within
one run. Maestro extends this to **cross-RUN** persistence — the same hero
generated on Day 2 reuses Day-1's face / style anchor + accumulated physics
profile. Differentiates from VideoMemory which is per-run only.

Dedup uses embedding cosine; under the mock embedder identical asset stems
collapse to the same vector and thus reuse the same entity_id deterministically.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..types import PersistentEntity


def _stable_entity_id(canonical_name: str) -> str:
    h = hashlib.md5(canonical_name.encode("utf-8")).hexdigest()
    return f"E{h[:12]}"


class EntityStore:
    """Cross-run dedup keyed on canonical_name AND embedding similarity.

    JSONL-backed. `find_or_create(canonical_name, embedding)`:
      • if any existing entity's embedding cosine ≥ MATCH_THRESHOLD against
        the candidate → return that existing entity (cross-run reuse);
      • else create + persist.
    """

    MATCH_THRESHOLD = 0.85   # high bar to avoid false reuse

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.entities: list[PersistentEntity] = []
        self._by_id: dict[str, PersistentEntity] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ent = PersistentEntity(
                entity_id=d["entity_id"],
                canonical_name=d["canonical_name"],
                embedding=embed_text(d["canonical_name"]),
                source_paths=d.get("source_paths", []),
                style_descriptors=d.get("style_descriptors", {}),
                appearance_log=d.get("appearance_log", []),
                physics_profile=d.get("physics_profile", {}),
                first_seen_ts=float(d.get("first_seen_ts", 0.0)),
                last_seen_ts=float(d.get("last_seen_ts", 0.0)),
            )
            self.entities.append(ent)
            self._by_id[ent.entity_id] = ent

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self.entities:
                f.write(json.dumps({
                    "entity_id": e.entity_id,
                    "canonical_name": e.canonical_name,
                    "source_paths": e.source_paths,
                    "style_descriptors": e.style_descriptors,
                    "appearance_log": e.appearance_log,
                    "physics_profile": e.physics_profile,
                    "first_seen_ts": e.first_seen_ts,
                    "last_seen_ts": e.last_seen_ts,
                }, ensure_ascii=False) + "\n")

    def find_or_create(
        self,
        canonical_name: str,
        source_path: str = "",
        task_id: str = "",
        bbox: Optional[list[float]] = None,
    ) -> PersistentEntity:
        emb = embed_text(canonical_name)
        # First, look for a strong embedding match.
        for ent in self.entities:
            if ent.embedding is None:
                continue
            if cosine(emb, ent.embedding) >= self.MATCH_THRESHOLD:
                # Cross-run reuse: log the new appearance + update timestamps.
                if source_path and source_path not in ent.source_paths:
                    ent.source_paths.append(source_path)
                ent.appearance_log.append(
                    {"task_id": task_id, "source_path": source_path,
                     "bbox": bbox or [], "ts": time.time()}
                )
                ent.last_seen_ts = time.time()
                self._persist()
                return ent
        # Otherwise, create new.
        ent = PersistentEntity(
            entity_id=_stable_entity_id(canonical_name),
            canonical_name=canonical_name,
            embedding=emb,
            source_paths=[source_path] if source_path else [],
            appearance_log=[{"task_id": task_id, "source_path": source_path,
                             "bbox": bbox or [], "ts": time.time()}],
            first_seen_ts=time.time(),
            last_seen_ts=time.time(),
        )
        self.entities.append(ent)
        self._by_id[ent.entity_id] = ent
        self._persist()
        return ent

    def get(self, entity_id: str) -> Optional[PersistentEntity]:
        return self._by_id.get(entity_id)

    def __len__(self) -> int:
        return len(self.entities)
