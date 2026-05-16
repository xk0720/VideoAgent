"""Persistent NarrativeMemory store.

Backends:
    • SQLite       — stdlib ``sqlite3`` for metadata
    • FAISS        — ``faiss-cpu`` for CLIP ANN index (optional; falls back to
                     numpy cosine search when faiss is not importable)
    • numpy .npz   — heavy per-shot ShotFeatures arrays

The store is process-local; we don't try to be multi-writer safe.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..types import (
    Character,
    CinematographyTags,
    Event,
    MusicProfile,
    MusicSection,
    NarrativeMemory,
    Shot,
    ShotFeatures,
    Story,
)
from .schema import StoreSchemas

# ── Optional FAISS ────────────────────────────────────────────────
try:                                                            # pragma: no cover
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    faiss = None                                                # type: ignore[assignment]
    _HAS_FAISS = False


class MemoryStore:
    """Read/write narrative memory to disk.

    Layout under ``root``:
        root/
          memory.sqlite
          features/<shot_id>.npz
          embeddings.npy         (N, D) parallel to ``embeddings_ids.json``
          embeddings_ids.json    list[str] aligned with embeddings.npy rows
          faiss.index            (only if faiss is installed)
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "features").mkdir(exist_ok=True)
        self.db_path = self.root / "memory.sqlite"
        self.embed_path = self.root / "embeddings.npy"
        self.embed_ids_path = self.root / "embeddings_ids.json"
        self.faiss_path = self.root / "faiss.index"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        StoreSchemas.apply(self._conn)
        self._faiss_index = None
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_ids: list[str] = []
        self._load_embeddings()

    # ─────── public API ───────

    def add_shot(self, shot: Shot, save_features: bool = True) -> None:
        cur = self._conn.cursor()
        feat_path: Optional[str] = None
        if save_features and shot.features is not None:
            feat_path = self._dump_features(shot.shot_id, shot.features)
        cin = shot.cinematography
        cur.execute(
            """
            INSERT OR REPLACE INTO shots
              (shot_id, source_video, start_time, end_time, caption,
               shot_scale, shot_movement, shot_angle, framing,
               avg_flow_mag, dialogue, feature_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                shot.shot_id, shot.source_video, shot.start_time, shot.end_time,
                shot.caption, cin.shot_scale, cin.shot_movement, cin.shot_angle, cin.framing,
                (shot.features.avg_flow_magnitude if shot.features else 0.0),
                shot.dialogue, feat_path,
            ),
        )
        for cid in shot.character_ids:
            cur.execute(
                "INSERT OR IGNORE INTO shot_characters(shot_id, character_id) VALUES (?, ?)",
                (shot.shot_id, cid),
            )
        # add embedding to the flat numpy index
        if shot.features is not None and shot.features.clip_embedding is not None:
            self._append_embedding(shot.shot_id, shot.features.clip_embedding)
        self._conn.commit()

    def add_event(self, event: Event) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?)",
            (event.event_id, event.summary, event.start_time, event.end_time,
             json.dumps(event.shot_ids)),
        )
        self._conn.commit()

    def add_story(self, story: Story) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO stories VALUES (?, ?, ?, ?, ?)",
            (story.story_id, story.title, story.summary, story.arc_role,
             json.dumps(story.event_ids)),
        )
        self._conn.commit()

    def add_character(self, char: Character) -> None:
        face_p = self._dump_array(f"face_{char.character_id}.npy", char.face_embedding)
        voice_p = self._dump_array(f"voice_{char.character_id}.npy", char.voice_embedding)
        self._conn.execute(
            "INSERT OR REPLACE INTO characters VALUES (?, ?, ?, ?, ?, ?)",
            (char.character_id, char.name, json.dumps(char.appearance_shot_ids),
             char.profile_summary, face_p, voice_p),
        )
        self._conn.commit()

    def set_music_profile(self, mp: MusicProfile) -> None:
        beats_p = self.root / "music_beats.npy"
        np.save(beats_p, np.array(mp.beats, dtype=np.float32))
        secs = [
            {"name": s.name, "start_time": s.start_time, "end_time": s.end_time,
             "energy_db": s.energy_db, "num_beats": s.num_beats} for s in mp.sections
        ]
        self._conn.execute(
            "INSERT OR REPLACE INTO music VALUES (?, ?, ?, ?, ?)",
            (str(mp.audio_path) if mp.audio_path else "", mp.duration, mp.bpm,
             str(beats_p), json.dumps(secs)),
        )
        self._conn.commit()

    # ─────── reads / materializations ───────

    def get_shot(self, shot_id: str, load_features: bool = False) -> Optional[Shot]:
        row = self._conn.execute("SELECT * FROM shots WHERE shot_id = ?", (shot_id,)).fetchone()
        if not row:
            return None
        cins = CinematographyTags(
            shot_scale=row["shot_scale"], shot_movement=row["shot_movement"],
            shot_angle=row["shot_angle"], framing=row["framing"],
        )
        features: Optional[ShotFeatures] = None
        if load_features and row["feature_path"]:
            features = self._load_features(row["feature_path"])
        char_ids = [r["character_id"] for r in self._conn.execute(
            "SELECT character_id FROM shot_characters WHERE shot_id = ?", (shot_id,))]
        return Shot(
            shot_id=row["shot_id"], source_video=row["source_video"],
            start_time=row["start_time"], end_time=row["end_time"],
            caption=row["caption"] or "", cinematography=cins,
            features=features, character_ids=char_ids, dialogue=row["dialogue"],
        )

    def all_shot_ids(self) -> list[str]:
        return [r["shot_id"] for r in self._conn.execute("SELECT shot_id FROM shots")]

    def load_full_memory(self, load_features: bool = False) -> NarrativeMemory:
        nm = NarrativeMemory()
        for sid in self.all_shot_ids():
            shot = self.get_shot(sid, load_features=load_features)
            if shot:
                nm.shots[sid] = shot
        for r in self._conn.execute("SELECT * FROM events"):
            nm.events[r["event_id"]] = Event(
                event_id=r["event_id"], shot_ids=json.loads(r["shot_ids"] or "[]"),
                summary=r["summary"] or "", start_time=r["start_time"] or 0.0,
                end_time=r["end_time"] or 0.0,
            )
        for r in self._conn.execute("SELECT * FROM stories"):
            nm.stories[r["story_id"]] = Story(
                story_id=r["story_id"], title=r["title"] or "",
                event_ids=json.loads(r["event_ids"] or "[]"),
                summary=r["summary"] or "", arc_role=r["arc_role"] or "rising",
            )
        for r in self._conn.execute("SELECT * FROM characters"):
            face = self._load_array(r["face_emb_path"])
            voice = self._load_array(r["voice_emb_path"])
            nm.characters[r["character_id"]] = Character(
                character_id=r["character_id"], name=r["name"],
                face_embedding=face, voice_embedding=voice,
                appearance_shot_ids=json.loads(r["appearance_shot_ids"] or "[]"),
                profile_summary=r["profile_summary"] or "",
            )
        row = self._conn.execute("SELECT * FROM music LIMIT 1").fetchone()
        if row:
            secs = [MusicSection(**d) for d in json.loads(row["sections"] or "[]")]
            beats = np.load(row["beats_path"]).tolist() if row["beats_path"] else []
            nm.music_profile = MusicProfile(
                audio_path=Path(row["audio_path"]) if row["audio_path"] else None,
                duration=row["duration"], bpm=row["bpm"], beats=beats,
                sections=secs,
            )
        return nm

    # ─────── embedding / ANN  ───────

    def search_by_embedding(self, query_emb: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Return list of (shot_id, cosine_similarity) sorted desc."""
        if self._embeddings is None or len(self._embedding_ids) == 0:
            return []
        q = query_emb.astype("float32").reshape(1, -1)
        if _HAS_FAISS and self._faiss_index is not None:
            faiss.normalize_L2(q)                                # type: ignore[attr-defined]
            D, I = self._faiss_index.search(q, min(top_k, len(self._embedding_ids)))
            return [(self._embedding_ids[i], float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        # numpy fallback (cosine similarity on L2-normalised vectors).
        emb = self._embeddings
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims = (emb_n @ qn.T).ravel()
        idx = np.argsort(-sims)[:top_k]
        return [(self._embedding_ids[i], float(sims[i])) for i in idx]

    # ─────── internals ───────

    def _dump_features(self, shot_id: str, feats: ShotFeatures) -> str:
        p = self.root / "features" / f"{shot_id}.npz"
        np.savez_compressed(
            p,
            clip_embedding=feats.clip_embedding,
            start_flow=feats.start_flow if feats.start_flow is not None else np.zeros(0),
            end_flow=feats.end_flow if feats.end_flow is not None else np.zeros(0),
            start_saliency=feats.start_saliency if feats.start_saliency is not None else np.zeros(0),
            end_saliency=feats.end_saliency if feats.end_saliency is not None else np.zeros(0),
            avg_flow_magnitude=np.array([feats.avg_flow_magnitude], dtype=np.float32),
        )
        return str(p.relative_to(self.root))

    def _load_features(self, rel_path: str) -> ShotFeatures:
        npz = np.load(self.root / rel_path, allow_pickle=False)
        def opt(name: str) -> Optional[np.ndarray]:
            a = npz[name]
            return None if a.size == 0 else a
        return ShotFeatures(
            clip_embedding=npz["clip_embedding"],
            start_flow=opt("start_flow"),
            end_flow=opt("end_flow"),
            start_saliency=opt("start_saliency"),
            end_saliency=opt("end_saliency"),
            avg_flow_magnitude=float(npz["avg_flow_magnitude"][0]),
        )

    def _dump_array(self, name: str, arr: Optional[np.ndarray]) -> Optional[str]:
        if arr is None:
            return None
        p = self.root / name
        np.save(p, arr)
        return str(p)

    def _load_array(self, path: Optional[str]) -> Optional[np.ndarray]:
        if not path:
            return None
        try:
            return np.load(path)
        except FileNotFoundError:
            return None

    def _append_embedding(self, shot_id: str, emb: np.ndarray) -> None:
        emb = emb.astype("float32").reshape(-1)
        if self._embeddings is None:
            self._embeddings = emb[None, :]
        else:
            if self._embeddings.shape[1] != emb.shape[0]:
                raise ValueError(
                    f"Embedding dim mismatch: index has {self._embeddings.shape[1]} "
                    f"vs new {emb.shape[0]}"
                )
            self._embeddings = np.vstack([self._embeddings, emb[None, :]])
        self._embedding_ids.append(shot_id)
        np.save(self.embed_path, self._embeddings)
        self.embed_ids_path.write_text(json.dumps(self._embedding_ids))
        if _HAS_FAISS:                                          # pragma: no cover
            self._rebuild_faiss()

    def _load_embeddings(self) -> None:
        if self.embed_path.exists() and self.embed_ids_path.exists():
            self._embeddings = np.load(self.embed_path)
            self._embedding_ids = json.loads(self.embed_ids_path.read_text())
        if _HAS_FAISS and self._embeddings is not None:        # pragma: no cover
            self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:                          # pragma: no cover
        assert self._embeddings is not None
        d = self._embeddings.shape[1]
        idx = faiss.IndexFlatIP(d)                              # cosine via normalised IP
        emb = self._embeddings.astype("float32").copy()
        faiss.normalize_L2(emb)
        idx.add(emb)
        self._faiss_index = idx
        faiss.write_index(idx, str(self.faiss_path))

    def close(self) -> None:
        self._conn.close()


def add_shots(store: MemoryStore, shots: Iterable[Shot]) -> None:
    for s in shots:
        store.add_shot(s)


__all__ = ["MemoryStore", "add_shots"]
