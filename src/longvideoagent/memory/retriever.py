"""Query-time interface to NarrativeMemory.

Open-source dependency: numpy only (FAISS used transparently via MemoryStore
when installed).

Public surface used by the rest of the system:
    • retrieve_by_query(query_text, top_k)   — semantic shot lookup
    • retrieve_by_character(character_id)
    • estimate_feasibility(query_text)       — used by DirectorAgent
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..types import NarrativeMemory, Shot
from .store import MemoryStore


# Encoder typing: text -> embedding ndarray (D,). v0.1 uses a deterministic
# hash-based mock; v0.2 swaps in CLIP text encoder from open_clip / transformers.
TextEncoder = Callable[[str], np.ndarray]


def _mock_text_encoder(dim: int = 512) -> TextEncoder:
    def _enc(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(dim).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)
    return _enc


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        text_encoder: Optional[TextEncoder] = None,
        embed_dim: int = 512,
    ) -> None:
        self.store = store
        self.encode = text_encoder or _mock_text_encoder(embed_dim)

    def retrieve_by_query(self, query: str, top_k: int = 20) -> list[tuple[Shot, float]]:
        q_emb = self.encode(query)
        hits = self.store.search_by_embedding(q_emb, top_k=top_k)
        out: list[tuple[Shot, float]] = []
        for sid, sim in hits:
            shot = self.store.get_shot(sid, load_features=False)
            if shot:
                out.append((shot, sim))
        return out

    def retrieve_by_character(self, character_id: str) -> list[Shot]:
        rows = self.store._conn.execute(
            "SELECT shot_id FROM shot_characters WHERE character_id = ?",
            (character_id,),
        )
        shots = []
        for r in rows:
            s = self.store.get_shot(r["shot_id"], load_features=False)
            if s:
                shots.append(s)
        return shots

    def estimate_feasibility(self, query: str, top_k: int = 20) -> float:
        """Rough proxy used by DirectorAgent: max cosine sim of top-K hits.
        0 → query not represented in memory; 1 → strong match.
        """
        hits = self.store.search_by_embedding(self.encode(query), top_k=top_k)
        if not hits:
            return 0.0
        # cosine sim ∈ [-1, 1]; map to [0, 1].
        top_sim = hits[0][1]
        return float(max(0.0, min(1.0, (top_sim + 1.0) / 2.0)))


def load_memory(store_root) -> NarrativeMemory:
    """Convenience: build a NarrativeMemory directly from a store on disk."""
    return MemoryStore(store_root).load_full_memory(load_features=False)


__all__ = ["MemoryRetriever", "load_memory", "TextEncoder"]
