"""RetrievalTool — ground generation in the user's uploaded assets (E1).

Borrowed (cite):
  • ViMax — asset indexing: index frames/reference images + embeddings, retrieve
    along the timeline to keep long-video identity/style consistent.
  • DIRECT — CLIP-similarity shot retrieval over a memory pool.

OUR use: this is how Maestro keeps cross-shot identity/style consistent in a
*generation* (not editing) pipeline — the Generator pulls identity/style anchor
images from AssetMemory to condition the video model, and the loop can retrieve
source shots as visual references for grounding. Reuses the old repo's narrative
memory idea, repurposed for generation grounding.
"""
from __future__ import annotations

from pathlib import Path

from ..embeddings import cosine, embed_text
from ..types import AssetMemory
from .base import BaseTool


class RetrievalTool(BaseTool):
    name = "retrieve_assets"

    def __init__(self, asset_memory: AssetMemory):
        self.memory = asset_memory

    def retrieve_identity_refs(self, identity_refs: list[str]) -> list[Path]:
        out: list[Path] = []
        for rid in identity_refs:
            ident = self.memory.identity_anchors.get(rid)
            if ident and ident.source:
                out.append(Path(ident.source))
        return out

    def retrieve_style_refs(self, style_refs: list[str]) -> list[Path]:
        known = {s.style_id: s for s in self.memory.style_anchors}
        return [Path(known[s].source) for s in style_refs if s in known and known[s].source]

    def retrieve_source_shots(self, query: str, top_k: int = 3) -> list[str]:
        """DIRECT-style semantic retrieval over source shots -> shot_ids."""
        shots = list(self.memory.video_shots.values())
        if not shots:
            return []
        q = embed_text(query)
        scored = [
            (cosine(q, s.clip_embedding) if s.clip_embedding is not None else 0.0, s.shot_id)
            for s in shots
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in scored[:top_k]]

    def run(self, identity_refs: list[str]) -> list[Path]:  # BaseTool contract
        return self.retrieve_identity_refs(identity_refs)
