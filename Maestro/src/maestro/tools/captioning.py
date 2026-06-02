"""CaptioningTool — analysis category. Caption a video or image (VLM stub).

Used to ground planning in actual visual content of user-provided assets — what
UniVA's Analysis tools do via Qwen2.5-VL. v0.2.2 mock returns a deterministic
keyword-derived caption; v0.3 wires a real MLLM behind the same `run` signature.
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseTool


class CaptioningTool(BaseTool):
    name = "caption"
    category = "analysis"
    description = "Generate a natural-language caption for an image or video file."

    def run(self, media: str | Path, kind: str = "auto") -> str:
        p = Path(media)
        stem = p.stem.replace("_", " ").replace("-", " ")
        if kind == "auto":
            kind = "video" if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"} else "image"
        # Deterministic mock so tests can assert structure. Real VLM swap-in
        # only changes this body.
        return f"a {kind} of {stem or 'unknown subject'}"
