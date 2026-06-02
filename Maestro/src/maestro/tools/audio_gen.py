"""AudioGenTool — generation category. TTS / music / sound-effect synthesis stub.

UniVA bundles audio generation (TTS, music, SFX) so a Director can ask for a
narration line or a background motif. Maestro v0.2.2 mock writes a placeholder
mp3 metadata file; v0.3 wires Bark / MusicGen / a hosted API behind the same
`run` signature.
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseTool


class AudioGenTool(BaseTool):
    name = "audio_gen"
    category = "generation"
    description = "Synthesize TTS narration or short music/SFX clips; returns the output path."
    side_effects = True

    def run(
        self,
        prompt: str,
        out_path: str | Path,
        duration: float = 4.0,
        kind: str = "music",   # "music" | "tts" | "sfx"
        seed: int = 0,
    ) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Mock body (real backends overwrite with actual audio bytes).
        out.write_text(
            f"MOCK AUDIO\nkind={kind}\nprompt={prompt}\nduration={duration}\nseed={seed}\n",
            encoding="utf-8",
        )
        return out
