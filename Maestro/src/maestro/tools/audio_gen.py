"""AudioGenTool — generation category. TTS / foley / music synthesis.

UniVA bundles audio generation so a Director can ask for a narration line, an
ambient foley bed, or a background motif. Maestro v0.4 delegates to a configured
`BaseAudioGenClient` (models/audio_gen_backends.py):
  • kind="tts"   → client.speech(...)   REAL (MiniMax speech-2.5 via WaveSpeed)
  • kind="foley" → client.foley(...)    REAL (MMAudio-v2 via WaveSpeed; needs a video)
  • kind="music" → deterministic MOCK placeholder (no real music endpoint in
    UniVA's verified wavespeed_api.py yet — kept honest as a stub)

The default client is MockAudioGenClient, so `default_registry()` and every
existing test stay byte-for-byte unchanged. The `run` signature keeps `out_path`
as the 2nd positional arg for back-compat; `video_path` is a new keyword.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.audio_gen_backends import BaseAudioGenClient, MockAudioGenClient
from .base import BaseTool


class AudioGenTool(BaseTool):
    name = "audio_gen"
    category = "generation"
    description = "Synthesize TTS narration, video foley, or short music clips; returns the output path."
    side_effects = True

    def __init__(self, client: Optional[BaseAudioGenClient] = None):
        # Default mock keeps the key-free CPU path identical to v0.2.2.
        self.client: BaseAudioGenClient = client or MockAudioGenClient()

    def run(
        self,
        prompt: str,
        out_path: str | Path,
        duration: float = 4.0,
        kind: str = "music",   # "music" | "tts" | "foley"
        seed: int = 0,
        video_path: str | Path | None = None,
        **kwargs,
    ) -> Path:
        out = Path(out_path)
        if kind == "tts":
            return self.client.speech(prompt, out, seed=seed)
        if kind == "foley":
            if video_path is None:
                raise ValueError(
                    "audio_gen kind='foley' needs a `video_path` (MMAudio scores "
                    "the moving pixels to produce ambient sound)."
                )
            return self.client.foley(
                video_path, prompt, out, duration=max(1, int(round(duration))),
                seed=seed,
            )
        # "music" / unknown → deterministic mock placeholder (no real backend yet).
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"MOCK AUDIO\nkind={kind}\nprompt={prompt}\nduration={duration}\nseed={seed}\n",
            encoding="utf-8",
        )
        return out
