"""Real audio-generation backends (v0.4).

Two capabilities are REAL (hosted WaveSpeed API, no local GPU, training-free):
  • foley  — video → ambient sound (MMAudio-v2 scores the moving pixels)
  • speech — text → narration (MiniMax speech-2.5 TTS)
Music remains a PLACEHOLDER: UniVA's verified `utils/wavespeed_api.py` exposes
no music endpoint, so `supported_kinds()` honestly omits it on the real client
and the AudioGenTool keeps emitting the deterministic mock motif for now.

The submit → poll predictions/{id}/result → download outputs[0] protocol and the
exact endpoint paths / payload field names are PORTED from UniVA
(2511.08521, utils/wavespeed_api.py functions `audio_gen` and `speech_gen`) —
they are proven against the live API. We keep Maestro's house style: an ABC +
factory, a loud-on-missing-key `_headers()`, lazy `import requests`/`base64`
inside methods, and methods that return a `Path` or raise (never a dict).

All clients run on the SINGLE $WAVESPEED_API_KEY Maestro already uses for video.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# ABC — the contract the pipeline / AudioGenTool delegates to
# ─────────────────────────────────────────────────────────────
class BaseAudioGenClient(ABC):
    @abstractmethod
    def foley(
        self,
        video_path: Path,
        prompt: str,
        out_path: Path,
        duration: int = 5,
        seed: int = 0,
    ) -> Path:
        """video → audio (ambient/foley). Needs a VIDEO to score."""
        ...

    @abstractmethod
    def speech(
        self,
        text: str,
        out_path: Path,
        voice_id: str = "Wise_Woman",
        emotion: str = "neutral",
        speed: float = 1.0,
        seed: int = 0,
    ) -> Path:
        """text → speech (TTS narration)."""
        ...

    @abstractmethod
    def supported_kinds(self) -> set[str]:
        """Which audio kinds this client REALLY produces ("foley"|"tts"|"music")."""
        ...


# ─────────────────────────────────────────────────────────────
# Mock — unchanged placeholder bytes (keeps the default path light)
# ─────────────────────────────────────────────────────────────
class MockAudioGenClient(BaseAudioGenClient):
    """Writes the same placeholder bytes the v0.2.2 mock AudioGenTool did, so the
    default (key-free, CPU-only) path is byte-for-byte unchanged."""

    def __init__(self, name: str = "mock-audio-gen"):
        self.name = name

    def foley(
        self,
        video_path: Path,
        prompt: str,
        out_path: Path,
        duration: int = 5,
        seed: int = 0,
    ) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"MOCK AUDIO\nkind=foley\nprompt={prompt}\nduration={duration}\nseed={seed}\n",
            encoding="utf-8",
        )
        return out

    def speech(
        self,
        text: str,
        out_path: Path,
        voice_id: str = "Wise_Woman",
        emotion: str = "neutral",
        speed: float = 1.0,
        seed: int = 0,
    ) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"MOCK AUDIO\nkind=tts\nprompt={text}\nduration={0.0}\nseed={seed}\n",
            encoding="utf-8",
        )
        return out

    def supported_kinds(self) -> set[str]:
        # The mock fakes all three; nothing is real.
        return {"foley", "tts", "music"}


# ─────────────────────────────────────────────────────────────
# WaveSpeed — hosted API audio (no local GPU; UniVA's verified route)
# ─────────────────────────────────────────────────────────────
class WaveSpeedAudioClient(BaseAudioGenClient):
    """WaveSpeed REST audio backend. Ports UniVA `utils/wavespeed_api.py`:

      • foley  ← `audio_gen`  → POST {BASE}/wavespeed-ai/mmaudio-v2
      • speech ← `speech_gen` → POST {BASE}/minimax/speech-2.5-turbo-preview

    Both follow the same submit → poll predictions/{id}/result → download
    outputs[0] protocol as WaveSpeedClient (video). One key powers everything.

    config (models.audio_gen):
      name: "wavespeed"
      foley_model_id:  "wavespeed-ai/mmaudio-v2"
      speech_model_id: "minimax/speech-2.5-turbo-preview"
      voice_id: "Wise_Woman"
      api_key: ...          # or $WAVESPEED_API_KEY
      poll_interval: 2.0
      timeout: 600
    """

    BASE = "https://api.wavespeed.ai/api/v3"

    def __init__(self, name: str = "wavespeed", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("WAVESPEED_API_KEY")
        self.foley_model_id = self.config.get("foley_model_id", "wavespeed-ai/mmaudio-v2")
        self.speech_model_id = self.config.get(
            "speech_model_id", "minimax/speech-2.5-turbo-preview"
        )
        self.default_voice = self.config.get("voice_id", "Wise_Woman")
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.timeout = float(self.config.get("timeout", 600))

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "WaveSpeedAudioClient needs an API key: set $WAVESPEED_API_KEY or "
                "models.audio_gen.api_key (or switch back to 'mock-audio-gen')."
            )
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    # ── shared submit → poll → download (UniVA protocol) ──
    def _run_task(self, model_id: str, payload: dict, out_path: Path) -> Path:
        import requests  # in our [all] extras; loud ImportError otherwise

        resp = requests.post(
            f"{self.BASE}/{model_id}", json=payload, headers=self._headers(),
            timeout=60,
        )
        resp.raise_for_status()
        task_id = resp.json()["data"]["id"]

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = requests.get(
                f"{self.BASE}/predictions/{task_id}/result",
                headers=self._headers(), timeout=30,
            )
            r.raise_for_status()
            data = r.json()["data"]
            status = data.get("status")
            if status == "completed":
                url = data["outputs"][0]
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                audio = requests.get(url, timeout=120)
                audio.raise_for_status()
                out_path.write_bytes(audio.content)
                return out_path
            if status == "failed":
                raise RuntimeError(f"WaveSpeed audio task {task_id} failed: "
                                   f"{data.get('error', 'unknown error')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"WaveSpeed audio task {task_id} did not finish within "
                           f"{self.timeout}s")

    def foley(
        self,
        video_path: Path,
        prompt: str,
        out_path: Path,
        duration: int = 5,
        seed: int = 0,
    ) -> Path:
        """video → audio via MMAudio-v2 (ported from UniVA `audio_gen`).

        MMAudio scores the moving pixels, so a real VIDEO is required. A LOCAL
        path is read + base64-encoded into a data:video/<mime>;base64,... URI
        exactly like UniVA's os.path.exists branch; an http(s) URL is passed
        through. A missing/non-video path raises (never silently produces
        nothing)."""
        import base64

        vp = str(video_path)
        if vp.startswith("http://") or vp.startswith("https://"):
            video_data_uri = vp
        else:
            p = Path(vp)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(
                    f"foley() needs a video file to score; not found: {vp}"
                )
            ext = p.suffix.lower()
            if ext not in (".mp4", ".mov", ".avi", ".mkv"):
                raise ValueError(
                    f"foley() needs a video (.mp4/.mov/.avi/.mkv) to score, got '{ext}': {vp}"
                )
            mime = "mp4"  # MMAudio accepts mp4 container; matches UniVA default
            video_b64 = base64.b64encode(p.read_bytes()).decode()
            video_data_uri = f"data:video/{mime};base64,{video_b64}"

        payload = {
            "duration": max(1, int(round(duration))),
            "guidance_scale": 4.5,
            "mask_away_clip": False,
            "negative_prompt": "",
            "num_inference_steps": 25,
            "prompt": prompt,
            "video": video_data_uri,
        }
        return self._run_task(self.foley_model_id, payload, out_path)

    def speech(
        self,
        text: str,
        out_path: Path,
        voice_id: str = "Wise_Woman",
        emotion: str = "neutral",
        speed: float = 1.0,
        seed: int = 0,
    ) -> Path:
        """text → speech via MiniMax speech-2.5 (ported from UniVA `speech_gen`)."""
        payload = {
            "emotion": emotion,
            "english_normalization": False,
            "pitch": 0,
            "speed": speed,
            "text": text,
            "voice_id": voice_id or self.default_voice,
            "volume": 1,
        }
        return self._run_task(self.speech_model_id, payload, out_path)

    def supported_kinds(self) -> set[str]:
        # Music is NOT real — UniVA's verified file has no music endpoint.
        return {"foley", "tts"}


# ─────────────────────────────────────────────────────────────
# Factory — mirrors models/video_gen.build_video_gen
# ─────────────────────────────────────────────────────────────
def build_audio_gen(spec: str | dict | None) -> BaseAudioGenClient:
    """None / "mock*" → MockAudioGenClient; "wavespeed"/"mmaudio"/"minimax" →
    WaveSpeedAudioClient; anything else → ValueError."""
    name = "mock-audio-gen"
    config: dict = {}
    if isinstance(spec, dict):
        name = spec.get("name", name)
        config = spec
    elif isinstance(spec, str):
        name = spec
    key = (name or "").lower()
    if key.startswith("mock"):
        return MockAudioGenClient(name=name)
    if key in ("wavespeed", "mmaudio", "minimax"):
        return WaveSpeedAudioClient(name=name, config=config)
    raise ValueError(
        f"Unknown audio_gen backend '{name}'. Known: mock*, wavespeed, mmaudio, minimax"
    )
