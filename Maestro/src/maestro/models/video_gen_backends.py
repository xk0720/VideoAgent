"""Real video-generation backends (v0.4).

These implement the SAME `BaseVideoGenClient.generate(...)` contract as the
mock, so the rest of the data flow is unchanged — flip
`models.video_gen.name` in the config and (for local backends) fill the TODO
bodies.

Conditioning contract (v0.4): prompt + first_frame (C2 keyframe anchor) +
reference_images (E1 identity/style). There is NO physics control signal —
the sketch-as-controller line is dead; physics is VERIFIED from generated
pixels (physics/), never injected into the generator.

Backends:
  • WaveSpeedClient — hosted REST API (submit → poll → download), the same
    service UniVA (2511.08521) uses for all its generation. This one is FULLY
    IMPLEMENTED: with $WAVESPEED_API_KEY set, `maestro` produces real pixels
    with no local GPU — the fastest route to the minimal real chain.
  • OmniWeavingClient / WanClient — local-weights skeletons (fill + GPU).
  • VeoApiClient — hosted API skeleton.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from .video_gen import BaseVideoGenClient


# ─────────────────────────────────────────────────────────────
# WaveSpeed — hosted API backend (no local GPU; UniVA's route)
# ─────────────────────────────────────────────────────────────
class WaveSpeedClient(BaseVideoGenClient):
    """WaveSpeed REST API (https://wavespeed.ai). Pattern mirrors UniVA's
    `utils/wavespeed_api.py`: POST the task → poll predictions/{id}/result →
    download the output URL to `out_path`.

    config:
      models.video_gen:
        name: "wavespeed"
        model_id: "bytedance/seedance-v1-pro-t2v-480p"   # or any t2v/i2v id
        api_key: ...          # or $WAVESPEED_API_KEY
        poll_interval: 2.0
        timeout: 600
    """

    BASE = "https://api.wavespeed.ai/api/v3"

    def __init__(self, name: str = "wavespeed", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("WAVESPEED_API_KEY")
        self.model_id = self.config.get("model_id", "bytedance/seedance-v1-pro-t2v-480p")
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.timeout = float(self.config.get("timeout", 600))

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "WaveSpeedClient needs an API key: set $WAVESPEED_API_KEY or "
                "models.video_gen.api_key (or switch back to 'mock-video-gen')."
            )
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        import base64

        import requests  # std in our [all] extras; loud ImportError otherwise

        payload: dict = {"prompt": prompt, "duration": max(1, int(round(duration))),
                         "seed": seed}
        model_id = self.model_id
        if first_frame is not None and Path(first_frame).exists():
            payload["image"] = "data:image/png;base64," + base64.b64encode(
                Path(first_frame).read_bytes()
            ).decode()
            # i2v variant if the configured id is the t2v one
            model_id = model_id.replace("-t2v-", "-i2v-")

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
                video = requests.get(url, timeout=120)
                video.raise_for_status()
                out_path.write_bytes(video.content)
                return out_path
            if status == "failed":
                raise RuntimeError(f"WaveSpeed task {task_id} failed: "
                                   f"{data.get('error', 'unknown error')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"WaveSpeed task {task_id} did not finish within "
                           f"{self.timeout}s")

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}


# ─────────────────────────────────────────────────────────────
# OmniWeaving — preferred local backend (multi-condition native)
# ─────────────────────────────────────────────────────────────
class OmniWeavingClient(BaseVideoGenClient):
    """Tencent-Hunyuan/OmniWeaving. Native text + multi-image conditioning:
    first-frame anchor (C2) + reference images (E1) + text."""

    def __init__(self, name: str = "omniweaving", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.weights_path = self.config.get("weights_path") or os.getenv("OMNIWEAVING_WEIGHTS")
        self.device = self.config.get("device", "cuda")
        self._pipe = None  # lazy-loaded

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        # TODO(real): load the real pipeline, e.g.
        #   import torch; from omniweaving import OmniWeavingPipeline
        #   self._pipe = OmniWeavingPipeline.from_pretrained(self.weights_path).to(self.device)
        raise RuntimeError(
            "OmniWeavingClient not wired yet. Set models.video_gen.weights_path "
            "(or $OMNIWEAVING_WEIGHTS) and implement _ensure_loaded()/generate(). "
            "Until then use 'wavespeed' (API) or 'mock-video-gen'."
        )

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        self._ensure_loaded()
        # TODO(real): run the pipeline and write `out_path`, e.g.
        #   frames = self._pipe(prompt=prompt, image=first_frame,
        #                       ref_images=reference_images,
        #                       num_frames=int(duration*fps), seed=seed)
        #   write_video(out_path, frames, fps)
        raise NotImplementedError("OmniWeaving generate() not wired")

    def supported_conditions(self) -> set[str]:
        return {"first_frame", "reference_images"}


# ─────────────────────────────────────────────────────────────
# Wan — open-weight backup
# ─────────────────────────────────────────────────────────────
class WanClient(BaseVideoGenClient):
    """Wan2.x local I2V/T2V with first/last-frame anchoring. Backup to OmniWeaving."""

    def __init__(self, name: str = "wan", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.weights_path = self.config.get("weights_path") or os.getenv("WAN_WEIGHTS")
        self.device = self.config.get("device", "cuda")
        self._pipe = None

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        # TODO(real): load Wan pipeline; use first_frame as the I2V anchor;
        # write `out_path`.
        raise RuntimeError(
            "WanClient not wired yet. Set models.video_gen.weights_path (or "
            "$WAN_WEIGHTS) and implement generate()."
        )

    def supported_conditions(self) -> set[str]:
        return {"first_frame", "reference_images"}


# ─────────────────────────────────────────────────────────────
# API fallback — Veo / Sora (no local weights)
# ─────────────────────────────────────────────────────────────
class VeoApiClient(BaseVideoGenClient):
    """Hosted API fallback (text + optional first image)."""

    def __init__(self, name: str = "veo-api", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("VEO_API_KEY")

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        if not self.api_key:
            raise RuntimeError("VeoApiClient needs $VEO_API_KEY (or config api_key).")
        # TODO(real): call the hosted endpoint, poll, download to out_path.
        raise NotImplementedError("Veo generate() not wired")

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}


_REGISTRY = {
    "wavespeed": WaveSpeedClient,
    "omniweaving": OmniWeavingClient,
    "wan": WanClient,
    "veo": VeoApiClient,
    "veo-api": VeoApiClient,
    "sora": VeoApiClient,
}


def build_real_video_gen(name: str, config: Optional[dict] = None) -> BaseVideoGenClient:
    key = name.split("-")[0].lower() if name else ""
    cls = _REGISTRY.get(name.lower()) or _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown video_gen backend '{name}'. Known: {list(_REGISTRY)}")
    return cls(name=name, config=config)
