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

        return self._run_task(model_id, payload, out_path)

    # ── shared submit → poll predictions/{id}/result → download (UniVA protocol) ──
    def _run_task(self, model_id: str, payload: dict, out_path: Path) -> Path:
        import requests  # std in our [all] extras; loud ImportError otherwise

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

    @staticmethod
    def _video_data_uri(video_path: Path) -> str:
        """Local video → data:video/mp4;base64,...; http(s) URL passes through.
        Matches UniVA's os.path.exists branch in audio_gen / vace_api."""
        import base64

        vp = str(video_path)
        if vp.startswith("http://") or vp.startswith("https://"):
            return vp
        p = Path(vp)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"video not found: {vp}")
        return "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()

    def frame_to_frame(
        self,
        prompt: str,
        first_frame: Path,
        last_frame: Path,
        out_path: Path,
        duration: int = 5,
        seed: int = 0,
    ) -> Path:
        """First-last-frame video (wan-flf2v). Ported from UniVA
        `frame_to_frame_video`: POST {BASE}/wavespeed-ai/wan-flf2v with both
        endpoint frames base64'd. Capability "flf2v" (optional; see
        BaseVideoGenClient.capabilities)."""
        import base64

        first_b64 = base64.b64encode(Path(first_frame).read_bytes()).decode()
        last_b64 = base64.b64encode(Path(last_frame).read_bytes()).decode()
        payload = {
            "duration": max(1, int(round(duration))),
            "enable_safety_checker": True,
            "first_image": f"data:image/jpeg;base64,{first_b64}",
            "guidance_scale": 5,
            "last_image": f"data:image/jpeg;base64,{last_b64}",
            "negative_prompt": "",
            "num_inference_steps": 30,
            "prompt": prompt,
            "seed": seed,
            "size": "832*480",
        }
        return self._run_task("wavespeed-ai/wan-flf2v", payload, out_path)

    def edit_video(
        self,
        prompt: str,
        video_path: Path,
        out_path: Path,
        backend: str = "runway",
        task: str = "depth",
        seed: int = 0,
    ) -> Path:
        """Edit existing footage. Capability "edit" (optional). Two ported routes:
          • backend="runway" → POST {BASE}/runwayml/gen4-aleph  (UniVA
            `runway_video_editing` — free-form prompt edit)
          • backend="vace"   → POST {BASE}/wavespeed-ai/wan-2.1-14b-vace (UniVA
            `vace_api` — structure-guided edit; `task` is depth/pose/etc.)
        """
        video_data_uri = self._video_data_uri(video_path)
        if backend == "runway":
            payload = {
                "aspect_ratio": "16:9",
                "prompt": prompt,
                "video": video_data_uri,
            }
            return self._run_task("runwayml/gen4-aleph", payload, out_path)
        if backend == "vace":
            payload = {
                "context_scale": 1,
                "duration": 5,
                "flow_shift": 16,
                "guidance_scale": 5,
                "images": [],
                "negative_prompt": "",
                "num_inference_steps": 40,
                "prompt": prompt,
                "seed": seed,
                "size": "1280*720",
                "task": task,
                "video": video_data_uri,
            }
            return self._run_task("wavespeed-ai/wan-2.1-14b-vace", payload, out_path)
        raise ValueError(f"edit_video backend must be 'runway' or 'vace', got '{backend}'")

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}

    def capabilities(self) -> set[str]:
        # Phase-2 routing seed: t2v/i2v via generate(), plus the optional
        # frame_to_frame (flf2v) and edit_video (edit) methods.
        return {"t2v", "i2v", "flf2v", "edit"}


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
