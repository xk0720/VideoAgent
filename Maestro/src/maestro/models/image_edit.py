"""Keyframe local-edit wrapper (C2).

Two backends behind one `edit(keyframe, instruction, out_path)` contract:
  • MockImageEditClient — writes an edited-keyframe stub (default, key-free).
  • WaveSpeedImageEditClient — REAL instruction-driven image editing via
    bytedance/seedream-v4/edit (the route UniVA's `seedream_v4_edit` verified
    live; same submit → poll → download protocol and the SAME
    $WAVESPEED_API_KEY as video). This closes the tool-library gap where
    keyframe_edit — a headline repair tool — silently ran on a mock even in
    real runs.

The keyframe image is uploaded via the official media endpoint and passed by
URL (upload_media, shared with video/audio). A non-image / missing keyframe
raises loudly — never a silently faked edit.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseImageEditClient(ABC):
    @abstractmethod
    def edit(self, keyframe: Path, instruction: str, out_path: Path) -> Path:
        ...


class MockImageEditClient(BaseImageEditClient):
    def __init__(self, name: str = "mock-image-edit"):
        self.name = name

    def edit(self, keyframe: Path, instruction: str, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        src = Path(keyframe).read_text(encoding="utf-8") if Path(keyframe).exists() else ""
        out_path.write_text(
            f"MOCK EDITED KEYFRAME\nfrom={keyframe}\ninstruction={instruction}\n"
            f"prev={src[:120]}\n",
            encoding="utf-8",
        )
        return out_path


class WaveSpeedImageEditClient(BaseImageEditClient):
    """Instruction-driven image edit via bytedance/seedream-v4/edit.

    config (models.image_edit):
      name: "wavespeed"
      model_id: "bytedance/seedream-v4/edit"
      size: "2048*2048"
      api_key: ...          # or $WAVESPEED_API_KEY
      poll_interval: 2.0
      timeout: 300
    """

    BASE = "https://api.wavespeed.ai/api/v3"

    def __init__(self, name: str = "wavespeed", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("WAVESPEED_API_KEY")
        self.model_id = self.config.get("model_id", "bytedance/seedream-v4/edit")
        self.size = self.config.get("size", "2048*2048")
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.timeout = float(self.config.get("timeout", 300))

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "WaveSpeedImageEditClient needs an API key: set "
                "$WAVESPEED_API_KEY or models.image_edit.api_key "
                "(or switch back to 'mock-image-edit')."
            )
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def edit(self, keyframe: Path, instruction: str, out_path: Path) -> Path:
        import requests  # std in our [all] extras; loud ImportError otherwise

        self._headers()                                # loud key check first
        kf = Path(keyframe)
        if not kf.exists() or not kf.is_file():
            raise FileNotFoundError(f"keyframe not found: {keyframe}")
        if kf.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ValueError(
                f"keyframe must be an image (.png/.jpg/.jpeg/.webp), got "
                f"'{kf.suffix}': {keyframe} — a real edit cannot run on a "
                "non-image (mock stubs stay on the mock client)."
            )
        from .video_gen_backends import upload_media

        payload = {
            "enable_base64_output": False,
            "enable_sync_mode": False,
            "images": [upload_media(self.api_key, kf)],
            "prompt": instruction,
            "size": self.size,
        }
        resp = requests.post(f"{self.BASE}/{self.model_id}", json=payload,
                             headers=self._headers(), timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"WaveSpeed image edit submit failed: HTTP {resp.status_code} "
                f"— {resp.text[:2000]}"
            )
        task_id = resp.json()["data"]["id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = requests.get(f"{self.BASE}/predictions/{task_id}/result",
                             headers=self._headers(), timeout=30)
            if r.status_code >= 500:
                time.sleep(self.poll_interval)
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"WaveSpeed image edit poll failed: HTTP {r.status_code} "
                    f"— {r.text[:2000]}"
                )
            data = r.json()["data"]
            status = data.get("status")
            if status == "completed":
                url = data["outputs"][0]
                out_path = Path(out_path)
                # keyframe edits must stay IMAGES (an .txt out_path from an
                # older caller is rewritten to .png so downstream i2v works)
                if out_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    out_path = out_path.with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img = requests.get(url, timeout=120)
                img.raise_for_status()
                out_path.write_bytes(img.content)
                return out_path
            if status == "failed":
                raise RuntimeError(f"WaveSpeed image edit task {task_id} "
                                   f"failed: {data.get('error', 'unknown')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"WaveSpeed image edit task {task_id} did not "
                           f"finish within {self.timeout}s")


def build_image_edit(spec: str | dict | None) -> BaseImageEditClient:
    """None / "mock*" → MockImageEditClient; "wavespeed"/"seedream" →
    WaveSpeedImageEditClient (real, loud without a key at call time)."""
    name = "mock-image-edit"
    config: dict = {}
    if isinstance(spec, dict):
        name = spec.get("name", name)
        config = spec
    elif isinstance(spec, str):
        name = spec
    key = (name or "").lower()
    if key.startswith("mock") or not key:
        return MockImageEditClient(name=name)
    if key in ("wavespeed", "seedream", "seedream-v4"):
        return WaveSpeedImageEditClient(name=name, config=config)
    raise ValueError(
        f"Unknown image_edit backend '{name}'. Known: mock*, wavespeed, seedream"
    )
