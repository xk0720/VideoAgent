"""Video-generation client ABC + mock + factory.

Open-source backends that real subclasses talk to:
    • OmniWeaving (Tencent-Hunyuan)   https://github.com/Tencent-Hunyuan/OmniWeaving
    • Wan2.6                          https://github.com/Wan-Video
    • Veo (Google) via google-genai   https://ai.google.dev/gemini-api

The conditioning surface is wider than text-only on purpose: hybrid
retrieval+generation relies on passing the *previous* segment's end-frame
and a *character reference image* as anchors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from ...config import configs_dir, load_yaml
from ...utils.video_io import write_silent_color_clip


class BaseVideoGenClient(ABC):
    backend_name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        first_frame: Optional[np.ndarray] = None,
        last_frame: Optional[np.ndarray] = None,
        reference_images: Optional[list[np.ndarray]] = None,
        flow_field: Optional[np.ndarray] = None,
        cinematography_hint: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Path: ...

    @abstractmethod
    def supported_conditions(self) -> set[str]: ...


class MockVideoGenClient(BaseVideoGenClient):
    """Writes a deterministic solid-colour mp4 of the requested duration.

    Useful because ffmpeg-driven assembly still gets a real file it can
    concatenate, so the v0.1 pipeline produces a playable output.
    """

    backend_name = "mock"

    def supported_conditions(self) -> set[str]:
        # Mock claims everything OmniWeaving claims, so EditorAgent's branching
        # matches the production path.
        return {"text", "first_frame", "last_frame", "reference_images", "flow_field"}

    def generate(self, prompt, duration, out_path,
                 first_frame=None, last_frame=None, reference_images=None,
                 flow_field=None, cinematography_hint=None, seed=None) -> Path:
        seed = seed if seed is not None else (abs(hash(prompt)) % (2**16))
        # Derive a colour from the seed so neighbouring generations differ visually.
        r = (seed * 73) % 200 + 40
        g = (seed * 151) % 200 + 40
        b = (seed * 211) % 200 + 40
        return write_silent_color_clip(out_path, duration_s=max(0.5, duration),
                                       fps=24, width=320, height=240, color=(b, g, r))


def build_video_gen_from_config(backend: str, mocks_enabled: bool = True) -> BaseVideoGenClient:
    if mocks_enabled:
        return MockVideoGenClient()
    cfg = load_yaml(configs_dir() / "models" / "video_gen.yaml")
    spec = cfg["aliases"].get(backend)
    if spec is None:
        raise KeyError(f"Video-gen alias {backend!r} not present in video_gen.yaml")
    name = spec["backend"]
    if name == "omniweaving":
        from .omniweaving import OmniWeavingClient
        return OmniWeavingClient()
    if name == "wan_local":
        from .wan_local import WanLocalClient
        return WanLocalClient()
    if name == "api_client":
        from .api_client import ApiVideoGenClient
        return ApiVideoGenClient(provider=spec["provider"], model=spec["model"])
    raise ValueError(f"Unknown video-gen backend: {name}")


__all__ = ["BaseVideoGenClient", "MockVideoGenClient", "build_video_gen_from_config"]
