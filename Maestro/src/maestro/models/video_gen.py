"""Video generation wrapper. MUST accept conditioning (control signal / first
frame / reference images) or the physics sketch layer is meaningless.

v0.1 MockVideoGenClient writes a tiny placeholder file per clip and per keyframe
(no GPU). v0.2: OmniWeaving / Wan / Veo|Sora behind the same `generate` signature.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseVideoGenClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        control_signal: Optional[Path] = None,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        ...

    @abstractmethod
    def supported_conditions(self) -> set[str]:
        ...


class MockVideoGenClient(BaseVideoGenClient):
    def __init__(self, name: str = "mock-video-gen"):
        self.name = name

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        control_signal: Optional[Path] = None,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = (
            f"MOCK VIDEO\nmodel={self.name}\nprompt={prompt}\nduration={duration}\n"
            f"fps={fps}\ncontrol_signal={control_signal}\nfirst_frame={first_frame}\n"
            f"reference_images={reference_images}\nseed={seed}\n"
        )
        # Write a real (tiny, non-playable) file so downstream path handling works.
        out_path.write_text(meta, encoding="utf-8")
        return out_path

    def supported_conditions(self) -> set[str]:
        return {"control_signal", "first_frame", "reference_images"}


def build_video_gen(spec: str | dict | None) -> BaseVideoGenClient:
    name = "mock-video-gen"
    config: dict = {}
    if isinstance(spec, dict):
        name = spec.get("name", name)
        config = spec
    elif isinstance(spec, str):
        name = spec
    if name.startswith("mock"):
        return MockVideoGenClient(name=name)
    # Real backend (OmniWeaving / Wan / Veo|Sora). Lazy import so v0.1 stays light.
    from .video_gen_backends import build_real_video_gen
    return build_real_video_gen(name, config)
