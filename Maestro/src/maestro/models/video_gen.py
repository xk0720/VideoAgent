"""Video generation wrapper. Conditioning = first frame (C2 keyframe anchor)
+ reference images (E1 identity/style). There is NO physics control signal —
the sketch-as-controller line is dead (v0.4); physics is verified from the
generated pixels, never injected.

v0.1 MockVideoGenClient writes a tiny placeholder file per clip and per keyframe
(no GPU). Real: OmniWeaving / Wan / Veo|Sora behind the same `generate` signature.
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
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        ...

    @abstractmethod
    def supported_conditions(self) -> set[str]:
        ...

    def capabilities(self) -> set[str]:
        """Coarse capability set this backend offers ("t2v"|"i2v"|"flf2v"|"edit").

        Default = {"t2v","i2v"} (every backend's `generate` covers those). This
        is the SEED of the Phase-2 capability registry the routing skill will
        read to pick a backend per requested task — extra capabilities
        (flf2v / edit) live on optional methods (frame_to_frame / edit_video)
        declared via hasattr, NOT abstractmethods, so backends that lack them
        (Mock / OmniWeaving / Wan / Veo) are not forced to implement them.
        """
        return {"t2v", "i2v"}


class MockVideoGenClient(BaseVideoGenClient):
    def __init__(self, name: str = "mock-video-gen"):
        self.name = name

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
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = (
            f"MOCK VIDEO\nmodel={self.name}\nprompt={prompt}\nduration={duration}\n"
            f"fps={fps}\nfirst_frame={first_frame}\n"
            f"reference_images={reference_images}\nseed={seed}\n"
        )
        # Write a real (tiny, non-playable) file so downstream path handling works.
        out_path.write_text(meta, encoding="utf-8")
        return out_path

    def supported_conditions(self) -> set[str]:
        return {"first_frame", "reference_images"}


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
