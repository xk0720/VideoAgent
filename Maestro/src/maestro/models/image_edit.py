"""Keyframe local-edit wrapper (C2). v0.1 mock writes an edited keyframe stub."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


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


def build_image_edit(spec: str | dict | None) -> BaseImageEditClient:
    name = "mock-image-edit"
    if isinstance(spec, dict):
        name = spec.get("name", name)
    elif isinstance(spec, str):
        name = spec
    return MockImageEditClient(name=name)
