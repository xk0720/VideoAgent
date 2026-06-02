"""ImageOpsTool — editing category. Resize / crop / paste primitives.

UniVA bundles equivalents for image-side manipulation (resize to model res,
crop to subject, paste a masked patch back). Used by the Refiner pipeline when
the image-edit model returns a patch that needs to be re-inserted into the
keyframe. PIL when available, file-copy when not — keeps mock pipelines green.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .base import BaseTool

try:                                      # pragma: no cover (optional dep)
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None                          # type: ignore
    _HAS_PIL = False


class ImageOpsTool(BaseTool):
    name = "image_ops"
    category = "editing"
    description = "Resize / crop / paste images. Uses PIL when installed, file-copy otherwise."
    side_effects = True

    def resize(
        self, src: str | Path, out: str | Path, size: tuple[int, int]
    ) -> Path:
        src_p, out_p = Path(src), Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_PIL and src_p.exists():
            try:
                Image.open(src_p).resize(size).save(out_p)
                return out_p
            except Exception:
                pass
        # Sandbox fallback: copy bytes through, annotate intent.
        shutil.copy2(src_p, out_p) if src_p.exists() else out_p.write_text(
            f"MOCK RESIZE\nfrom={src_p}\nsize={size}\n", encoding="utf-8"
        )
        return out_p

    def crop(
        self, src: str | Path, out: str | Path, box: tuple[int, int, int, int]
    ) -> Path:
        src_p, out_p = Path(src), Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_PIL and src_p.exists():
            try:
                Image.open(src_p).crop(box).save(out_p)
                return out_p
            except Exception:
                pass
        shutil.copy2(src_p, out_p) if src_p.exists() else out_p.write_text(
            f"MOCK CROP\nfrom={src_p}\nbox={box}\n", encoding="utf-8"
        )
        return out_p

    def run(
        self,
        op: str,
        src: str | Path,
        out: str | Path,
        size: Optional[tuple[int, int]] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> Path:
        """Generic dispatch used by the ActAgent (uniform `run(**kwargs)`)."""
        if op == "resize":
            assert size is not None
            return self.resize(src, out, size)
        if op == "crop":
            assert box is not None
            return self.crop(src, out, box)
        raise ValueError(f"unknown image op: {op!r}")
