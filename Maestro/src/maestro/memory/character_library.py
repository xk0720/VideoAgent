"""跨片角色库(2026-07-31 用户裁决"现在做"):同一个角色的官方肖像
跨影片复用 —— 这次片里的"橘白猫"和上次片里的是同一只。

设计(极简版,与 ViMax 肖像注册表同源、与我们的记忆哲学同构):
  · 库 = 一个目录 + index.json([{name, descriptor, file}],原子写);
  · 命中 = 名字相同(大小写不敏感)且描述符词重叠 ≥ 0.6 —— 名字相同
    但长相不同(另一部片里另一只猫也叫 the cat)绝不误配;
  · 入库 = 拷贝肖像文件进库目录;查询命中记 uses(使用台账)。
诚实链:库目录不可写/文件丢失 → 静默视为未命中,绝不阻塞正流程。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger

log = get_logger("charlib")


def _words(text: str) -> list[str]:
    import re
    return [w.lower() for w in re.findall(r"[a-zA-Z一-鿿0-9]+", text or "")]


def _overlap(a: str, b: str) -> float:
    wa, wb = set(_words(a)), set(_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


class CharacterLibrary:
    """目录级角色肖像库;index.json 是唯一台账。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index: list[dict] = []
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            idx = self.root / "index.json"
            if idx.exists():
                self.index = json.loads(idx.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("character library unavailable (%s) — running "
                        "without cross-film reuse", exc)

    def _save(self) -> None:
        try:
            tmp = self.root / "index.json.tmp"
            tmp.write_text(json.dumps(self.index, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
            tmp.replace(self.root / "index.json")
        except Exception as exc:
            log.warning("character library save failed: %s", exc)

    def lookup(self, name: str, descriptor: str,
               threshold: float = 0.6) -> Optional[Path]:
        """名字相同且描述符词重叠 ≥ threshold → 肖像路径;否则 None。"""
        for row in self.index:
            if row.get("name", "").strip().lower() != name.strip().lower():
                continue
            if _overlap(descriptor, row.get("descriptor", "")) >= threshold:
                p = self.root / row.get("file", "")
                if p.exists():
                    row["uses"] = int(row.get("uses", 0)) + 1
                    self._save()
                    log.info("character library HIT: %s ← %s", name, p.name)
                    return p
                log.warning("character library: index row for %s points to "
                            "a missing file — ignored", name)
        return None

    def add(self, name: str, descriptor: str, image_path: Path) -> Optional[Path]:
        src = Path(image_path)
        if not src.exists():
            return None
        slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:40]
        dst = self.root / f"{slug}_{int(time.time())}{src.suffix}"
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            log.warning("character library add failed: %s", exc)
            return None
        self.index.append({"name": name, "descriptor": descriptor,
                           "file": dst.name, "uses": 0})
        self._save()
        log.info("character library ADD: %s → %s", name, dst.name)
        return dst
