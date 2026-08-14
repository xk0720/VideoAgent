"""记忆库读取与检索。池子小(≈16卡)——全量加载, 检索=标签过滤; 向量检索留到池子变大。"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import MEMORY_DIR

log = logging.getLogger("viral_studio")


class MemoryStore:
    def __init__(self, root: Path = MEMORY_DIR):
        self.root = root
        self.videos: Dict[str, dict] = self._load(root / "videos")
        self.assets: Dict[str, dict] = self._load(root / "assets" / "cards")
        self.patterns: Dict[str, dict] = self._load(root / "patterns")
        log.info("记忆库: %d 视频卡 / %d 资产卡 / %d 策略卡",
                 len(self.videos), len(self.assets), len(self.patterns))

    @staticmethod
    def _load(d: Path) -> Dict[str, dict]:
        out = {}
        for p in sorted(d.glob("*.yaml")):
            card = yaml.safe_load(p.read_text(encoding="utf-8"))
            cid = card.get("asset_id") or card.get("pattern_id") or card.get("video_id")
            out[cid] = card
        return out

    def get(self, ref: str) -> Optional[dict]:
        return self.assets.get(ref) or self.patterns.get(ref) or self.videos.get(ref)

    def asset_clip_path(self, asset_id: str) -> Optional[Path]:
        """卡内 clip/bgm 路径均相对 memory 根目录。"""
        card = self.assets.get(asset_id)
        return self.root / card["source"]["clip"] if card else None

    def asset_bgm_path(self, asset_id: str) -> Optional[Path]:
        card = self.assets.get(asset_id)
        bgm = (card or {}).get("source", {}).get("bgm")
        return self.root / bgm if bgm else None

    # ── 提供给 agent 的上下文摘要 ──────────────────────────
    def digest_for_planner(self) -> str:
        """策划视角: 视频结构谱 + 资产/策略一句话索引(全文太长, 卡id供导演深读)。"""
        lines = ["## 记忆库摘要", "", "### 结构模板(来自已拆解的爆款)"]
        for vid, v in self.videos.items():
            lines.append(f"- {vid}《{v['title']}》 {v['specs']['duration_s']}s/"
                         f"{v['specs'].get('bpm','?')}BPM")
            for seg in v.get("structure", []):
                lines.append(f"    {seg['t']} [{seg['role']}] {seg['desc']}")
        lines.append("")
        lines.append("### 段级资产卡(可引用 asset_ref)")
        for aid, a in self.assets.items():
            compat = a.get("compat", {}).get("animate_preflight", "untested")
            lines.append(f"- {aid} [{a['kind']}|animate:{compat}] "
                         f"{a['content']['summary'][:60]} | 适用: "
                         f"{a.get('usage', {}).get('when_to_use', '')[:60]}")
        lines.append("")
        lines.append("### 策略卡(可引用 pattern_ref)")
        for pid, p in self.patterns.items():
            lines.append(f"- {pid}《{p['name']}》 {p['what_it_does'][:70]} | "
                         f"何时用: {p['when_to_use'][:60]}")
        return "\n".join(lines)

    def cards_for_director(self, refs: List[str]) -> str:
        """导演视角: 被策划引用的卡片全文(YAML 原文, 含 prompt 模板与 compat)。"""
        chunks = []
        for ref in refs:
            card = self.get(ref)
            if card:
                chunks.append(f"===== 卡片 {ref} =====\n"
                              + yaml.safe_dump(card, allow_unicode=True, sort_keys=False))
        return "\n".join(chunks) if chunks else "(无引用卡片)"
