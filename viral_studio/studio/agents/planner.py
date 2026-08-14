"""策划 agent: 商品 brief + 记忆库摘要 → 创意方向。"""
import json
import logging
from pathlib import Path

from ..llm import chat_json
from ..memory_store import MemoryStore
from ..schemas import CreativeDirection, ProductBrief

log = logging.getLogger("viral_studio")
PROMPT = (Path(__file__).parents[1] / "prompts" / "planner.md").read_text(encoding="utf-8")


class Planner:
    def __init__(self, mem: MemoryStore):
        self.mem = mem

    def plan(self, brief: ProductBrief) -> CreativeDirection:
        user = (f"## 商品 brief\n{brief.model_dump_json(indent=2)}\n\n"
                f"{self.mem.digest_for_planner()}\n\n"
                f"请输出 CreativeDirection JSON。")
        raw = chat_json(PROMPT, user, temperature=0.7)
        direction = CreativeDirection.model_validate(raw)
        # 引用兜底: 编造的 id 一律置空并记录(策划的引用只是建议, 硬校验在导演层)
        for seg in direction.structure:
            for attr in ("pattern_ref", "asset_ref"):
                ref = getattr(seg, attr)
                if ref and not self.mem.get(ref):
                    log.warning("策划引用了不存在的 %s='%s', 已置空", attr, ref)
                    setattr(seg, attr, None)
        return direction
