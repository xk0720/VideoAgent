"""导演 agent: 创意方向 → 可执行分镜脚本。

工程决策(实测教训): 大 JSON 一把梭输出不稳定 → **逐段导演**——每段一次小调用;
段落数与各段时长由策划层锁定, 导演只做执行决策(mode/refs/prompt), 程序负责
seg_id/时长/总长与 @ImageN 兜底; 校验不过只重导出错的段。
"""
import logging
import re
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from ..llm import chat_json
from ..memory_store import MemoryStore
from ..schemas import (CreativeDirection, ProductBrief, SegmentBrief,
                       SegmentPlan, ShotScript, ValidationReport)
from ..validate import validate_script

log = logging.getLogger("viral_studio")
PROMPT = (Path(__file__).parents[1] / "prompts" / "director.md").read_text(encoding="utf-8")
MAX_REPAIR_ROUNDS = 2
SEG_ERR = re.compile(r"^\[(seg\d+)\]")


class Director:
    def __init__(self, mem: MemoryStore):
        self.mem = mem

    def direct(self, brief: ProductBrief,
               direction: CreativeDirection) -> tuple[ShotScript, ValidationReport]:
        segs: List[SegmentPlan] = []
        for i, sb in enumerate(direction.structure, 1):
            segs.append(self._one_segment(brief, sb, i))
        script = self._assemble(brief, segs)
        report = validate_script(script, brief, self.mem)

        for round_i in range(1, MAX_REPAIR_ROUNDS + 1):
            if report.ok:
                break
            per_seg = {}
            for err in report.errors:
                m = SEG_ERR.match(err)
                if m:
                    per_seg.setdefault(m.group(1), []).append(err)
            if not per_seg:            # 只剩全局性错误——逐段重导救不了, 交人工
                break
            log.info("修复轮 %d: 重导 %d 个段落", round_i, len(per_seg))
            for seg_id, errs in per_seg.items():
                idx = next(i for i, s in enumerate(segs) if s.seg_id == seg_id)
                segs[idx] = self._one_segment(brief, direction.structure[idx],
                                              idx + 1, prev=segs[idx], errors=errs)
            script = self._assemble(brief, segs)
            report = validate_script(script, brief, self.mem)

        if not report.ok:
            log.error("仍有 %d 项阻断, 保留供人工审阅", len(report.errors))
        return script, report

    # ── 单段导演 ─────────────────────────────────────────
    def _one_segment(self, brief: ProductBrief, sb: SegmentBrief, idx: int,
                     prev: Optional[SegmentPlan] = None,
                     errors: Optional[List[str]] = None) -> SegmentPlan:
        refs = [r for r in (sb.pattern_ref, sb.asset_ref) if r]
        user = (f"## 商品 brief\n{brief.model_dump_json(indent=2)}\n\n"
                f"## 本段创意(策划已定, 不得改变段落职责与时长)\n"
                f"{sb.model_dump_json(indent=2)}\n\n"
                f"## 被引用卡片全文\n{self.mem.cards_for_director(refs)}\n\n"
                f"本段 seg_id 固定为 \"seg{idx:02d}\", duration_s 固定为 {sb.duration_s}。\n"
                f"请只输出这一个段落的 SegmentPlan JSON 对象。")
        if prev is not None and errors:
            user += ("\n\n## 你上一版本段\n" + prev.model_dump_json(indent=2)
                     + "\n\n## 必须修复的问题\n- " + "\n- ".join(errors)
                     + "\n\n输出修复后的 SegmentPlan JSON。")

        last: List[str] = []
        for attempt in range(3):
            raw = chat_json(PROMPT, user if not last else
                            user + "\n\n## 上次输出的结构错误\n- " + "\n- ".join(last),
                            temperature=0.3)
            try:
                seg = SegmentPlan.model_validate(raw)
            except ValidationError as e:
                last = [f"{err['loc']} — {err['msg']}" for err in e.errors()[:8]]
                log.info("seg%02d 第 %d 次结构错误 ×%d", idx, attempt + 1, len(last))
                continue
            seg.seg_id = f"seg{idx:02d}"          # 程序锁定, 不信任模型
            seg.role = sb.role
            # 用户裁决(2026-08-14): 生成时长与时间戳只能整数秒 → 锁定时就取整
            seg.duration_s = float(max(1, round(sb.duration_s)))
            return seg
        raise RuntimeError(f"seg{idx:02d} 连续 3 次无法产出合法结构: {last[:3]}")

    # ── 组装与确定性兜底 ─────────────────────────────────
    def _assemble(self, brief: ProductBrief, segs: List[SegmentPlan]) -> ShotScript:
        script = ShotScript(
            product_name=brief.name,
            total_duration_s=round(sum(s.duration_s for s in segs), 2),
            segments=segs)
        for s in script.segments:      # @ImageN 超编 → 先补挂未用人物图, 池尽则钳制编号
            mentions = [int(m.group(1)) for m in re.finditer(r"@Image(\d+)", s.prompt)]
            need = max(mentions) if mentions else 0
            pool = [p for p in brief.person_hooks
                    if p not in s.person_hook_refs and p not in s.product_image_refs]
            while len(s.person_hook_refs) + len(s.product_image_refs) < need and pool:
                s.person_hook_refs.append(pool.pop(0))
            avail = len(s.person_hook_refs) + len(s.product_image_refs)
            if need > avail and avail > 0:
                s.prompt = re.sub(r"@Image(\d+)",
                                  lambda m: f"@Image{min(int(m.group(1)), avail)}",
                                  s.prompt)
                log.info("%s: @Image 编号钳制到 ≤%d(参考图池耗尽)", s.seg_id, avail)
        return script
