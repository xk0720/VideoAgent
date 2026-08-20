"""Act 编译 —— 分镜脚本(已含完整 pipeline)→ 可执行调用计划。

自从 Planner 直接输出填好的 pipeline 后, 这一层只做两件事: 汇总计费口径、
组装整片时间轴。**不改写 pipeline** —— 需要先生成背景图之类的前置步骤, 一律
在 skill 卡自己的 pipeline 里显式写出来, 保证"所见即所得"。
"""
import copy
import logging
from typing import Dict, List

log = logging.getLogger("viral_studio")

BILLED = {"kling_omni_video": "video_s", "seedance_t2v": "video_s",
          "animate_move": "video_s", "sonilo_text_to_music": "music_s",
          "minimax_tts": "tts_chars", "image_generation": "image_calls"}


class ActCompiler:
    def __init__(self, store, brief: dict, bgm_source=None):
        self.store = store
        self.brief = brief
        self.bgm_source = bgm_source

    def compile_segment(self, seg) -> dict:
        card = self.store.get(seg.skill_id) or {}
        calls: List[dict] = copy.deepcopy(seg.pipeline)   # 原样带过, 不增删
        return {"seg_id": seg.seg_id, "part": seg.part, "skill_id": seg.skill_id,
                "variant": seg.variant, "t0": seg.t0, "t1": seg.t1,
                "duration_s": seg.duration_s,
                "tail_s": float(card.get("tail_s", 0.5))
                if (card.get("produces") or {}).get("variants") else 0.0,
                "calls": calls}

    def compile(self, sb) -> dict:
        segs = [self.compile_segment(s) for s in sb.segments]
        cost = {"video_s": 0.0, "music_s": 0.0, "tts_chars": 0, "image_calls": 0}
        for s in segs:
            for c in s["calls"]:
                kind = BILLED.get(c["tool"])
                if kind == "video_s":
                    cost["video_s"] += float(c["params"].get("duration") or s["duration_s"])
                elif kind == "music_s":
                    cost["music_s"] += float(c["params"].get("duration") or 0)
                elif kind == "tts_chars":
                    cost["tts_chars"] += len(str(c["params"].get("text", "")))
                elif kind == "image_calls":
                    cost["image_calls"] += 1
        return {"product_name": sb.product_name, "person_count": sb.person_count,
                "total_duration_s": round(sum(s["duration_s"] + s["tail_s"] for s in segs), 2),
                "segments": segs, "cost_estimate": cost}
