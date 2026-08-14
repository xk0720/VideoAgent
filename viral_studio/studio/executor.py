"""执行层 P2 形态: 把 ShotScript 翻译成逐段工具调用计划(dry, 不提交)。

P3 接真实提交时, 每个 plan 项对应一次 API 调用:
  wan2.2-animate-move  → 百炼异步任务(oss上传/轮询协议已在 hook_remake 验证, 届时 copy 进来)
  seedance_t2v         → WaveSpeed bytedance/seedance-2.0/text-to-video
                         (reference_images ≤9, @ImageN; duration int 4-15; 届时 copy 客户端)
装配约定(继承已验证实现): 生成物一律剪回 duration_s; reuse_motion 段带 asset BGM 切片;
连续段 BGM 按"最长连续成功段"整段铺回。
"""
import math
from typing import List

from .memory_store import MemoryStore
from .schemas import ShotScript


def build_tool_plan(script: ShotScript, mem: MemoryStore) -> List[dict]:
    plan = []
    for s in script.segments:
        item = {"seg_id": s.seg_id, "role": s.role, "duration_s": s.duration_s,
                "mode": s.mode, "audio": s.bgm_source}
        if s.mode == "reuse_motion":
            clip = mem.asset_clip_path(s.asset_ref)
            bgm = mem.asset_bgm_path(s.asset_ref)
            item.update({
                "tool": "wan2.2-animate-move",
                "payload": {"ref_image": s.person_hook_refs[0] if s.person_hook_refs else None,
                            "driving_video": str(clip) if clip else None,
                            "mode": "wan-pro"},
                "bgm_slice": str(bgm) if bgm else None,
                "billed_estimate_s": s.duration_s,
            })
        else:  # self_create / self_create_multiwindow / vo → seedance t2v
            # duration 域 = 整数 4-15s(用户裁决: 只能整数); 目标短于4s时生成4s再剪回
            gen_s = min(15, max(4, math.ceil(s.duration_s)))
            item.update({
                "tool": "seedance-2.0/text-to-video",
                "payload": {"prompt": s.prompt,
                            "duration": gen_s,
                            "aspect_ratio": "9:16",
                            "generate_audio": s.mode == "vo",
                            "reference_images": s.person_hook_refs + s.product_image_refs},
                "trim_to_s": s.duration_s,
                "billed_estimate_s": gen_s,
            })
        plan.append(item)
    return plan
