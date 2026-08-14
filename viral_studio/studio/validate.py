"""分镜脚本确定性校验器——导演 agent 的"安检门"。

规则来源全部是实测教训, 不是审美:
  - animate 只吃 compat=pass_verified 的资产(fail_* 两轮实测被拒; untested 需醒目警告)
  - reuse_motion 必须带走资产对应 BGM 切片(用户裁决 2026-08-14)
  - 口播段不配 BGM(用户裁决), 台词必须写进 prompt
  - seedance 时长域 4-15s 整数; 短段按 max(4,ceil) 生成后剪回(帧级 conform 惯例)
  - prompt 纪律: 相机指令第一句; 禁 360/spin/rotate quickly(安全转身准则);
    @ImageN 编号不得超过参考图数量
"""
import math
import re
from pathlib import Path
from typing import List

from .memory_store import MemoryStore
from .schemas import ProductBrief, ShotScript, ValidationReport

FORBIDDEN_MOTION = re.compile(r"\b(360|full turn|spin|rotate quickly)\b", re.I)
CAMERA_WORDS = re.compile(r"camera|static|locked|handheld|push|pan|zoom", re.I)
# 用户裁决: 生成时长/时间戳只能整数秒 → prompt 里不得出现小数秒
FRACTIONAL_SECONDS = re.compile(r"\b\d+\.\d+\s*(?:s\b|s\)|sec|second)", re.I)


def validate_script(script: ShotScript, brief: ProductBrief,
                    mem: MemoryStore) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []

    if len(script.segments) < 2:
        errors.append(f"段落数 {len(script.segments)} 过少——不得丢弃创意方向里的段落, 逐段落实")
    total = sum(s.duration_s for s in script.segments)
    if abs(total - script.total_duration_s) > 0.51:
        errors.append(f"总时长不一致: 段落合计 {total:.1f}s != 声明 {script.total_duration_s:.1f}s")
    dev = abs(total - brief.duration_target_s) / max(brief.duration_target_s, 1e-6)
    if dev > 0.4:
        errors.append(f"总时长 {total:.1f}s 偏离目标 {brief.duration_target_s:.1f}s 超40%——按创意方向补全段落")
    elif dev > 0.15:
        warnings.append(f"总时长 {total:.1f}s 偏离目标 {brief.duration_target_s:.1f}s 超15%")

    seen_ids = set()
    for s in script.segments:
        p = f"[{s.seg_id}]"
        if s.seg_id in seen_ids:
            errors.append(f"{p} seg_id 重复")
        seen_ids.add(s.seg_id)
        if s.duration_s <= 0.3:
            errors.append(f"{p} 时长 {s.duration_s}s 过短(<0.3s)")
        if abs(s.duration_s - round(s.duration_s)) > 1e-6:
            errors.append(f"{p} 时长 {s.duration_s}s 非整数秒(用户裁决: 生成时长只能整数)")

        # ── 引用存在性 ────────────────────────────────────
        for ref in s.person_hook_refs + s.product_image_refs:
            if not Path(ref).exists():
                errors.append(f"{p} 参考图不存在: {ref}")

        # ── 模式规则 ─────────────────────────────────────
        if s.mode == "reuse_motion":
            if s.model != "wan2.2-animate-move":
                errors.append(f"{p} reuse_motion 只能配 animate 模型")
            card = mem.assets.get(s.asset_ref or "")
            if not card:
                errors.append(f"{p} asset_ref '{s.asset_ref}' 不在记忆库")
            else:
                compat = card.get("compat", {}).get("animate_preflight", "untested")
                if compat.startswith("fail"):
                    errors.append(f"{p} 资产 {s.asset_ref} 实测被拒({compat}), 禁止 reuse_motion")
                elif compat != "pass_verified":
                    warnings.append(f"{p} 资产 {s.asset_ref} animate 兼容性未实测, 有被拒风险")
                src_dur = card["source"]["t1"] - card["source"]["t0"]
                if s.duration_s > src_dur + 0.05:
                    errors.append(f"{p} 时长 {s.duration_s}s 超过驱动素材 {src_dur:.2f}s")
                if src_dur < 2.0:
                    errors.append(f"{p} 驱动素材 {src_dur:.2f}s < API 下限 2s")
            if s.bgm_source != "asset_bgm":
                errors.append(f"{p} reuse_motion 必须带走资产 BGM(bgm_source=asset_bgm)")
            if not s.person_hook_refs:
                errors.append(f"{p} reuse_motion 缺人物参考图")
            if not s.prompt.strip():
                warnings.append(f"{p} 建议写一句基本画面描述作台账(animate 不消费该字段)")
            if (s.speech_text or "").strip():
                errors.append(f"{p} reuse_motion 段不写台词(用户裁决 2026-08-14): "
                              f"speech_text 必须为空, 口型跟随驱动素材; 新台词请用 vo 段")

        elif s.mode in ("self_create", "self_create_multiwindow", "vo"):
            if s.model != "seedance_t2v":
                errors.append(f"{p} 自创/口播段当前只支持 seedance_t2v")
            if not s.prompt.strip():
                errors.append(f"{p} 自创段 prompt 为空")
            else:
                if FORBIDDEN_MOTION.search(s.prompt):
                    errors.append(f"{p} prompt 含高危动作词(360/spin/…), 违反安全转身准则")
                if FRACTIONAL_SECONDS.search(s.prompt):
                    errors.append(f"{p} prompt 含小数秒时长/时间戳(只能整数秒)")
                if not CAMERA_WORDS.search(s.prompt.strip()[:90]):
                    warnings.append(f"{p} prompt 开头未见相机指令(家规: 相机先行)")
                n_refs = len(s.person_hook_refs) + len(s.product_image_refs)
                for m in re.finditer(r"@Image(\d+)", s.prompt):
                    if int(m.group(1)) > n_refs:
                        errors.append(f"{p} prompt 引用 @Image{m.group(1)} 超出参考图数量 {n_refs}")
                gen_s = max(4, math.ceil(s.duration_s))
                if gen_s > 15:
                    errors.append(f"{p} 生成时长 {gen_s}s 超 seedance 上限 15s, 需拆段")

        if s.mode == "self_create_multiwindow":
            if not s.window_plan:
                errors.append(f"{p} multiwindow 缺 window_plan")
            else:
                for w in s.window_plan:
                    if (abs(w.t0 - round(w.t0)) > 1e-6
                            or abs(w.t1 - round(w.t1)) > 1e-6):
                        errors.append(f"{p} 窗口 [{w.t0},{w.t1}] 非整数时间戳(只能整数)")
                    if w.t1 - w.t0 < 0.99:
                        errors.append(f"{p} 窗口 [{w.t0},{w.t1}] 短于 1s(整数时间戳下最小窗口=1s)")
                for a, b in zip(s.window_plan, s.window_plan[1:]):
                    if abs(a.t1 - b.t0) > 0.05:
                        errors.append(f"{p} 窗口不连续: {a.t1} → {b.t0}")
                span = s.window_plan[-1].t1 - s.window_plan[0].t0
                if abs(span - s.duration_s) > 0.51:
                    errors.append(f"{p} 窗口总跨度 {span:.1f}s != 段时长 {s.duration_s:.1f}s")

        if s.mode == "vo":
            if not (s.speech_text or "").strip():
                errors.append(f"{p} 口播段缺 speech_text")
            elif s.speech_text not in s.prompt:
                errors.append(f"{p} 台词必须原文写进 prompt(she says: \"...\")")
            if s.bgm_source != "none":
                errors.append(f"{p} 口播段不配 BGM(用户裁决)")

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
