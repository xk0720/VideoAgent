"""引用槽位校验器(方案 A,2026-07-16 裁决)。

问题:@ImageN/@VideoN/"reference image N" 的编号由执行器装配 payload 时
决定;brain 若在 prompt 里自己引用素材,编号靠 skill 教它猜 —— 猜错无人
拦截。方案 A 把编号从"要遵守的规则"变成"发给它的数据":执行器先算出
【槽位清单】(pipeline.window_loop._slot_manifest,与装配同源),写 prompt
的人只许用清单里的 ID;本模块是出口闸 —— 确定性校验,不靠 LLM 自觉。

闸门规则:
- 引用了清单外的编号 → 整条 prompt 作废(返回 ""),调用方落内容感知
  兜底模板(enhancer 路径先重试一次)。错编号永远到不了 API。
- 可引用槽位漏提 → 自动补一句(素材白上传 = 浪费钱,补齐是安全操作)。
- FIRST_FRAME/LAST_FRAME 这类不可 @ 引用的槽位不参与校验(i2v/flf2v
  路线没有引用通道,prompt 只描述运动)。
"""
from __future__ import annotations

import re

# @Image3 / @Video1 / "reference image 2" / <<<image_1>>>(百炼可灵方言,
# 2026-08-03)—— 大小写不敏感;闸门本身按槽位 ID 比对,与方言无关。
_REF_TOKEN_RE = re.compile(
    r"@(?:Image|Video)\d+|reference image \d+|<<<(?:image|video)_\d+>>>",
    re.IGNORECASE)


# 可灵记号变体归一(2026-08-04 实跑:brain 写了 <<image_1>>(两层尖括号),
# 闸门既没拦也没算"已提及" → 自动补句追加了第二套引用,prompt 爆长、
# 绑定被稀释)。1~4 层尖括号/大小写/空格下划线变体一律归一为 <<<image_N>>>。
_KLING_VARIANT_RE = re.compile(
    r"<{1,4}\s*(image|video)[\s_]?(\d+)\s*>{1,4}", re.IGNORECASE)


def normalize_ref_tokens(prompt: str) -> str:
    """写手笔误的引用记号确定性归一(只动记号,不动其他文字)。"""
    return _KLING_VARIANT_RE.sub(
        lambda m: f"<<<{m.group(1).lower()}_{m.group(2)}>>>",
        str(prompt or ""))


def validate_references(prompt: str, slots: list[dict]) -> tuple[str, dict]:
    """prompt 的引用 ⊆ 清单?→ (修正后的 prompt, 审计记录)。

    返回 ("", {...ok: False...}) = 引用了不存在的槽位,救不了 —— 调用方
    必须弃用这条 prompt(落兜底/重试),绝不带错编号出门。
    """
    prompt = normalize_ref_tokens(prompt)
    allowed = {str(r.get("slot", "")).lower(): r for r in slots
               if r.get("referenceable")}
    found = {m.group(0) for m in _REF_TOKEN_RE.finditer(prompt)}
    unknown = sorted(t for t in found if t.lower() not in allowed)
    if unknown:
        return "", {"ok": False, "unknown": unknown,
                    "allowed": sorted(r["slot"] for r in slots
                                      if r.get("referenceable"))}
    mentioned = {t.lower() for t in found}
    appended = []
    fixed = prompt.rstrip()
    for r in slots:
        if not r.get("referenceable"):
            continue
        # 执行器专有行豁免(2026-08-06 rainnight run4:pin_frame 行的
        # 说明文本被当内容补挂进出门 prompt)—— 它的提及由执行器机器
        # 句负责,补挂闸绝不代劳。
        if r.get("source") == "pin_frame" \
                or "executor owns its mention" in str(r.get("content", "")):
            continue
        slot = str(r.get("slot", ""))
        if slot.lower() in mentioned:
            continue
        content = str(r.get("content", "")).strip()
        if slot.lower().startswith("@video"):
            fixed += (f" {slot} is {content} — continue its motion "
                      "seamlessly." if content
                      else f" Continue {slot}'s motion seamlessly.")
        else:
            fixed += (f" {slot} shows: {content} — keep it consistent."
                      if content else f" Keep {slot} consistent.")
        appended.append(slot)
    return fixed, {"ok": True, "unknown": [], "appended": appended}
