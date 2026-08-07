"""参考图选择 agent(ViMax reference_image_selector 文本级严格移植)。

从「多视图肖像 + 帧历史序列」中为目标帧挑 ≤8 张参考,并产出
"哪个元素参考哪张图" 的生图 prompt。三大一致性目标(角色/环境/风格)
与全部选取规则逐字采用;ViMax 的两级(文本→多模态)选择,v1 移植
文本级(其自身就是 ViMax 的第一级;多模态级待 mllm 多图接口通)。
"""
from __future__ import annotations

import re

from ..logging_utils import brain_log, get_logger
from ..pipeline.window_loop import _extract_json

log = get_logger("maestro.cinegraph")


def _normalize_image_refs(text: str, n: int) -> str:
    """1 基笔误归一(2026-08-07 run6:模型写 "Image 1 为主场景…Image 2
    中的老大",契约是 0 基,而装配前缀只定义到 Image n-1)——文本引用
    全体 ∈[1..n] 且不含 Image 0 时判定 1 基,统一减一(升序替换防连锁)。"""
    ks = sorted({int(m) for m in re.findall(r"Image\s*(\d+)", text or "")})
    if ks and ks[0] >= 1 and ks[-1] == n:
        log.warning("cinegraph: selector text used 1-based Image refs %s "
                    "(contract is 0-based) — renumbering", ks)
        for k in ks:
            text = re.sub(rf"Image\s*{k}\b", f"Image {k - 1}", text)
    return text

# ── ViMax system prompt 原文移植(输出契约换 STRICT JSON)──
_SELECT_SYSTEM = """
[Role]
You are a professional visual creation assistant skilled in multimodal image analysis and reasoning.

[Task]
Your core task is to intelligently select the most suitable reference images from a provided set of reference image descriptions (including multiple character reference images and existing scene images from prior frames) based on the user's text description (describing the target frame), ensuring that the subsequently generated image meets the following key consistencies:
- Character Consistency: The appearance (e.g. gender, ethnicity, age, facial features, hairstyle, body shape), clothing, expression, posture, etc., of the generated character should highly match the reference image descriptions.
- Environmental Consistency: The scene of the generated image (e.g., background, lighting, atmosphere, layout) should remain coherent with the existing image descriptions from prior frames.
- Style Consistency: The visual style of the generated image (e.g., realistic, cartoon, film-like, color tone) should harmonize with the reference image descriptions.

[Input]
You will receive a text description of the target frame, along with a sequence of reference image descriptions.
- The text description of the target frame is enclosed within <FRAME_DESC> and </FRAME_DESC>.
- The sequence of reference image descriptions is enclosed within <SEQ_DESC> and </SEQ_DESC>. Each description is prefixed with its index, starting from 0.

[Output]
STRICT JSON only:
{"ref_image_indices": [<0-based ints>], "text_prompt": "<image-generation prompt: describe the image to be created, specifying which elements should reference which image. The index refers to the position in ref_image_indices, not the original sequence number. Refer to reference images strictly as 'Image N'.>"}

[Guidelines]
- Ensure that the language of all output values (not include keys) matches that used in the frame description.
- The reference image descriptions may depict the same character from different angles, in different outfits, or in different scenes. Identify the description closest to the version described by the user
- Prioritize image descriptions with similar compositions, i.e., shots taken by the same camera.
- The images from prior frames are arranged in chronological order. Give higher priority to more recent images (those closer to the end of the sequence).
- Choose reference image descriptions that are as concise as possible and avoid including duplicate information.
- When a new character appears in the frame description, prioritize selecting their portrait image description (if available) to ensure accurate depiction of their appearance. Pay attention to whether the character is facing the camera from the front, side, or back. Choose the most suitable view as the reference image for the character.
- For character portraits, you can only select at most one image from multiple views (front, side, back). Choose the most appropriate one based on the frame description.
- Select at most **8** optimal reference image descriptions.
- The text guiding image editing should be as concise as possible.
"""


def select_pairs_by_indices(pairs, indices):
    """ViMax 原样移植:LLM 给的下标越界即拒(负数会经 Python 索引
    静默选错图)。"""
    invalid = [i for i in indices
               if not isinstance(i, int) or i < 0 or i >= len(pairs)]
    if invalid:
        raise ValueError(f"ref_image_indices out of range: {invalid} "
                         f"(have {len(pairs)} images)")
    return [pairs[i] for i in indices]


def select_reference_images(llm, available_pairs: list,
                            frame_description: str) -> dict:
    """→ {"reference_image_path_and_text_pairs": [...], "text_prompt": str}
    坏 JSON/越界下标重试一次;仍失败 → 诚实降级:全量参考(≤8,近期
    优先)+ 帧描述直出。"""
    seq = "\n".join(f"Image {i}: {t}"
                    for i, (_, t) in enumerate(available_pairs))
    prompt = (_SELECT_SYSTEM
              + f"\n\n<FRAME_DESC>\n{frame_description}\n</FRAME_DESC>\n"
              + f"\n<SEQ_DESC>\n{seq}\n</SEQ_DESC>")
    for _attempt in range(2):
        raw = ""
        err = None
        try:
            raw = llm.complete(prompt)
            data = _extract_json(raw)
            idxs = list((data or {}).get("ref_image_indices") or [])
            text = str((data or {}).get("text_prompt") or "").strip()
            pairs = select_pairs_by_indices(available_pairs, idxs)
            if not text:
                raise ValueError("empty text_prompt")
        except Exception as exc:
            err = str(exc)[:200]
            pairs, text = None, None
        brain_log("cinegraph/reference_select", {
            "raw": raw, "usable": err is None, "error": err,
            "parsed": ({"ref_image_indices": idxs, "text_prompt": text}
                       if err is None else None),
            "context": {"frame_description": frame_description,
                        "n_available": len(available_pairs)}})
        if err is None:
            return {"reference_image_path_and_text_pairs": pairs,
                    "text_prompt": _normalize_image_refs(text, len(pairs))}
        log.warning("cinegraph: reference selection unusable (%s) — %s",
                    err, "retrying" if _attempt == 0 else "degrading")
        prompt += (f"\n\nYOUR PREVIOUS REPLY WAS INVALID: {err}. "
                   "Output the corrected STRICT JSON.")
    tail = available_pairs[-8:]
    return {"reference_image_path_and_text_pairs": tail,
            "text_prompt": ("Create an image following the given "
                            f"description: {frame_description}")}
