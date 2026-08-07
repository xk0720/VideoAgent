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
            # \b 对中文失效(汉字属 \w,"2中"无边界)→ 数字前瞻断字
            text = re.sub(rf"Image\s*{k}(?!\d)", f"Image {k - 1}", text)
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

Below is an example of the input format:
<FRAME_DESC>
[Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob's expression shifts from surprise to delight as he recognizes Alice.
</FRAME_DESC>

<SEQ_DESC>
Image 0: A front-view portrait of Alice.
Image 1: A front-view portrait of Bob.
Image 2: [Camera 0] Medium shot of the supermarket aisle. Alice and Bob are shown in profile facing the right side of the frame. Bob is on the right side of the frame, and Alice is on the left side. Alice, looking down and pushing a shopping cart, follows closely behind Bob and accidentally bumps into his heel.
Image 3: [Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob quickly turns around, and his expression shifts from neutral to surprised.
Image 4: [Camera 2] Shot from Bob's over-the-shoulder perspective. Bob is on the side closer to the camera, with only his shoulder appearing in the lower right corner of the frame. Alice is on the side farther from the camera, positioned slightly left of center in the frame. Alice looks down, then up as she prepares to apologize. Upon realizing it's someone familiar, her expression shifts to one of surprise.
</SEQ_DESC>

[Output]
You need to select up to 8 of the most relevant reference images based on the user's description and put the corresponding indices in the ref_image_indices field of the output. At the same time, you should generate a text prompt that describes the image to be created, specifying which elements in the generated image should reference which image description (and which elements within it).

STRICT JSON only:
{"ref_image_indices": [<ints>], "text_prompt": "<str>"}
- ref_image_indices: Indices of reference images selected from the provided images. For example, [0, 2, 5] means selecting the first, third, and sixth images. The indices should be 0-based.
- text_prompt: Text description to guide the image generation. You need to describe the image to be generated, specifying which elements in the generated image should reference which image (and which elements within it). For example, 'Create an image following the given description: \nThe man is standing in the landscape. The man should reference Image 0. The landscape should reference Image 1.' Here, the index of the reference image should refer to its position in the ref_image_indices list, not the sequence number in the provided image list. Refer to the reference image must be in the format of Image N. Do not use any other word except Image.

[Guidelines]
- Ensure that the language of all output values (not include keys) matches that used in the frame description.
- The reference image descriptions may depict the same character from different angles, in different outfits, or in different scenes. Identify the description closest to the version described by the user
- Prioritize image descriptions with similar compositions, i.e., shots taken by the same camera.
- The images from prior frames are arranged in chronological order. Give higher priority to more recent images (those closer to the end of the sequence).
- Choose reference image descriptions that are as concise as possible and avoid including duplicate information. For example, if Image 3 depicts the facial features of Bob from the front, and Image 1 also depicts Bob's facial features from the front-view portrait, then Image 1 is redundant and should not be selected.
- When a new character appears in the frame description, prioritize selecting their portrait image description (if available) to ensure accurate depiction of their appearance. Pay attention to whether the character is facing the camera from the front, side, or back. Choose the most suitable view as the reference image for the character.
- For character portraits, you can only select at most one image from multiple views (front, side, back). Choose the most appropriate one based on the frame description. For example, when depicting a character from the side, choose the side view of the character.
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
