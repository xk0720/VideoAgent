"""首帧工厂(ViMax camera_image_generator + pipeline 帧编排的移植)。

三条产线:
1. generate_frame —— 选图 agent 的产物 → 生图。ViMax 的
   generate_single_image(prompt, reference_image_paths) 在本库的等价物是
   seedream-v4/edit 的多图通道(images[0]=主参考,images[1:]=其余参考);
   无参考 → flux t2i。prompt = "Image i: <text>" 前缀 + 选图 agent 的
   text_prompt(ViMax 装配原样)。
2. derive_new_camera —— 子机位首帧派生:让【视频模型】生成一段
   "含一次硬切的双镜视频"(refer=父机位首帧),再找切点取切后首帧。
   ViMax 用 scenedetect;本库用相邻帧像素差(MAD)峰值检测,免新依赖,
   找不到切点 → 末帧兜底(ViMax 同款兜底)。
3. replace_characters —— 派生帧构图/背景对、元素可能错 → seedream 编辑
   "以此图为主参考,把人物换成官方肖像,背景不动"(ViMax 指令原义)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger

log = get_logger("maestro.cinegraph")

# ViMax generate_transition_video 的 prompt 原文
_TWO_SHOT_PROMPT = (
    "Two shots. The transition between the shots is a cut to. The style "
    "of the two shots should be consistent."
    "\nThe first shot description: {first_desc}."
    "\nThe second shot description: {second_desc}.")

# ViMax 喂给选图 agent 的"换人不换景"说明(pipeline 原文原义)
NEW_CAMERA_NOTE = (
    "The composition and background are correct but some elements may be "
    "wrong. The wrong elements should be replaced.\nWrong elements: "
    "{missing_info}.\nYou must select this image as the main reference "
    "and replace the characters in the image with the provided character "
    "portraits. Don't change the background.")


def generate_frame(image_edit, t2i_fn, selector_output: dict,
                   out_path: Path) -> Path:
    """选图产物 → 帧图(幂等)。"""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = selector_output["reference_image_path_and_text_pairs"]
    text = selector_output["text_prompt"]
    prefix = "".join(f"Image {i}: {t}\n" for i, (_, t) in enumerate(pairs))
    prompt = f"{prefix}\n{text}"
    refs = [Path(p) for p, _ in pairs if Path(p).exists()]
    if refs and image_edit is not None:
        # seedream 多图:images[0]=主参考(Image 0),其余随行
        return Path(image_edit.edit(refs[0], prompt, out_path,
                                    references=refs[1:] or None))
    if t2i_fn is None:
        raise RuntimeError("cinegraph: no image generator available")
    return Path(t2i_fn(prompt, out_path))


def derive_new_camera(video_gen, parent_ff: Path, parent_desc: str,
                      child_desc: str, out_dir: Path,
                      tag: str) -> Optional[Path]:
    """双镜切派生(幂等):→ 新机位帧 png;全程失败 → None(调用方
    回落纯选图生成)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_out = out_dir / f"new_camera_{tag}.png"
    if frame_out.exists():
        return frame_out
    video_out = out_dir / f"two_shot_{tag}.mp4"
    try:
        if not video_out.exists():
            prompt = _TWO_SHOT_PROMPT.format(first_desc=parent_desc,
                                             second_desc=child_desc)
            old_audio = getattr(video_gen, "generate_audio", False)
            video_gen.generate_audio = False
            try:
                video_gen.generate(prompt, 5, video_out, fps=24, seed=777,
                                   reference_images=[Path(parent_ff)])
            finally:
                video_gen.generate_audio = old_audio
        cut = _frame_after_cut(video_out, frame_out)
        return cut
    except Exception as exc:
        log.warning("cinegraph: new-camera derivation FAILED for %s (%s) "
                    "— falling back to pure reference generation",
                    tag, str(exc)[:160])
        return None


def _frame_after_cut(video_path: Path, out_png: Path) -> Path:
    """切点检测(scenedetect 的免依赖等价):相邻帧 MAD 峰值 > 阈值
    即切点,取切后一帧;无切点 → 末帧(ViMax 同款兜底)。"""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    prev = None
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"unreadable video: {video_path}")
    best_i, best_mad = None, 0.0
    for i in range(1, len(frames)):
        a = cv2.resize(frames[i - 1], (160, 90)).astype("float32")
        b = cv2.resize(frames[i], (160, 90)).astype("float32")
        mad = float(np.abs(a - b).mean())
        if mad > best_mad:
            best_i, best_mad = i, mad
    # 阈值经验:镜内相邻帧 ≈1-6,真切点 >12(与 _DUP_FRAME_MAD 同源)
    if best_i is not None and best_mad >= 12.0:
        pick = frames[min(best_i + 1, len(frames) - 1)]
        log.info("cinegraph: cut detected at frame %d (MAD %.1f)",
                 best_i, best_mad)
    else:
        pick = frames[-1]
        log.warning("cinegraph: no hard cut found (max MAD %.1f) — "
                    "using the LAST frame as the new camera image",
                    best_mad)
    cv2.imwrite(str(out_png), pick)
    return out_png
