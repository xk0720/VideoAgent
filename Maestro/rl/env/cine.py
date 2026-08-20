# 2026-08-19 用户令【训练=生产完全同构 + rl/ 自包含】:本文件为
# src/maestro/cinegraph/first_frame_factory.py 中 _spaced_retry 与
# _frame_after_cut 的逐字拷贝(仅日志器改指 rl/env 内部 shim)。
# 改生产原件必须同步改这里(tests/unit/test_rl_env_parity.py 锁差异)。
from __future__ import annotations

import time
from pathlib import Path

from env.logging_utils import get_logger

log = get_logger("maestro.cinegraph")

# 帧是关键路径:图像端点的分钟级网络抖动(2026-08-06 事故:上传三连
# SSL EOF 杀掉整个 run)靠客户端内秒级重试骑不过去 —— 这里再加一层
# 间隔拉开的重试;持续故障仍然如实上抛。
# 梯子加长(2026-08-07 run6:代理阵发断连一波数分钟,90s 窗被打穿):
# 累计覆盖 ~8.5 分钟。
_SPACED_WAITS_S = (0, 30, 60, 120, 300)


def _spaced_retry(fn, tag: str):
    last: Exception | None = None
    for wait in _SPACED_WAITS_S:
        if wait:
            log.warning("cinegraph: %s failed (%s) — retrying in %ds",
                        tag, str(last)[:140], wait)
            time.sleep(wait)
        try:
            return fn()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"cinegraph: {tag} failed after "
                       f"{len(_SPACED_WAITS_S)} spaced attempts: {last}")


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


def frame_review_ok(mllm, llm, frame: Path, want_desc: str) -> bool:
    """帧审查(共享版,2026-08-07):VLM 图注 × LLM 裁决,【只认矛盾,
    不认缺席】(图注是有损摘要,漏项不是罪证 —— run5 教训)。任一端
    缺席/异常 → fail-open 放行并留痕(审查是辅助,不是关卡制造机)。"""
    if mllm is None or llm is None:
        return True
    _cap = getattr(mllm, "caption_image", None)
    if _cap is None:
        return True
    try:
        got = str(_cap(Path(frame)) or "")[:600]
        from env.skills import extract_json as _extract_json
        raw = llm.complete(
            "A frame was generated for an intended description; you "
            "only see a LOSSY caption of it. Reject ONLY if the "
            "caption CONTRADICTS the intent (wrong subject, wrong "
            "setting, wrong style, or a key character clearly absent "
            "or extra). A detail the caption merely does not mention "
            "is NOT a defect.\nINTENDED: " + str(want_desc)[:600]
            + "\nGENERATED (caption): " + got
            + '\nSTRICT JSON only: {"ok": true|false, '
              '"reason": "<one sentence>"}')
        d = _extract_json(raw) or {}
        if not bool(d.get("ok", True)):
            log.warning("frame review REJECTED — %s", d.get("reason"))
            return False
    except Exception as exc:
        log.warning("frame review errored (%s) — fail-open, frame "
                    "accepted", str(exc)[:160])
        return True
    return True


def _probe_fps(path: Path) -> float:
    """容器申报的 fps;拿不到 → 0.0(与生产 timeline._probe_fps 同构,
    decord 优先、cv2 兜底)。"""
    p = Path(path)
    try:
        import decord  # type: ignore

        fps = float(decord.VideoReader(str(p)).get_avg_fps())
        if fps > 0:
            return fps
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def extract_frame(video_path: Path, idx: int, out_path: Path):
    """第 idx 帧(越界取最近合法帧)写 PNG;解不出 → None(与生产
    timeline.extract_frame 同构,解码栈用 cv2)。"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    i = max(0, min(int(idx), len(frames) - 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frames[i]):
        return None
    return out_path
