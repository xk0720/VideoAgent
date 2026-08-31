"""生成前全量预检(确定性, 零生成费): 把"跑了才知道"的失败尽量搬到跑之前。

对象 = 一个 run 目录里的 storyboard.json + call_plan.json。
每条检查都对应一次实测踩过的坑; 新坑修完就在这里加一条, 让它只坑一次。
"""
import json
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

from .render import PROJECT_ROOT

KLING_REF = re.compile(r"<<<image_(\d+)>>>")
OLD_DIALECT = re.compile(r"@Image\d")
CJK = re.compile(r"[一-鿿]")
SPEECH_CPS = 5.0            # 中文语速实测 ≈5 字/秒


def _probe(path: str) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip() or 0)
    except Exception:                                   # noqa: BLE001
        return 0.0


def _exists(v: str) -> bool:
    p = Path(v)
    return (p if p.is_absolute() else PROJECT_ROOT / v).exists()


def run_pretest(out_dir: Path) -> Tuple[List[str], List[str]]:
    """返回 (errors, warnings)。errors 非空即不应进入付费执行。"""
    errs: List[str] = []
    warns: List[str] = []
    plan = json.loads((out_dir / "call_plan.json").read_text(encoding="utf-8"))

    prev_t1 = 0.0
    for seg in plan["segments"]:
        sid = seg["seg_id"]
        # 时间轴连续
        t0, t1 = float(seg.get("t0") or 0), float(seg.get("t1") or 0)
        if abs(t0 - prev_t1) > 0.01:
            errs.append(f"[{sid}] 时间轴断裂: t0={t0} 上段止于 {prev_t1}")
        prev_t1 = t1
        durs_by_id = {}
        for c in seg["calls"]:
            cid, tool, prm = c["id"], c["tool"], c["params"]
            tag = f"[{sid}.{cid}]"

            # 通用: 文本值不残留占位符 / 路径参数存在
            for k, v in prm.items():
                if isinstance(v, str):
                    if "{" in v and "}" in v:
                        errs.append(f"{tag} {k} 残留占位符")
                    if v.startswith("$"):
                        errs.append(f"{tag} {k} 未解析: {v[:40]}")
                    looks_path = ("/" in v and not v.startswith("@")
                                  and re.search(r"\.(png|jpe?g|mp4|m4a|mp3|wav)$", v))
                    if looks_path and not _exists(v):
                        errs.append(f"{tag} {k} 文件不存在: {v}")

            if tool == "kling_omni_video":
                pr = str(prm.get("prompt", ""))
                refer = prm.get("refer") or []
                d = prm.get("duration")
                if not isinstance(d, int) or not (3 <= d <= 15):
                    errs.append(f"{tag} duration={d} 超出可灵整数域 [3,15]")
                if OLD_DIALECT.search(pr):
                    errs.append(f"{tag} prompt 用了旧方言 @ImageN, 应为 <<<image_N>>>")
                for m in KLING_REF.finditer(pr):
                    if int(m.group(1)) > len(refer):
                        errs.append(f"{tag} 引用 <<<image_{m.group(1)}>>> 超出 refer 数 {len(refer)}")
                if refer and not KLING_REF.search(pr):
                    warns.append(f"{tag} 挂了参考图但 prompt 没有 <<<image_N>>> 身份锚")
                if not refer and not prm.get("first_frame") and not prm.get("aspect_ratio"):
                    errs.append(f"{tag} 纯文生视频缺 aspect_ratio(可灵硬性要求)")
                # 口播语速: says: "…" 的中文字数要塞得进镜头窗
                for shot in re.finditer(
                        r"Shot \d \((\d+)-(\d+)s\).*?says: \"([^\"]+)\"", pr, re.S):
                    a, b, line = int(shot.group(1)), int(shot.group(2)), shot.group(3)
                    need_s = len(CJK.findall(line)) / SPEECH_CPS
                    if need_s > (b - a) + 1.0:
                        errs.append(f"{tag} 台词「{line[:12]}…」需 {need_s:.1f}s "
                                    f"说完, 超出镜头窗 {b-a}s")

            elif tool == "animate_move":
                drv = str(prm.get("driving", ""))
                if _exists(drv):
                    dd = _probe(drv if Path(drv).is_absolute()
                                else str(PROJECT_ROOT / drv))
                    if dd < 2.0:
                        errs.append(f"{tag} 驱动 {Path(drv).name} 仅 {dd:.2f}s, "
                                    f"低于 2s API 地板(需回文补帧)")

            elif tool == "minimax_tts":
                txt = str(prm.get("text", ""))
                if not CJK.search(txt):
                    errs.append(f"{tag} TTS 文本非中文: {txt[:20]}")

            elif tool == "assemble_slots":
                nv = len(prm.get("videos") or [])
                nd = len(prm.get("durations") or [])
                if nv != nd:
                    errs.append(f"{tag} 槽位 {nv} 个但时长表 {nd} 项")

            elif tool in ("mix_audio", "concat_audio"):
                if prm.get("duration"):
                    durs_by_id[cid] = float(prm["duration"])

            if tool == "sonilo_text_to_music":
                md = float(prm.get("duration") or 0)
                seg_len = t1 - t0
                if md < seg_len:
                    warns.append(f"{tag} 音乐 {md:.0f}s 短于段长 {seg_len:.1f}s")

        # 段内 mix 时长与段长一致性(±1s 容差, 收尾卡有 tail 余量)
        for cid, d in durs_by_id.items():
            if d > (t1 - t0) + 1.01:
                warns.append(f"[{sid}.{cid}] duration={d} 超过段长 {t1-t0:.1f}s 逾 1s")

    return errs, warns
