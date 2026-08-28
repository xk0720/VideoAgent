"""Executor —— 执行调用计划, 产出成片。最后一公里。

输入 call_plan.json, 逐段按顺序执行每个调用:
  · 远程调用 → 对应后端客户端(kling / animate / TTS / 音乐 / 图像)
  · 本地调用 → local_tools 里的 ffmpeg / demucs / Pillow 实现
  · 参数里的 "@id" 在执行前替换成该段前序调用的实际产物路径
每段最后一个调用的产物即该段成品, 全部段落按时间轴顺序拼接成整片。

失败处理沿用实测结论:
  · 前置检测类拒绝(NoHuman/FullFace)确定性, 不重试, 直接标记该段失败
  · 网络/超时类失败重试一次
  · 某段失败不影响其他段; 成片由成功的段落拼接而成, 缺段如实记入台账
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .local_tools import REGISTRY as LOCAL, concat_av, probe_duration, run

log = logging.getLogger("viral_studio")

DETERMINISTIC = ("InvalidVideo.", "InvalidParameter", "censor", "DataInspection")


class Executor:
    def __init__(self, out_dir: Path, dashscope_key: str = "",
                 wavespeed_key: str = "", dry_run: bool = False,
                 kling_mode: str = "std"):
        self.out = out_dir
        self.dry = dry_run
        self.kling_mode = kling_mode
        self._ds, self._ws = dashscope_key, wavespeed_key
        self._clients: Dict[str, Any] = {}
        (out_dir / "gen").mkdir(parents=True, exist_ok=True)
        (out_dir / "work").mkdir(exist_ok=True)

    # ── 后端懒加载: 计划里没用到的路线不因缺 key 而报错 ──
    def _client(self, name: str):
        if name in self._clients:
            return self._clients[name]
        if name == "kling":
            from .backends.bailian_kling import BailianKlingClient
            c = BailianKlingClient(api_key=self._ds, mode=self.kling_mode)
        elif name == "animate":
            from .backends.bailian_animate import BailianAnimateClient
            c = BailianAnimateClient(api_key=self._ds, mode="wan-pro")
        elif name == "tts":
            from .backends.minimax_tts import MiniMaxTTSClient
            c = MiniMaxTTSClient(api_key=self._ws)
        elif name == "music":
            from .backends.sonilo_music import SoniloMusicClient
            c = SoniloMusicClient(api_key=self._ws)
        elif name == "image":
            from .backends.bailian_image import BailianImageClient
            c = BailianImageClient(api_key=self._ds)
        else:
            raise ValueError(f"未知后端 {name}")
        self._clients[name] = c
        return c

    # ── @引用替换 ────────────────────────────────────────
    @staticmethod
    def _resolve(val: Any, artifacts: Dict[str, str]) -> Any:
        if isinstance(val, str) and val.startswith("@"):
            key = val[1:]
            if key not in artifacts:
                raise KeyError(f"引用 @{key} 尚无产物(前序调用失败?)")
            return artifacts[key]
        if isinstance(val, list):
            return [Executor._resolve(v, artifacts) for v in val]
        if isinstance(val, dict):
            return {k: Executor._resolve(v, artifacts) for k, v in val.items()}
        return val

    # ── 单次调用 ─────────────────────────────────────────
    def _invoke(self, seg_id: str, call: dict, params: dict) -> tuple:
        """返回 (产物路径, task_id, 错误)。dry-run 时产出占位路径。"""
        tool, cid = call["tool"], call["id"]
        stem = f"{seg_id}_{cid}"
        if self.dry:
            ext = ".png" if tool == "image_generation" else (
                ".mp3" if tool in ("minimax_tts", "sonilo_text_to_music") else ".mp4")
            return str(self.out / "gen" / f"{stem}{ext}"), "dry", ""

        if tool in LOCAL:                              # 本地工具
            ext = ".wav" if tool in ("isolate_voice", "concat_audio", "punch_up") else ".mp4"
            dst = self.out / "work" / f"{stem}{ext}"
            fn = LOCAL[tool]
            first = {"isolate_voice": "source", "concat_audio": "audios",
                     "punch_up": "audio", "concat_av": "videos"}.get(tool, "video")
            args = dict(params)
            positional = args.pop(first, None)
            return fn(positional, dst, **args), "local", ""

        # 远程调用
        if tool == "kling_omni_video":
            dst = self.out / "gen" / f"{stem}.mp4"
            ok, tid, err = self._client("kling").generate(
                prompt=params.get("prompt", ""), duration=params.get("duration", 5),
                save_to=str(dst), first_frame=params.get("first_frame"),
                refer=params.get("refer"), audio=bool(params.get("audio")),
                aspect_ratio=params.get("aspect_ratio"))
        elif tool == "animate_move":
            dst = self.out / "gen" / f"{stem}.mp4"
            ok, tid, err = self._client("animate").animate(
                params["ref"], params["driving"], str(dst))
        elif tool == "minimax_tts":
            dst = self.out / "gen" / f"{stem}.mp3"
            ok, tid, err = self._client("tts").speak(
                params["text"], str(dst), voice_id=params.get("voice_id"),
                emotion=params.get("emotion", "happy"))
        elif tool == "sonilo_text_to_music":
            dst = self.out / "gen" / f"{stem}.mp3"
            ok, tid, err = self._client("music").text_to_music(
                params["prompt"], int(params.get("duration", 8)), str(dst))
        elif tool == "image_generation":
            dst = self.out / "gen" / f"{stem}.png"
            ok, tid, err = self._client("image").generate(
                params["prompt"], str(dst), size=params.get("size", "720*1280"))
        else:
            return "", "", f"未实现的工具 {tool}"
        return (str(dst) if ok else ""), tid, err

    def _invoke_with_retry(self, seg_id: str, call: dict, params: dict) -> dict:
        rec = {"id": call["id"], "tool": call["tool"], "local": bool(call.get("local"))}
        for attempt in (1, 2):
            try:
                path, tid, err = self._invoke(seg_id, call, params)
            except Exception as e:                     # noqa: BLE001
                path, tid, err = "", "", f"{type(e).__name__}: {e}"
            if path:
                rec.update(ok=True, path=path, task_id=tid)
                return rec
            if any(k in err for k in DETERMINISTIC):    # 确定性拒绝, 重试无用
                log.info("    %s 确定性失败, 不重试: %s", call["id"], err[:90])
                break
            if attempt == 1:
                log.info("    %s 失败, 重试一次: %s", call["id"], err[:90])
        rec.update(ok=False, path="", task_id=tid, error=err)
        return rec

    # ── 逐段执行 ─────────────────────────────────────────
    def run_segment(self, seg: dict) -> dict:
        sid = seg["seg_id"]
        artifacts: Dict[str, str] = {}
        records: List[dict] = []
        log.info("  ▶ %s [%s] %d 步", sid, seg.get("skill_id", "?"), len(seg["calls"]))
        for call in seg["calls"]:                      # 顺序即拓扑序(校验器已保证)
            try:
                params = self._resolve(call["params"], artifacts)
            except KeyError as e:
                records.append({"id": call["id"], "tool": call["tool"],
                                "ok": False, "error": str(e)})
                log.warning("    %s 跳过: %s", call["id"], e)
                break
            t0 = time.time()
            rec = self._invoke_with_retry(sid, call, params)
            rec["elapsed_s"] = round(time.time() - t0, 1)
            records.append(rec)
            log.info("    %s %-22s %s %.0fs", "✓" if rec["ok"] else "✗",
                     rec["tool"], Path(rec.get("path", "")).name, rec["elapsed_s"])
            if not rec["ok"]:
                break
            artifacts[call["id"]] = rec["path"]
        final = records[-1]["path"] if records and records[-1].get("ok") else ""
        return {"seg_id": sid, "t0": seg.get("t0"), "t1": seg.get("t1"),
                "ok": bool(final), "output": final, "calls": records}

    # ── 整片 ─────────────────────────────────────────────
    def execute(self, plan: dict) -> dict:
        log.info("执行 %d 段 | 预计计费: 视频 %.0fs 音乐 %.0fs TTS %d 字 图像 %d 张",
                 len(plan["segments"]), plan["cost_estimate"]["video_s"],
                 plan["cost_estimate"]["music_s"], plan["cost_estimate"]["tts_chars"],
                 plan["cost_estimate"]["image_calls"])
        segs = [self.run_segment(s) for s in plan["segments"]]
        kept = [s["output"] for s in segs if s["ok"]]
        summary = {"product": plan.get("product_name"), "segments": segs,
                   "dropped": [s["seg_id"] for s in segs if not s["ok"]], "final": ""}
        if not kept:
            log.error("无成功段落, 无法合片")
        elif self.dry:
            log.info("dry-run: 跳过合片")
        else:
            final = self.out / "final.mp4"
            concat_av(kept, final)
            summary["final"] = str(final)
            summary["duration_s"] = round(probe_duration(final), 2)
            log.info("成片: %s (%.1fs, 保留 %d/%d 段)", final, summary["duration_s"],
                     len(kept), len(segs))
        (self.out / "execution.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
