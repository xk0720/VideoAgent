"""执行 agent: 分镜脚本 → 逐段生成 → 装配成片。

v1 只调视频生成模型(以后加剪辑等复杂流程)。三条纪律来自实战:
  1. 失败即剔除, 不用原片兜底(用户裁决) —— 但整片仍要能交付, 缺段照拼;
  2. 前置检测类拒绝(NoHuman/FullFace)不重试(实测确定性); 网络类错误重试一次;
  3. 生成物一律 conform 回段落声明的整数秒时长, 再配音轨拼接。
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from ..assemble import concat, conform, overlay_bgm, probe_duration
from ..backends.bailian_animate import BailianAnimateClient, is_deterministic_reject
from ..backends.seedance import SeedanceClient
from ..memory_store import MemoryStore
from ..schemas import SegmentPlan, ShotScript

log = logging.getLogger("viral_studio")


class ExecutorAgent:
    def __init__(self, mem: MemoryStore, out_dir: Path,
                 dashscope_key: str = "", wavespeed_key: str = "",
                 animate_mode: str = "wan-std", resolution: str = "720p",
                 generate_audio: bool = True, workers: int = 2):
        self.mem = mem
        self.out_dir = out_dir
        self.workers = workers
        self.animate_mode = animate_mode
        self._ds_key, self._ws_key = dashscope_key, wavespeed_key
        self._resolution, self._gen_audio = resolution, generate_audio
        self._animate: Optional[BailianAnimateClient] = None
        self._seedance: Optional[SeedanceClient] = None

    # 懒建: 脚本里没有某条路线时, 不因缺 key 而报错
    def animate_client(self) -> BailianAnimateClient:
        if self._animate is None:
            self._animate = BailianAnimateClient(api_key=self._ds_key,
                                                 mode=self.animate_mode)
        return self._animate

    def seedance_client(self) -> SeedanceClient:
        if self._seedance is None:
            self._seedance = SeedanceClient(api_key=self._ws_key,
                                            resolution=self._resolution,
                                            generate_audio=self._gen_audio)
        return self._seedance

    # ── 逐段生成 ─────────────────────────────────────────
    def _run_segment(self, seg: SegmentPlan) -> Dict:
        rec: Dict = {"seg_id": seg.seg_id, "mode": seg.mode,
                     "duration_s": seg.duration_s, "status": "failed",
                     "raw": "", "task_id": "", "error": "", "billed_s": 0.0}
        raw = self.out_dir / "gen" / f"{seg.seg_id}_raw.mp4"
        try:
            if seg.mode == "reuse_motion":
                card = self.mem.assets.get(seg.asset_ref or "")
                if not card:
                    rec["error"] = f"资产 {seg.asset_ref} 不在记忆库"
                    return rec
                driving = self.mem.asset_clip_path(seg.asset_ref)
                if not driving or not driving.exists():
                    rec["error"] = f"驱动素材缺失: {driving}"
                    return rec
                ok, task_id, err = self.animate_client().animate(
                    seg.person_hook_refs[0], str(driving), str(raw))
                rec["billed_s"] = probe_duration(str(driving)) if ok else 0.0
            else:
                refs = list(seg.person_hook_refs) + list(seg.product_image_refs)
                client = self.seedance_client()
                gen_s = client.snap_duration(seg.duration_s)
                ok, task_id, err = client.generate(
                    seg.prompt, seg.duration_s, str(raw), reference_images=refs)
                rec["billed_s"] = float(gen_s) if ok else 0.0
                if ok and gen_s > seg.duration_s:
                    log.info("%s 生成 %ds → 剪回 %.0fs(不足4s的必然浪费)",
                             seg.seg_id, gen_s, seg.duration_s)
            rec.update(task_id=task_id, error=err)
            if ok:
                rec.update(status="succeeded", raw=str(raw))
            elif not is_deterministic_reject(err):        # 网络类 → 免费重试一次
                log.info("%s 非确定性失败, 重试一次", seg.seg_id)
                return self._retry_once(seg, rec, raw)
        except Exception as e:                            # noqa: BLE001 单段不拖垮整片
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    def _retry_once(self, seg: SegmentPlan, rec: Dict, raw: Path) -> Dict:
        try:
            if seg.mode == "reuse_motion":
                driving = self.mem.asset_clip_path(seg.asset_ref)
                ok, task_id, err = self.animate_client().animate(
                    seg.person_hook_refs[0], str(driving), str(raw))
                rec["billed_s"] = probe_duration(str(driving)) if ok else 0.0
            else:
                refs = list(seg.person_hook_refs) + list(seg.product_image_refs)
                client = self.seedance_client()
                ok, task_id, err = client.generate(
                    seg.prompt, seg.duration_s, str(raw), reference_images=refs)
                rec["billed_s"] = float(client.snap_duration(seg.duration_s)) if ok else 0.0
            rec.update(task_id=task_id, error=err,
                       status="succeeded" if ok else "failed",
                       raw=str(raw) if ok else "")
        except Exception as e:                            # noqa: BLE001
            rec["error"] = f"重试仍失败: {type(e).__name__}: {e}"
        return rec

    # ── 主流程 ───────────────────────────────────────────
    def execute(self, script: ShotScript, bgm: Optional[str] = None,
                bgm_volume: float = 0.8) -> Dict:
        (self.out_dir / "gen").mkdir(parents=True, exist_ok=True)
        log.info("执行 %d 段(并发 %d)", len(script.segments), self.workers)

        records: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._run_segment, s): s for s in script.segments}
            for f in as_completed(futs):
                rec = f.result()
                records[rec["seg_id"]] = rec
                log.info("%s [%s] → %s %s", rec["seg_id"], rec["mode"],
                         rec["status"], rec["error"][:90])

        # 装配: 成功段 conform(配自带 BGM) → 拼接 → 可选全片 BGM
        parts, kept = [], []
        for seg in script.segments:                      # 保持时间轴顺序
            rec = records[seg.seg_id]
            if rec["status"] != "succeeded":
                continue
            audio = None
            if seg.bgm_source == "asset_bgm" and seg.asset_ref:
                p = self.mem.asset_bgm_path(seg.asset_ref)
                audio = str(p) if p and p.exists() else None
            dst = self.out_dir / "conform" / f"{seg.seg_id}.mp4"
            conform(rec["raw"], str(dst), seg.duration_s, audio)
            rec["conform"] = str(dst)
            parts.append(str(dst))
            kept.append(seg.seg_id)

        summary = {"segments": [records[s.seg_id] for s in script.segments],
                   "kept": kept,
                   "dropped": [s.seg_id for s in script.segments
                               if records[s.seg_id]["status"] != "succeeded"],
                   "billed_s": round(sum(r["billed_s"] for r in records.values()), 1),
                   "final": ""}
        if not parts:
            log.error("无任何成功段落, 无法合片")
            return summary

        final = self.out_dir / "final.mp4"
        concat(parts, str(final))
        if bgm and Path(bgm).exists():
            mixed = self.out_dir / "final_bgm.mp4"
            overlay_bgm(str(final), bgm, str(mixed), volume=bgm_volume)
            final = mixed
        summary["final"] = str(final)
        summary["final_duration_s"] = round(probe_duration(str(final)), 2)
        log.info("成片: %s (%.1fs, 保留 %d/%d 段, 计费≈%.0f 秒)",
                 final, summary["final_duration_s"], len(kept),
                 len(script.segments), summary["billed_s"])
        (self.out_dir / "execution.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
