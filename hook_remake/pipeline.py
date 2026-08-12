"""hook_remake 测试链路编排: 切镜 → 平均分 → animate-move → 对齐拼回。

设计文档 HookRemakeAgent_DESIGN.md 的最小可跑子集:
  - 跳过全部视频理解(无镜头卡/身份聚类/take 重组);
  - "选角" = 按 person_hook 数量平均分;
  - 单路线 MOVE(背景与 hook 图一致 —— 用户需求);
  - 无 QC, 失败镜头直接回退原片段, 保证成片总能拼出来。
"""
import json
import logging
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import requests as _plain_requests

from assigner import assign_hooks
from bailian_animate import BailianAnimateClient
from interfaces import HookAsset, Manifest, Shot, ShotJob, SourceInfo
from splitter import concat_and_mux, conform_clip, make_driving_clip, probe, split_shots

log = logging.getLogger("hook_remake")

IMG_MAX_SIDE = 4096          # 百炼参考图约束: 200–4096px, ≤5MB, 比例 1:3–3:1
IMG_MAX_BYTES = 5 * 1024 * 1024


# ── hooks 入库 ──────────────────────────────────────────────
def prepare_hooks(hooks: Dict[str, str], out_dir: Path) -> (List[HookAsset], List[str]):
    """person_hook_* 下载并规范化到 API 约束内; object_hook_* 本版忽略。"""
    hook_dir = out_dir / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    persons, ignored = [], []
    for slot in sorted(hooks):
        if not slot.startswith("person_hook"):
            ignored.append(slot)
            continue
        src = hooks[slot]
        raw = hook_dir / f"{slot}_raw"
        if src.startswith(("http://", "https://")):
            r = _plain_requests.get(src, timeout=120)
            r.raise_for_status()
            raw.write_bytes(r.content)
        else:
            p = Path(src).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"{slot}: 找不到 {src}")
            shutil.copy(p, raw)
        persons.append(HookAsset(slot=slot, source=src,
                                 local_path=_normalize_image(raw, hook_dir / f"{slot}.jpg")))
    if ignored:
        log.info("本版不处理商品特写, 忽略: %s", ", ".join(ignored))
    if not persons:
        raise ValueError("hooks 里没有 person_hook_*")
    return persons, ignored


def _normalize_image(raw: Path, dst: Path) -> str:
    """用 ffmpeg 把参考图规范到百炼约束内(超尺寸缩、超 5MB 降质重编)。"""
    p = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_streams", str(raw)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"参考图无法解析(不是图片?): {raw}")
    st = json.loads(p.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    if min(w, h) < 200:
        log.warning("%s 最短边 %dpx < 200px, API 可能拒绝", raw.name, min(w, h))
    ar = w / h
    if ar > 3 or ar < 1 / 3:
        log.warning("%s 宽高比 %.2f 超出 1:3–3:1, API 可能拒绝", raw.name, ar)

    vf = []
    if max(w, h) > IMG_MAX_SIDE:
        vf = ["-vf", f"scale='min({IMG_MAX_SIDE},iw)':'min({IMG_MAX_SIDE},ih)':"
                     f"force_original_aspect_ratio=decrease"]
    for q in ("2", "5", "8", "12"):     # jpg 质量逐级降到 5MB 以内
        subprocess.run(["ffmpeg", "-y", "-i", str(raw), *vf, "-frames:v", "1",
                        "-q:v", q, str(dst)], capture_output=True, text=True)
        if dst.exists() and dst.stat().st_size <= IMG_MAX_BYTES:
            return str(dst)
    raise RuntimeError(f"{raw.name} 压不进 5MB")


# ── 主流程 ──────────────────────────────────────────────────
def run(video: str, hooks: Dict[str, str], cfg: dict, out_root: Path,
        dry_run: bool = False, assume_yes: bool = False) -> Manifest:
    out_dir = out_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("输出目录: %s", out_dir)

    src = probe(video)
    log.info("原片: %dx%d @%.2ffps, %.1fs, 音轨=%s",
             src.width, src.height, src.fps, src.duration_s, src.has_audio)

    sp = cfg["split"]
    shots = split_shots(src, out_dir,
                        threshold=sp["threshold"],
                        min_scene_len_frames=sp["min_scene_len_frames"],
                        fallback_interval_s=sp["fallback_interval_s"],
                        max_clip_s=sp["max_clip_s"])

    persons, ignored = prepare_hooks(hooks, out_dir)
    assignment = assign_hooks(shots, [a.slot for a in persons], cfg["assign"])

    # 排产(含 <2.1s 的回文补齐), limit 之外的镜头标 skipped
    limit = int(cfg.get("limit", 0))
    jobs: List[ShotJob] = []
    for shot in shots:
        job = ShotJob(shot_idx=shot.idx, hook_slot=assignment[shot.idx])
        if limit and shot.idx >= limit:
            job.status = "skipped"
        else:
            drv, padded, drv_s = make_driving_clip(shot, out_dir, sp["min_clip_s"])
            job.driving_path, job.padded, job.driving_s = drv, padded, drv_s
        jobs.append(job)

    est = round(sum(j.driving_s for j in jobs if j.status == "planned"), 1)
    manifest = Manifest(source=src, person_hooks=persons, ignored_hooks=ignored,
                        shots=shots, assignment=assignment, jobs=jobs,
                        config=cfg, estimated_billed_s=est)
    _save(manifest, out_dir)

    n_plan = sum(1 for j in jobs if j.status == "planned")
    log.info("排产: %d 生成 / %d 跳过(limit=%s), 预计计费 ≈ %.0f 秒视频",
             n_plan, len(jobs) - n_plan, limit or "全量", est)
    if dry_run:
        log.info("dry-run 结束, 台账见 %s/manifest.json", out_dir)
        return manifest
    if not assume_yes:
        ans = input(f"将真实调用 {cfg['model']}({cfg['mode']}) {n_plan} 次, "
                    f"预计计费约 {est:.0f} 秒视频, 继续? [y/N] ")
        if ans.strip().lower() != "y":
            log.info("已取消(可先 --dry-run 检查台账)")
            return manifest

    # 生成: hook 图串行预上传(缓存去重), 镜头并行
    client = BailianAnimateClient(
        api_key=cfg["api_key"], model=cfg["model"], mode=cfg["mode"],
        check_image=cfg["check_image"], watermark=cfg["watermark"],
        base_url=cfg["base_url"], poll_interval_s=cfg["poll_interval_s"],
        timeout_s=cfg["timeout_s"])
    hook_oss = {}
    for a in persons:
        a.oss_url = client.upload(a.local_path)
        hook_oss[a.slot] = a.oss_url

    gen_dir = out_dir / "gen"
    gen_dir.mkdir(exist_ok=True)

    def _one(job: ShotJob) -> ShotJob:
        try:
            video_oss = client.upload(job.driving_path)
            gen = gen_dir / f"shot_{job.shot_idx:03d}_gen.mp4"
            ok, task_id, err = client.animate(hook_oss[job.hook_slot],
                                              video_oss, str(gen))
            job.task_id = task_id
            if ok:
                job.status, job.gen_path = "succeeded", str(gen)
            else:
                job.status, job.error = "failed", err
        except Exception as e:                       # noqa: BLE001 — 单镜失败不拖垮整片
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            log.error("镜头 %d 异常: %s", job.shot_idx, job.error)
        return job

    todo = [j for j in jobs if j.status == "planned"]
    with ThreadPoolExecutor(max_workers=int(cfg["workers"])) as pool:
        futs = {pool.submit(_one, j): j for j in todo}
        for fut in as_completed(futs):
            j = fut.result()
            log.info("镜头 %d [%s] → %s", j.shot_idx, j.hook_slot, j.status)

    # 对齐 + 拼回: 成功用生成片, 失败/跳过回退原片段
    conf_dir = out_dir / "conform"
    conf_dir.mkdir(exist_ok=True)
    ordered = []
    for shot, job in zip(shots, jobs):
        conf = conf_dir / f"shot_{shot.idx:03d}.mp4"
        source_clip = job.gen_path if job.status == "succeeded" else shot.clip_path
        conform_clip(source_clip, str(conf), src, shot.duration_s)
        job.conform_path = str(conf)
        ordered.append(str(conf))

    manifest.final_video = concat_and_mux(ordered, src, out_dir, cfg["audio"])
    _save(manifest, out_dir)
    n_ok = sum(1 for j in jobs if j.status == "succeeded")
    n_fail = sum(1 for j in jobs if j.status == "failed")
    log.info("完成: %s (成功 %d / 失败回退 %d / 跳过 %d)",
             manifest.final_video, n_ok, n_fail, len(jobs) - n_ok - n_fail)
    return manifest


def _save(m: Manifest, out_dir: Path) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(m.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8")
