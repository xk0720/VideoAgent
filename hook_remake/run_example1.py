#!/usr/bin/env python3
"""example1 专用驱动: 开场合并长片段 + 失败剔除 + 逐段原声(用户策略 2026-08-12)。

与 run_test.py 的三点不同:
  1. s00–s05 不逐镜补齐, 按 s05 结束时间从原片直切 [0, 5.00s) 连续长片段
     一次生成(天然满足 ≥2s 下限, 内部硬切属画面内容);
  2. 生成失败(NoHuman/FullFace/审核/其他)直接剔除不回退原片, 只记台账;
     仅提交前的网络类异常免费重试一次;
  3. 合片逐段严格配原 BGM: 每个保留片段配自己 [t0,t1) 的原声切片(段内音画
     同步), 生成物一律剪回原片段精确时长后再配对拼接 → 完整音视频。

用法: python run_example1.py --mode wan-pro   (之后再 --mode wan-std)
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bailian_animate import BailianAnimateClient  # noqa: E402
from interfaces import Shot, SourceInfo           # noqa: E402
from splitter import FF_ENC, conform_clip, make_driving_clip  # noqa: E402

log = logging.getLogger("hook_remake")

SPLIT_DIR = HERE / "outputs/split_example1_20260812_203234"
SOURCE = HERE / "_examples/example1/video.mp4"
MIN_CLIP_S = 2.1
OPEN_REEL_END_IDX = 5          # 开场合并到 s05(含)

HOOKS = {                      # 平均分: 开场→1, s06–s10→2, s11–s15→3
    "person_hook_1": HERE / "_examples/person_hook_1.png",
    "person_hook_2": HERE / "_examples/person_hook_2.png",
    "person_hook_3": HERE / "_examples/person_hook_3.png",
}


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}\n{p.stderr[-500:]}")


def build_units(out_dir: Path):
    """单元表: 开场长片段(直切) + s06..s15(复用切好的镜头文件)。"""
    meta = json.loads((SPLIT_DIR / "cuts.json").read_text())
    src = SourceInfo(**meta["source"])
    shots = [Shot(**r) for r in meta["shots"]]

    units = []
    head = shots[:OPEN_REEL_END_IDX + 1]
    reel = out_dir / "open_reel.mp4"
    _run(["ffmpeg", "-y", "-i", str(SOURCE), "-ss", f"{head[0].t0:.3f}",
          "-to", f"{head[-1].t1:.3f}", "-an", *FF_ENC, str(reel)])
    units.append({"name": "s00-05_reel", "t0": head[0].t0, "t1": head[-1].t1,
                  "dur": round(head[-1].t1 - head[0].t0, 3),
                  "hook": "person_hook_1", "driving": str(reel), "padded": False})

    for s in shots[OPEN_REEL_END_IDX + 1:]:
        hook = "person_hook_2" if s.idx <= 10 else "person_hook_3"
        driving, padded, _ = make_driving_clip(s, out_dir, MIN_CLIP_S)
        units.append({"name": f"s{s.idx:02d}", "t0": s.t0, "t1": s.t1,
                      "dur": s.duration_s, "hook": hook,
                      "driving": driving, "padded": padded})
    return src, units


def generate(units, client, out_dir: Path, workers: int = 2):
    hook_oss = {slot: client.upload(str(p)) for slot, p in HOOKS.items()
                if slot in {u["hook"] for u in units}}
    gen_dir = out_dir / "gen"
    gen_dir.mkdir(exist_ok=True)

    def _one(u):
        gen = gen_dir / f"{u['name']}_gen.mp4"
        for attempt in (1, 2):     # 第2次仅用于提交前网络异常(免费)
            try:
                vid_oss = client.upload(u["driving"])
                ok, task_id, err = client.animate(hook_oss[u["hook"]], vid_oss, str(gen))
                u["task_id"], u["error"] = task_id, err
                u["status"] = "succeeded" if ok else "skipped"
                if ok:
                    u["gen"] = str(gen)
                return u           # API 层失败(含 NoHuman/FullFace)不重掷 → 剔除
            except Exception as e:                    # noqa: BLE001 网络类
                u["status"], u["error"] = "skipped", f"{type(e).__name__}: {e}"
                log.warning("%s 网络异常(第%d次): %s", u["name"], attempt, e)
        return u

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, u): u for u in units}
        for f in as_completed(futs):
            u = f.result()
            log.info("%s [%s] → %s %s", u["name"], u["hook"], u["status"],
                     u.get("error", "")[:80])


def assemble(src: SourceInfo, units, out_dir: Path, mode: str) -> str:
    """最长连续成功段策略: 相邻成功片段归成 run, 每个 run 视频先拼、再按
    [run_t0, run_t1) 时间戳整段截原 BGM 铺上 —— BGM 只在被剔除处断开,
    run 内部零接缝; 全部成功时即"整片拼完 + 整条 BGM 一次铺"。"""
    work = out_dir / "assemble"
    work.mkdir(exist_ok=True)
    kept = [u for u in units if u["status"] == "succeeded"]   # 已按时间轴有序
    if not kept:
        log.error("没有任何成功片段, 无法合片")
        return ""
    for u in kept:                                  # 画面剪回精确原时长(丢补齐尾)
        conf = work / f"{u['name']}_v.mp4"
        conform_clip(u["gen"], str(conf), src, u["dur"])
        u["conform"] = str(conf)

    runs, cur = [], [kept[0]]
    for u in kept[1:]:                              # 时间戳连续(±20ms)才算同一 run
        if abs(u["t0"] - cur[-1]["t1"]) < 0.02:
            cur.append(u)
        else:
            runs.append(cur)
            cur = [u]
    runs.append(cur)

    run_files = []
    for ri, run in enumerate(runs):
        t0, t1 = run[0]["t0"], run[-1]["t1"]
        lst = work / f"run{ri}_list.txt"
        lst.write_text("\n".join(f"file '{u['conform']}'" for u in run),
                       encoding="utf-8")
        run_v = work / f"run{ri}_v.mp4"             # run 内先拼画面
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
              *FF_ENC, "-an", str(run_v)])
        run_av = work / f"run{ri}_av.mp4"           # 整段截 BGM 一次铺上
        _run(["ffmpeg", "-y", "-i", str(run_v),
              "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(SOURCE),
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
              "-c:a", "aac", "-ar", "44100", "-shortest", str(run_av)])
        run_files.append(str(run_av))
        log.info("run%d: %s → %s (%.2f–%.2fs, %d 段连续)",
                 ri, run[0]["name"], run[-1]["name"], t0, t1, len(run))

    final = out_dir / f"remake_{mode.replace('wan-', '')}.mp4"
    if len(run_files) == 1:
        Path(run_files[0]).rename(final)
    else:
        lst = out_dir / "concat_list.txt"
        lst.write_text("\n".join(f"file '{p}'" for p in run_files),
                       encoding="utf-8")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
              *FF_ENC, "-c:a", "aac", str(final)])
    log.info("合片: %s (保留 %d/%d 段, %d 个连续 run)",
             final, len(kept), len(units), len(runs))
    return str(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wan-pro", "wan-std"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = HERE / "outputs" / f"example1_{args.mode.replace('wan-', '')}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src, units = build_units(out_dir)
    est = sum(u["dur"] if not u["padded"] else max(u["dur"] * 2, MIN_CLIP_S)
              for u in units)
    log.info("单元: %d 个(开场合并+逐镜), 提交驱动总长≈%.0fs, mode=%s",
             len(units), est, args.mode)

    client = BailianAnimateClient(api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                                  mode=args.mode)
    for u in units:
        u.setdefault("status", "planned")
    generate(units, client, out_dir, args.workers)
    final = assemble(src, units, out_dir, args.mode)

    (out_dir / "manifest.json").write_text(
        json.dumps({"mode": args.mode, "final": final, "units": units},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for u in units if u["status"] == "succeeded")
    log.info("完成: %s | 成功 %d / 剔除 %d", final or "(无成片)",
             ok, len(units) - ok)


if __name__ == "__main__":
    main()
