#!/usr/bin/env python3
"""被拒镜头可通性实验台: s09/s10/s12/s14/s15 (example1)。

pro 轮里这五镜被前置检测拒绝(s09=NoHuman 镜中人太小; 其余=FullFace 手机挡脸)。
本脚本逐镜试三类"翻案"变体, 回答"到底跑不跑得通":

  raw    原样重试(主要验证拒绝是否确定性, std 轮已顺带回答一次)
  crop   中心裁剪放大(默认保留 65% 画幅再放回 720x1280): 人物占比变大,
         专攻 NoHuman"人太小"
  merge  相邻镜头合并加长(s09+s10 / s12+s13 / s14+s15): 检测扫整段,
         段内只要存在合格帧就可能过 —— 开场直切成功同款原理

费用: 被拒不计费, 只有真跑通的尝试按输出秒计费(merge 组时长较长, 全成上限
约 26 计费秒; 预期大部分被拒 = 接近零成本)。

用法:
  python test_hard_shots.py                          # 默认 crop+merge, wan-std
  python test_hard_shots.py --variants raw crop merge
  python test_hard_shots.py --shots s09 s14 --mode wan-pro
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
from splitter import FF_ENC, make_driving_clip    # noqa: E402

log = logging.getLogger("hook_remake")

SPLIT_DIR = HERE / "outputs/split_example1_20260812_203234"
SOURCE = HERE / "_examples/example1/video.mp4"
MIN_CLIP_S = 2.1

HOOK_OF = {"s09": "person_hook_2", "s10": "person_hook_2",
           "s12": "person_hook_3", "s14": "person_hook_3",
           "s15": "person_hook_3"}
# 相邻合并组: 名字 → (t0取自, t1取自, 用哪张hook)
MERGE_GROUPS = {
    "s09": ("s09+s10", 9, 10, "person_hook_2"),
    "s10": ("s09+s10", 9, 10, "person_hook_2"),
    "s12": ("s12+s13", 12, 13, "person_hook_3"),
    "s14": ("s14+s15", 14, 15, "person_hook_3"),
    "s15": ("s14+s15", 14, 15, "person_hook_3"),
}


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}\n{p.stderr[-500:]}")


def _cut(t0, t1, dst, crop_keep=None):
    vf = []
    if crop_keep:                                  # 中心裁剪后放回原分辨率
        f = crop_keep
        vf = ["-vf", (f"crop=iw*{f}:ih*{f}:(iw-iw*{f})/2:(ih-ih*{f})/2,"
                      f"scale=720:1280,setsar=1")]
    _run(["ffmpeg", "-y", "-i", str(SOURCE), "-ss", f"{t0:.3f}",
          "-to", f"{t1:.3f}", *vf, "-an", *FF_ENC, str(dst)])


def build_attempts(shots_sel, variants, crop_keep, out_dir):
    meta = json.loads((SPLIT_DIR / "cuts.json").read_text())
    rows = {f"s{r['idx']:02d}": r for r in meta["shots"]}
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    attempts, seen_merge = [], set()
    for name in shots_sel:
        r = rows[name]
        for var in variants:
            if var == "merge":
                gname, i0, i1, hook = MERGE_GROUPS[name]
                if gname in seen_merge:
                    continue
                seen_merge.add(gname)
                t0, t1 = rows[f"s{i0:02d}"]["t0"], rows[f"s{i1:02d}"]["t1"]
                dst = clip_dir / f"{gname.replace('+', '_')}.mp4"
                _cut(t0, t1, dst)
                attempts.append({"attempt": f"{gname}/merge", "hook": hook,
                                 "driving": str(dst), "dur": round(t1 - t0, 3)})
                continue
            dst = clip_dir / f"{name}_{var}.mp4"
            _cut(r["t0"], r["t1"], dst, crop_keep if var == "crop" else None)
            dur = r["duration_s"]
            if dur < MIN_CLIP_S:                   # s14 原样/裁剪仍需回文补齐
                shot = Shot(**{**r, "clip_path": str(dst)})
                driving, _, dur = make_driving_clip(shot, out_dir / var, MIN_CLIP_S)
            else:
                driving = str(dst)
            attempts.append({"attempt": f"{name}/{var}", "hook": HOOK_OF[name],
                             "driving": driving, "dur": dur})
    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+",
                    default=["s09", "s10", "s12", "s14", "s15"])
    ap.add_argument("--variants", nargs="+", default=["crop", "merge"],
                    choices=["raw", "crop", "merge"])
    ap.add_argument("--crop-keep", type=float, default=0.65,
                    help="crop 变体保留的画幅比例(中心), 默认 0.65")
    ap.add_argument("--mode", choices=["wan-std", "wan-pro"], default="wan-std")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = HERE / "outputs" / f"hardshot_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = build_attempts(args.shots, args.variants, args.crop_keep, out_dir)
    log.info("实验矩阵 %d 项: %s", len(attempts),
             ", ".join(a["attempt"] for a in attempts))

    client = BailianAnimateClient(api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                                  mode=args.mode)
    hooks = {a["hook"] for a in attempts}
    hook_oss = {h: client.upload(str(HERE / f"_examples/{h}.png")) for h in hooks}
    gen_dir = out_dir / "gen"
    gen_dir.mkdir(exist_ok=True)

    def _one(a):
        try:
            vid_oss = client.upload(a["driving"])
            dst = gen_dir / (a["attempt"].replace("/", "_") + ".mp4")
            ok, task_id, err = client.animate(hook_oss[a["hook"]], vid_oss, str(dst))
            a.update(task_id=task_id, passed=ok,
                     error=err, output=str(dst) if ok else "")
        except Exception as e:                     # noqa: BLE001
            a.update(passed=False, error=f"{type(e).__name__}: {e}")
        return a

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed({pool.submit(_one, a) for a in attempts}):
            a = f.result()
            log.info("%-16s → %s %s", a["attempt"],
                     "✅ 跑通" if a["passed"] else "❌ 被拒",
                     a.get("error", "")[:90])

    (out_dir / "results.json").write_text(
        json.dumps({"mode": args.mode, "crop_keep": args.crop_keep,
                    "attempts": attempts}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    n_ok = sum(1 for a in attempts if a.get("passed"))
    billed = sum(a["dur"] for a in attempts if a.get("passed"))
    log.info("结论: %d/%d 跑通, 实际计费≈%.0f 秒 | 详情 %s/results.json",
             n_ok, len(attempts), billed, out_dir)


if __name__ == "__main__":
    main()
