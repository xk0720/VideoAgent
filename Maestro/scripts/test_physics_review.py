#!/usr/bin/env python
"""对一个【已有视频】跑完整的 non-AI 物理评审链（真实后端，无 mock），逐步打印。

这条链就是 PhysicsConsistencyCritic 在主循环里做的事，拆开演示：

    ① 定位   GroundingDINO 在第 0 帧检测 prompt 里的实体 → 检测框质心 = 种子点
    ② 追踪   CoTracker 从种子点逐帧追踪 → 归一化屏幕轨迹 (x,y)∈[0,1]，y 向下
    ③ 可靠性 certify()：抖动/漂移门 —— 追踪器在生成视频上会说谎；不可信的轨迹
             绝不产生"测量"判定（降级给 VLM 层），这是别家没有的门
    ④ 定律   fit_best_law()：静止/匀速/匀加速（重力向量自由拟合，不假设 9.81）
             里挑残差最小的解释 + 4 个异常检测器（瞬移→物体恒存 / 空中反向→
             重力惯性 / 能量增加→守恒 / 加速度尖峰→碰撞），violation =
             max(最优拟合残差, 最重异常)∈[0,1]
    ⑤ 评审   PhysicsConsistencyCritic 端到端复算，产出 per-entity、带帧范围的
             PhysicsVerdict（source="law_verifier"）——喂给 brain 的最终形态

全部轨迹与中间产物存进时间戳目录（绝无 tempfile）：
    tracks.json         每个实体逐帧 (x,y) 归一化轨迹
    seeds.json          第 0 帧检测框 + 质心种子（像素坐标）
    certificates.json   每条轨迹的可靠性证书（certified/score/reason）
    law_reports.json    最优定律 + 残差 + 参数 + 异常（帧范围/严重度）
    verdicts.json       PhysicsVerdict 最终评审输出
    overlay.mp4         轨迹叠加可视化（有 cv2 时；缺库只跳过，绝不伪造）

用法：
    python scripts/test_physics_review.py --video outputs/xxx/demo.mp4 \
        --prompt "a glass falls off a table and shatters" [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.critics.physics_consistency import PhysicsConsistencyCritic  # noqa: E402
from maestro.models.detection_backends import build_detector              # noqa: E402
from maestro.physics.annotate import annotate_physics                     # noqa: E402
from maestro.physics.laws import analyze_track                            # noqa: E402
from maestro.physics.reliability import certify                           # noqa: E402
from maestro.physics.track_extractor_backends import _decode_frames       # noqa: E402
from maestro.physics.tracks import build_track_extractor                  # noqa: E402
from maestro.pipeline.timeline import _probe_fps                          # noqa: E402
from maestro.types import CandidateClip, ShotSpec                         # noqa: E402


def _section(t: str) -> None:
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def _save(run_dir: Path, name: str, obj) -> None:
    p = run_dir / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    💾 {p}")


def _overlay(frames, tracks: dict, out_path: Path, fps: int) -> None:
    """把每个实体的轨迹画在帧上输出 mp4（cv2 可用时）。缺库只打印跳过。"""
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        print("    （无 cv2，跳过 overlay 渲染——不伪造）")
        return
    H, W = frames[0].shape[:2]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (W, H))
    for t_idx, frame in enumerate(frames):
        img = np.ascontiguousarray(frame[..., ::-1])  # RGB → BGR
        for k, (name, tr) in enumerate(tracks.items()):
            c = colors[k % len(colors)]
            pts = tr[: t_idx + 1]
            for j in range(1, len(pts)):
                p0 = (int(pts[j - 1][0] * W), int(pts[j - 1][1] * H))
                p1 = (int(pts[j][0] * W), int(pts[j][1] * H))
                cv2.line(img, p0, p1, c, 2)
            if pts:
                cur = (int(pts[-1][0] * W), int(pts[-1][1] * H))
                cv2.circle(img, cur, 6, c, -1)
                cv2.putText(img, name, (cur[0] + 8, cur[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        vw.write(img)
    vw.release()
    print(f"    💾 {out_path}（轨迹叠加）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="已有视频路径（mp4 等）")
    ap.add_argument("--prompt", default="a glass falls off a table and shatters",
                    help="视频对应的场景描述（决定检测哪些实体）")
    ap.add_argument("--device", default="cuda", help="CoTracker/GroundingDINO 设备")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"❌ 视频不存在: {video}")
        return 2

    base = Path(args.out_dir or REPO_ROOT / "outputs")
    run_dir = base / f"physics_review_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"输入视频: {video}\n输出目录: {run_dir.resolve()}")

    # 真实 fps / 帧数：一切帧范围都以此为准。
    frames = _decode_frames(video)
    if frames is None or len(frames) < 4:
        print("❌ 视频解码失败或帧数 < 4（定律拟合最少要 4 帧）")
        return 2
    fps = int(round(_probe_fps(video) or 24.0))
    n = len(frames)
    H, W = frames[0].shape[:2]
    print(f"解码: {n} 帧 @ {fps}fps，{W}x{H}")

    spec = ShotSpec(shot_idx=0, duration=n / fps, prompt=args.prompt)
    spec.physics_annotation = annotate_physics(spec)
    entities = spec.physics_annotation.entities
    print(f"物理标注实体: {[(e.name, e.motion_class) for e in entities]}")

    # ── ① 定位：GroundingDINO 第 0 帧检测 → 质心种子 ──
    _section("① 定位（GroundingDINO 零样本检测，第 0 帧）")
    detector = build_detector({"name": "groundingdino", "device": args.device})
    seeds = {}
    for e in entities:
        dets = detector.detect(frames[0], e.name)   # bbox 归一化 [0,1]
        if dets:
            best = max(dets, key=lambda d: d["score"])
            bx = best["bbox"]
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            seeds[e.name] = {"bbox_norm": [round(v, 4) for v in bx],
                             "score": round(float(best["score"]), 3),
                             "centroid_px": [round(cx * W, 1), round(cy * H, 1)]}
            print(f"    {e.name}: 框={seeds[e.name]['bbox_norm']} "
                  f"score={seeds[e.name]['score']} 质心(像素)={seeds[e.name]['centroid_px']}")
        else:
            seeds[e.name] = None
            print(f"    {e.name}: 第 0 帧未检出（追踪将退回启发式种子，"
                  f"该实体的测量判定不可靠——诚实降级）")
    _save(run_dir, "seeds.json", seeds)

    # ── ② 追踪：CoTracker 从种子逐帧追踪（extract 内部就是 检测→质心→追踪）──
    _section("② 追踪（CoTracker，输出归一化屏幕轨迹）")
    extractor = build_track_extractor({
        "name": "cotracker", "device": args.device,
        "detector": {"name": "groundingdino", "device": args.device},
    })
    clip = CandidateClip(shot_idx=0, video_path=video)
    tracks = extractor.extract(clip, spec, entities, fps) or {}
    for name, tr in tracks.items():
        print(f"    {name}: {len(tr)} 帧  首={tuple(round(v,3) for v in tr[0])} "
              f"末={tuple(round(v,3) for v in tr[-1])}")
    _save(run_dir, "tracks.json",
          {k: [[round(x, 5), round(y, 5)] for x, y in v] for k, v in tracks.items()})

    # ── ③ 可靠性门：certify()——不可信轨迹绝不产生测量判定 ──
    _section("③ 可靠性门（certify：抖动/漂移/长度）")
    certs = {}
    for name, tr in tracks.items():
        c = certify(tr, fps)
        certs[name] = {"certified": c.certified,
                       "confidence": round(c.confidence, 3), "reason": c.reason}
        print(f"    {name}: certified={c.certified} confidence={c.confidence:.3f} "
              f"reason={c.reason or '—'}")
    _save(run_dir, "certificates.json", certs)

    # ── ④ 定律拟合 + 异常定位（只对已认证的轨迹）──
    _section("④ 定律拟合（静止/匀速/匀加速·自由重力）+ 异常检测")
    law_reports = {}
    for name, tr in tracks.items():
        if not certs[name]["certified"]:
            print(f"    {name}: 未认证 → 跳过测量（降级 VLM 层）")
            continue
        rep = analyze_track(name, tr, fps)
        law_reports[name] = {
            "best_law": rep.fit.law,
            "residual": round(rep.fit.residual, 4),
            "params": {k: (round(v, 5) if isinstance(v, float) else v)
                       for k, v in rep.fit.params.items()},
            "motion_range": round(rep.motion_range, 4),
            "violation": rep.violation,
            "anomalies": [{"kind": a.kind, "mode": a.mode.value,
                           "frame_range": list(a.frame_range),
                           "severity": round(a.severity, 3), "note": a.note}
                          for a in rep.anomalies],
        }
        print(f"    {name}: 最优定律={rep.fit.law} 残差={rep.fit.residual:.4f} "
              f"violation={rep.violation}")
        for a in rep.anomalies:
            print(f"       异常: {a.kind} → {a.mode.value} 帧{list(a.frame_range)} "
                  f"severity={a.severity:.2f}  {a.note}")
    _save(run_dir, "law_reports.json", law_reports)

    # ── ⑤ 端到端评审：PhysicsConsistencyCritic（主循环用的就是它）──
    _section("⑤ 评审输出（PhysicsConsistencyCritic → PhysicsVerdict）")
    critic = PhysicsConsistencyCritic(extractor=extractor)
    critic.review(clip, spec, fps=fps)
    verdicts = [{
        "entity": v.entity, "mode": v.mode.value,
        "frame_range": list(v.frame_range), "severity": round(v.severity, 3),
        "source": v.source, "suggested_intervention": v.suggested_intervention,
    } for v in clip.physics_verdicts]
    if verdicts:
        for v in verdicts:
            print(f"    ⚠ {v['entity']}: {v['mode']} 帧{v['frame_range']} "
                  f"severity={v['severity']}")
            print(f"       修复建议: {v['suggested_intervention'][:100]}")
    else:
        print("    ✅ 无物理违规判定（所有已认证实体的运动都有物理解释，"
              "或没有可测量的实体）")
    _save(run_dir, "verdicts.json", verdicts)

    # ── 轨迹叠加可视化 ──
    _section("轨迹叠加渲染")
    _overlay(frames, tracks, run_dir / "overlay.mp4", fps)

    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
