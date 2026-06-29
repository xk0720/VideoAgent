#!/usr/bin/env python
"""端到端跑通 review → 改进 闭环（真实后端），把每一环打印出来确认逻辑。

抽取的是现有 pipeline/generate_loop.py::generate_shot 的核心逻辑，但拆开手写、
逐步打印，方便你核对“评审之后怎么用于改进”：

    ① 生成        WaveSpeed 真生成一个候选 clip
    ② 评审        ReviewBoard 跑多 critic：
                    · SemanticCritic(VLM assess_semantic)      → m1 + 失败项 fix
                    · PhysicsCritic(VLM assess_physics)         → p1 + 物理 verdict
                    · PhysicsConsistencyCritic(参考自由定律验证) → p2 + 测量 verdict
                  打印每个 critic 的原始判决 + metric 总分
    ③ 决策        RefinerAgent.plan(clip) 把 verdict/fix 聚成：
                    · extra_prompt（反违例提示词，喂重生成）
                    · 要编辑的关键帧 + 编辑指令（喂 image-edit→条件重生成）
                  打印这份“改进计划”——这就是 review 如何转成动作
    ④ 改进        用提示词重生成一次（Tier-1 形态），再评审，打印前后总分对比
                  Verifier.is_better 决定是否接受

全部真实后端，无 mock：LLM/VLM 用 OpenAI，视频用 WaveSpeed，物理测量用真实
CoTracker+GroundingDINO（需 GPU）。无 GPU 时加 --no-physics-measure，会【省略】
物理测量 critic（绝不改用 mock）。所有产物落到带时间戳的目录。

用法：
    export OPENAI_API_KEY=...  WAVESPEED_API_KEY=...
    python scripts/test_review_improve.py --prompt "a glass falls off a table and shatters"
    # 无 GPU：再加 --no-physics-measure
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.agents.generator import GeneratorAgent          # noqa: E402
from maestro.agents.refiner import RefinerAgent              # noqa: E402
from maestro.agents.verifier import VerifierAgent            # noqa: E402
from maestro.critics.board import ReviewBoard                # noqa: E402
from maestro.critics.physics import PhysicsCritic            # noqa: E402
from maestro.critics.physics_consistency import PhysicsConsistencyCritic  # noqa: E402
from maestro.critics.semantic import SemanticCritic          # noqa: E402
from maestro.models import build_mllm, build_video_gen       # noqa: E402
from maestro.physics.annotate import annotate_physics        # noqa: E402
from maestro.physics.tracks import build_track_extractor     # noqa: E402
from maestro.tools.metric_tool import MetricTool             # noqa: E402
from maestro.types import ShotSpec                           # noqa: E402


def _section(t: str) -> None:
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def _show_review(tag: str, clip) -> None:
    print(f"\n  [{tag}] 评审结果:")
    print(f"    metric: " + "  ".join(
        f"{k}={v}" for k, v in clip.metric_scores.items()))
    fails = clip.checklist.failed_items
    print(f"    语义失败项 {len(fails)}:")
    for it in fails:
        print(f"      - ({it.kind}) {it.question}"
              + (f"  → fix: {it.fix_instruction}" if it.fix_instruction else ""))
    print(f"    物理 verdict {len(clip.physics_verdicts)}:")
    for v in clip.physics_verdicts:
        print(f"      - [{v.mode.value}] 帧{v.frame_range} 严重度={v.severity:.2f}"
              f" src={v.source} → {v.suggested_intervention}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a glass falls off a table and shatters on the floor")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI 模型（LLM+VLM）")
    ap.add_argument("--video-model", default="bytedance/seedance-v1-pro-t2v-480p")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda", help="CoTracker/GroundingDINO 设备")
    ap.add_argument("--no-physics-measure", action="store_true",
                    help="省略参考自由物理测量 critic（无 GPU 时用；不会改用 mock）")
    args = ap.parse_args()

    missing = [k for k in ("OPENAI_API_KEY", "WAVESPEED_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"❌ 缺少环境变量: {', '.join(missing)}")
        return 2

    base = Path(args.out_dir or os.getenv("MAESTRO_OUTPUT_ROOT") or REPO_ROOT / "outputs")
    run_dir = base / f"review_improve_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"用户指令: {args.prompt}")
    print(f"输出目录: {run_dir.resolve()}")

    # —— 组装组件（全部真实后端，无 mock）——
    mllm = build_mllm({"name": "openai", "model": args.model})
    video_gen = build_video_gen({"name": "wavespeed", "model_id": args.video_model})
    critics = [
        SemanticCritic(mllm=mllm),     # m1 语义：OpenAI VLM assess_semantic（真）
        PhysicsCritic(mllm=mllm),      # p1 物理：OpenAI VLM assess_physics（真）
    ]
    # p2 参考自由测量：真实 CoTracker + GroundingDINO（需要 GPU+torch）。
    # 无 GPU 时用 --no-physics-measure 直接【省略】这个 critic —— 不 mock 它。
    if not args.no_physics_measure:
        critics.append(PhysicsConsistencyCritic(
            extractor=build_track_extractor({
                "name": "cotracker", "device": args.device,
                "detector": {"name": "groundingdino", "device": args.device},
            })
        ))
        print("  p2 测量: 真实 CoTracker+GroundingDINO（需 GPU；缺 torch 会大声报错）")
    else:
        print("  p2 测量: 已省略（--no-physics-measure；不使用 mock）")
    board = ReviewBoard(critics=critics, metric_tool=MetricTool())
    generator = GeneratorAgent(video_gen=video_gen)
    refiner = RefinerAgent()
    verifier = VerifierAgent()

    spec = ShotSpec(shot_idx=0, duration=5.0, prompt=args.prompt)
    spec.physics_annotation = annotate_physics(spec)
    print(f"\n物理标注: 实体={[(e.name, e.motion_class) for e in spec.physics_annotation.entities]}"
          f"  预期失效模式={[m.value for m in spec.physics_annotation.expected_modes]}")

    # ① 生成
    _section("① 生成（WaveSpeed 真生成第一个候选）")
    best = generator.run(spec, run_dir, revision=0, seed=0, fps=8)
    print(f"  ✅ clip: {best.video_path}  ({Path(best.video_path).stat().st_size/1024:.0f} KB)")

    # ② 评审
    _section("② 评审（多 critic：语义 VLM + 物理 VLM + 参考自由测量）")
    board.review(best, spec, None, fps=8)
    _show_review("初评 rev0", best)

    # ③ 决策：review 如何转成改进动作
    _section("③ 决策（RefinerAgent.plan —— review verdict → 可执行修复）")
    plan = refiner.plan(best)
    print(f"  extra_prompt（反违例提示词，喂重生成）:\n    {plan['extra_prompt'] or '（无失败项）'}")
    print(f"  要编辑的关键帧 idx: {plan['edit_keyframe_idx']}")
    print(f"  关键帧编辑指令: {plan['edit_instruction'] or '（无）'}")

    if board.all_passed(best):
        _section("结果")
        print("  初评即全过，无需改进。review→改进链路验证完毕（无缺陷可修）。")
        print(f"\n📂 产物在: {run_dir.resolve()}")
        return 0

    # ④ 改进：用提示词重生成一次（Tier-1 形态），再评审，前后对比
    _section("④ 改进（Tier-1 形态：用 verdict 提示词重生成 → 再评审 → 前后对比）")
    hint = plan["extra_prompt"]
    print(f"  用提示词重生成: {hint}")
    cand = generator.run(spec, run_dir, revision=1, seed=1,
                         extra_prompt=hint, fps=8)
    board.review(cand, spec, None, fps=8)
    _show_review("重评 rev1", cand)

    before = best.metric_scores.get("weighted_total", 0.0)
    after = cand.metric_scores.get("weighted_total", 0.0)
    accepted = verifier.is_better(cand, best)
    _section("结果")
    print(f"  前(rev0) weighted_total = {before:.4f}")
    print(f"  后(rev1) weighted_total = {after:.4f}")
    print(f"  Verifier.is_better → {accepted}  （单调改进才接受）")
    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
