#!/usr/bin/env python
"""端到端跑通【BRAIN 编排的修复闭环】（真实后端，无 mock），逐回合打印。

这是 scripts/test_review_improve.py 的"代理化"升级版：把固定的 if-else tier 阶梯
换成一个真正会 function-calling 的 LLM 编排器（brain）。每回合：

    ① brain 读结构化评审（语义失败项 + 物理 verdict + metric 分数）
    ② brain 从工具菜单里挑【一个】工具调用，输出严格 JSON {tool,args,reason}
    ③ 执行该工具（regenerate / keyframe_edit / edit_clip / extend_clip /
       retrieve_replace / accept），重生成 + 重评审
    ④ 单调 Verifier 决定接受/拒绝（"brain 提议，闸门裁决"）；被拒的动作回灌给
       brain，下一回合它不会重复——这就是真实的 agent 循环，不是固定阶梯。

全部真实后端，无 mock：
    · brain      OpenAI LLM（build_llm openai）—— 通过 llm.complete + JSON 说话
    · 评审 VLM   OpenAI VLM（SemanticCritic + PhysicsCritic，build_mllm openai）
    · 生成/编辑  WaveSpeed（build_video_gen wavespeed，含 edit/extend 能力）
    · 物理测量   真实 CoTracker + GroundingDINO（需 GPU）；--no-physics-measure
                 时【省略】这个 critic（绝不改用 mock）

用法：
    export OPENAI_API_KEY=...  WAVESPEED_API_KEY=...
    python scripts/test_orchestrator.py --prompt "a glass falls off a table and shatters"
    # 无 GPU：再加 --no-physics-measure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.agents.defect_report import build_defect_report  # noqa: E402
from maestro.agents.generator import GeneratorAgent          # noqa: E402
from maestro.agents.orchestrator import OrchestratorAgent    # noqa: E402
from maestro.agents.refiner import RefinerAgent              # noqa: E402
from maestro.agents.verifier import VerifierAgent            # noqa: E402
from maestro.critics.board import ReviewBoard                # noqa: E402
from maestro.critics.physics import PhysicsCritic            # noqa: E402
from maestro.critics.physics_consistency import PhysicsConsistencyCritic  # noqa: E402
from maestro.critics.semantic import SemanticCritic          # noqa: E402
from maestro.critics.tournament import Tournament            # noqa: E402
from maestro.models import build_llm, build_mllm, build_video_gen  # noqa: E402
from maestro.models.image_edit import build_image_edit       # noqa: E402
from maestro.physics.annotate import annotate_physics        # noqa: E402
from maestro.physics.tracks import build_track_extractor     # noqa: E402
from maestro.tools.metric_tool import MetricTool             # noqa: E402
from maestro.types import ShotSpec                           # noqa: E402


def _section(t: str) -> None:
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def _show_review(tag: str, clip) -> None:
    print(f"  [{tag}] metric: " + "  ".join(
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


_LOCALIZED_TOOLS = {"regenerate_segment", "keyframe_edit_propagate", "frame_to_frame"}


def _show_defects(spec, clip) -> None:
    """Print the LOCALIZED DefectReport the brain reasons over this turn:
    which entity, which frames, severity, fix modality — worst-first."""
    rep = build_defect_report(clip, spec, fps=8)
    print(f"  ⟶ DefectReport（定位缺陷，worst-first，n_frames={rep.n_frames}）:")
    if not rep.defects:
        print("      （无定位缺陷）")
    for d in rep.sorted_by_severity():
        print(f"      - [{d.kind}/{d.fix_modality}] entity={d.entity or '?'} "
              f"帧{d.frame_range} 严重度={d.severity:.2f}  {d.note}")


def _cascade_depth(out_dir, spec, turn, tool) -> int:
    """Best-effort propagation cascade depth = number of *_cascade.mp4 segment
    files propagate_repair wrote for this turn (0 if not a localized tool)."""
    if tool not in _LOCALIZED_TOOLS:
        return 0
    prop_dir = Path(out_dir) / f"shot{spec.shot_idx:03d}_r{turn}_{tool}"
    return len(list(prop_dir.glob("seg*_cascade.mp4"))) if prop_dir.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a glass falls off a table and shatters on the floor")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI 模型（brain LLM + 评审 VLM）")
    ap.add_argument("--video-model", default="bytedance/seedance-v1-pro-t2v-480p")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda", help="CoTracker/GroundingDINO 设备")
    ap.add_argument("--max-turns", type=int, default=4, help="brain 修复回合上限")
    ap.add_argument("--n-candidates", type=int, default=2, help="初评 best-of-N 池")
    ap.add_argument("--no-physics-measure", action="store_true",
                    help="省略参考自由物理测量 critic（无 GPU 时用；不会改用 mock）")
    args = ap.parse_args()

    missing = [k for k in ("OPENAI_API_KEY", "WAVESPEED_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"❌ 缺少环境变量: {', '.join(missing)}")
        return 2

    base = Path(args.out_dir or os.getenv("MAESTRO_OUTPUT_ROOT") or REPO_ROOT / "outputs")
    run_dir = base / f"orchestrator_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"用户指令: {args.prompt}")
    print(f"输出目录: {run_dir.resolve()}")

    # —— 组装组件（全部真实后端，无 mock）——
    llm = build_llm({"name": "openai", "model": args.model})        # brain
    mllm = build_mllm({"name": "openai", "model": args.model})      # 评审 VLM
    video_gen = build_video_gen({"name": "wavespeed", "model_id": args.video_model})
    image_edit = build_image_edit(None)  # 真实 keyframe 编辑后端可在此替换
    critics = [SemanticCritic(mllm=mllm), PhysicsCritic(mllm=mllm)]
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
    orchestrator = OrchestratorAgent(
        llm=llm, generator=generator, refiner=refiner, image_edit=image_edit,
        max_turns=args.max_turns,
    )

    tournament = Tournament(judge=mllm)
    spec = ShotSpec(shot_idx=0, duration=5.0, prompt=args.prompt)
    spec.physics_annotation = annotate_physics(spec)
    print(f"\n物理标注: 实体={[(e.name, e.motion_class) for e in spec.physics_annotation.entities]}"
          f"  预期失效模式={[m.value for m in spec.physics_annotation.expected_modes]}")
    print(f"\nbrain 工具菜单: "
          + ", ".join(a["name"] for a in orchestrator.available_actions(
              video_gen=video_gen, asset_memory=None)))

    # ── 手写跑一遍 BRAIN 循环，逐回合打印（逻辑与 generate_shot_orchestrated 一致）──
    _section("① 初评（WaveSpeed 真生成 best-of-N → tournament 选最优）")
    candidates = []
    for s in range(args.n_candidates):
        c = generator.run(spec, run_dir, revision=0, seed=s, fps=8)
        board.review(c, spec, None, fps=8)
        candidates.append(c)
        _show_review(f"候选{s}", c)
    best = tournament.select(candidates, spec)
    _show_review("初评 best", best)

    brain_history: list[tuple[dict, str, float]] = []
    for turn in range(1, args.max_turns + 1):
        if board.all_passed(best):
            _section("收敛")
            print(f"  评审全过，brain 无需再修复。回合用尽前收敛于第 {turn-1} 回合。")
            break
        _section(f"② 回合 {turn} — brain 读【定位评审】→ function-call 一个工具")
        defect_report = build_defect_report(best, spec, fps=8)
        _show_defects(spec, best)
        menu = orchestrator.available_actions(video_gen=video_gen, asset_memory=None)
        decision = orchestrator.decide(best, spec, menu, brain_history,
                                       defect_report=defect_report)
        print("  brain 原始决策 (STRICT JSON):")
        print("    " + json.dumps(decision, ensure_ascii=False))
        if decision.get("tool") in _LOCALIZED_TOOLS:
            print(f"  → brain 选了【定位工具】{decision['tool']}"
                  "（修一个 segment → 前向级联重锚到收敛，而非整段重生成）")

        if decision.get("tool") == "accept":
            print("  → brain 选择 ACCEPT，停止修复。")
            brain_history.append((decision, "stop", best.metric_scores.get("weighted_total", 0.0)))
            break
        if decision.get("tool") == "__invalid__":
            print("  → brain 回复不可用（无法解析/越界工具）。真实环路会回落到确定性 "
                  "RepairRouter；本 demo 直接重试下一回合。")
            brain_history.append((decision, "invalid", best.metric_scores.get("weighted_total", 0.0)))
            continue

        print(f"  执行工具: {decision['tool']}  args={decision['args']}")
        cand = orchestrator.execute(decision, best, spec, run_dir, turn, board, fps=8)
        if cand is None:
            print("  → 工具无法执行（能力/素材缺失）。回落重试。")
            brain_history.append((decision, "noop", best.metric_scores.get("weighted_total", 0.0)))
            continue

        depth = _cascade_depth(run_dir, spec, turn, decision.get("tool"))
        if decision.get("tool") in _LOCALIZED_TOOLS:
            print(f"  → 传播级联深度 cascade_depth={depth}"
                  "（修复后向下游重锚的 segment 数；相似度>=阈值即提前停止）")
        _show_review(f"重评 turn{turn}", cand)
        before = best.metric_scores.get("weighted_total", 0.0)
        after = cand.metric_scores.get("weighted_total", 0.0)
        accepted = verifier.is_better(cand, best)
        print(f"  Verifier.is_better: 前={before:.4f} 后={after:.4f} → "
              f"{'接受 ✅' if accepted else '拒绝 ❌'}（单调改进才接受）")
        if accepted:
            best = cand
        brain_history.append((decision, "accepted" if accepted else "rejected", round(after, 4)))

    _section("结果")
    print(f"  最终 weighted_total = {best.metric_scores.get('weighted_total', 0.0):.4f}")
    print(f"  最终评审: 失败项 {len(best.checklist.failed_items)} 个, "
          f"物理 verdict {len(best.physics_verdicts)} 个")
    print(f"  brain 决策轨迹（{len(brain_history)} 回合）:")
    for i, (d, outcome, total) in enumerate(brain_history, 1):
        print(f"    回合{i}: tool={d.get('tool')} → {outcome} (total={total})")
    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
