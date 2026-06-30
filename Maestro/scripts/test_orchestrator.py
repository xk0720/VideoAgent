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
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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
from maestro.memory.skill_library import SkillLibrary       # noqa: E402
from maestro.physics.annotate import annotate_physics        # noqa: E402
from maestro.physics.tracks import build_track_extractor     # noqa: E402
from maestro.pipeline.generate_loop import generate_shot_orchestrated  # noqa: E402
from maestro.tools.metric_tool import MetricTool             # noqa: E402
from maestro.types import ShotSpec                           # noqa: E402


def _section(t: str) -> None:
    print(f"\n{'='*72}\n{t}\n{'='*72}")


# localized tools = those that trigger segment-level repair + downstream
# propagation (rendered specially in the per-turn trace below).
_LOCALIZED_TOOLS = {"regenerate_segment", "keyframe_edit_propagate", "frame_to_frame"}


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
    # RETRIEVE-FIRST repair skills (the headline): a learned, verified repair
    # workflow is replayed (via=skill) before the LLM re-reasons (via=llm).
    skill_library = SkillLibrary(run_dir / "skills.jsonl")
    orchestrator = OrchestratorAgent(
        llm=llm, generator=generator, refiner=refiner, image_edit=image_edit,
        skill_library=skill_library, max_turns=args.max_turns,
    )

    tournament = Tournament(judge=mllm)
    spec = ShotSpec(shot_idx=0, duration=5.0, prompt=args.prompt)
    spec.physics_annotation = annotate_physics(spec)
    print(f"\n物理标注: 实体={[(e.name, e.motion_class) for e in spec.physics_annotation.entities]}"
          f"  预期失效模式={[m.value for m in spec.physics_annotation.expected_modes]}")
    print(f"\nbrain 工具菜单: "
          + ", ".join(a["name"] for a in orchestrator.available_actions(
              video_gen=video_gen, asset_memory=None)))

    # ── 调用【真实】的 generate_shot_orchestrated（不再手写平行循环，杜绝漂移）──
    # brain 循环、retrieve-first 技能重放、片段传播、禁止过早 accept、收敛蒸馏，
    # 全部发生在真实函数内部；demo 只负责【渲染它记录的逐回合轨迹 res.actions】。
    _section("① 初评 + ② brain 修复循环（全部在真实 generate_shot_orchestrated 内运行）")
    print("  （逐回合的 INFO 日志由 maestro.logger 实时打印；下面再结构化复盘）")
    res = generate_shot_orchestrated(
        spec, board=board, generator=generator, refiner=refiner, verifier=verifier,
        cache_dir=run_dir, orchestrator=orchestrator, asset_memory=None,
        image_edit=image_edit, tournament=tournament, skill_library=skill_library,
        fps=8, n_candidates=args.n_candidates, max_turns=args.max_turns,
    )

    # ── 复盘真实循环记录的逐回合轨迹（res.actions：真实发生了几回合就有几条）──
    _section(f"③ brain 逐回合决策轨迹（真实 res.actions，共 {len(res.actions)} 回合）")
    for turn, a in enumerate(res.actions, 1):
        via = a.get("via", "?")
        tag = ("SKILL（重放已学修复工作流）" if via == "skill"
               else "LLM（在完整原子工具盘上重新推理）" if via == "llm"
               else via)
        print(f"\n  ── 回合 {turn} ──  via={tag}")
        defects = a.get("defects", [])
        print(f"    本回合定位缺陷（DefectReport，{len(defects)} 个）:")
        for d in defects:
            print(f"      - {d}")
        print(f"    brain 决策: tool={a.get('tool')}  args={a.get('args', {})}")
        if a.get("reason"):
            print(f"      reason: {a['reason']}")
        if a.get("tool") in _LOCALIZED_TOOLS:
            print("      → 定位工具：修一个 segment 后向下游级联重锚（相似度收敛即停）")
        print(f"    Verifier 裁决: outcome={a.get('outcome')}"
              + (f"  new_total={a.get('new_total')}" if "new_total" in a else ""))

    _section("结果")
    best = res.clip
    print(f"  收敛 converged = {res.converged}")
    print(f"  最终 weighted_total = {best.metric_scores.get('weighted_total', 0.0):.4f}")
    print(f"  最终评审: 失败项 {len(best.checklist.failed_items)} 个, "
          f"物理 verdict {len(best.physics_verdicts)} 个")
    # 蒸馏在真实循环内部完成（distill_repair）——这里只读取它的产物 id
    print(f"  蒸馏出的修复技能 id（res.distilled_repair_skill_id）= "
          f"{res.distilled_repair_skill_id or '（无：未收敛/无非平凡缺陷）'}")
    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
