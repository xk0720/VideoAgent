#!/usr/bin/env python
"""端到端跑通【窗口式全片生成】(真实后端,无 mock),逐阶段打印。

大循环(pipeline/window_loop.py,需求 §A-§E+§M):
    §A 剧本      prompt → 按时间顺序的 shot 列表 → StoryboardMemory 台账
    §B keyframe  brain 逐 shot 选:t2i 文生图 / 素材图 / 素材视频抽帧 / 无
    §C 窗口条件  brain 逐镜选:flf2v 桥接 / 尾段参考视频 / 上镜尾帧 i2v /
                 本镜 keyframe i2v / 纯 t2v(episode 命中可直接采纳)
    §D 每镜小循环 评审(VLM)→ 整理员汇总 → 缺陷定位(帧/段)→ brain 修复
                 → Verifier 闸门(全部在 generate_shot_orchestrated 内)
    §E 合成      时间顺序 ffmpeg concat → movie.mp4
    §M 记忆      storyboard.json 全程落盘;收工蒸馏 episode(good/bad)

全部真实后端,【模型一律来自 config】(configs/basic.yaml,--config 可换):
brain LLM=OpenAI;评审 VLM=Gemini(gemini-3.5-flash,$GEMINI_MODEL/
$GEMINI_BASE_URL 可覆盖);生成/编辑/t2i=WaveSpeed(seedance-2.0 全家桶)。
命令行不设模型旗子 —— 模型 id 归 config,每镜用哪条路由归 brain 的 skill
决策(image_plan / window_generation)。物理测量默认省略
(--with-physics-measure 打开,需 GPU)。

用法:
    export OPENAI_API_KEY=... GEMINI_API_KEY=... WAVESPEED_API_KEY=...
    python scripts/test_window_movie.py \
        --prompt "a glass falls off a table; shards scatter on the floor"
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

from maestro.agents.director import DirectorAgent              # noqa: E402
from maestro.agents.generator import GeneratorAgent            # noqa: E402
from maestro.agents.orchestrator import OrchestratorAgent      # noqa: E402
from maestro.agents.refiner import RefinerAgent                # noqa: E402
from maestro.agents.screenwriter import ScreenwriterAgent      # noqa: E402
from maestro.agents.verifier import VerifierAgent              # noqa: E402
from maestro.critics.board import ReviewBoard                  # noqa: E402
from maestro.critics.physics import PhysicsCritic              # noqa: E402
from maestro.critics.semantic import SemanticCritic            # noqa: E402
from maestro.critics.tournament import Tournament              # noqa: E402
from maestro.memory.episode_memory import EpisodeMemory        # noqa: E402
from maestro.memory.skill_library import SkillLibrary          # noqa: E402
from maestro.models import build_llm, build_mllm, build_video_gen  # noqa: E402
from maestro.models.image_edit import build_image_edit         # noqa: E402
from maestro.pipeline.window_loop import generate_movie_windowed  # noqa: E402
from maestro.tools.metric_tool import MetricTool               # noqa: E402


def _section(t: str) -> None:
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",
                    default="a glass falls off a table; shards scatter on the floor")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "basic.yaml"),
                    help="模型/参数全部来自这里(不设模型命令行旗子)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-turns", type=int, default=3, help="每镜修复回合上限")
    ap.add_argument("--n-candidates", type=int, default=1)
    ap.add_argument("--tail-seconds", type=float, default=2.0,
                    help="tiv2v_window 截取上镜尾段的秒数(config: window.tail_seconds)")
    ap.add_argument("--patience", type=int, default=2,
                    help="小循环连续 N 轮被拒即止损(≤0 关闭)")
    ap.add_argument("--quality-bar", type=float, default=None,
                    help="小循环总分达标线(≥即提前收工;默认关闭)")
    ap.add_argument("--with-physics-measure", action="store_true",
                    help="启用 CoTracker+GroundingDINO 测量 critic(需 GPU)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from maestro.config import load_yaml

    cfg = load_yaml(Path(args.config))
    models_cfg = cfg.get("models", {})
    # 必需 key 按 config 里选的后端推导(换供应商不用改脚本)
    _KEY_OF = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
               "qwen": "QWEN_API_KEY", "qwen-vl": "QWEN_API_KEY",
               "gpt": "OPENAI_API_KEY", "gpt-4o": "OPENAI_API_KEY",
               "wavespeed": "WAVESPEED_API_KEY"}
    def _key_for(spec):
        name = (spec or {}).get("name", "") if isinstance(spec, dict) else str(spec or "")
        return _KEY_OF.get(name.lower()) or _KEY_OF.get(name.split("-")[0].lower())
    required = {k for k in (_key_for(models_cfg.get("llm")),
                            _key_for(models_cfg.get("mllm")),
                            _key_for(models_cfg.get("video_gen"))) if k}
    missing = [k for k in sorted(required) if not os.getenv(k)]
    if missing:
        print(f"❌ 缺少环境变量: {', '.join(missing)}")
        return 2

    base = Path(args.out_dir or os.getenv("MAESTRO_OUTPUT_ROOT") or REPO_ROOT / "outputs")
    run_dir = base / f"movie_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"用户指令: {args.prompt}\n输出目录: {run_dir.resolve()}")

    llm = build_llm(models_cfg.get("llm"))
    mllm = build_mllm(models_cfg.get("mllm"))
    video_gen = build_video_gen(models_cfg.get("video_gen"))
    print(f"配置: {args.config}")
    print(f"  brain LLM = {getattr(llm, 'model', '?')}  |  评审 VLM = "
          f"{getattr(mllm, 'model', '?')}  |  视频 = "
          f"{getattr(video_gen, 'model_id', '?')}")
    critics = [SemanticCritic(mllm=mllm), PhysicsCritic(mllm=mllm)]
    if args.with_physics_measure:
        from maestro.critics.physics_consistency import PhysicsConsistencyCritic
        from maestro.physics.tracks import build_track_extractor
        critics.append(PhysicsConsistencyCritic(extractor=build_track_extractor({
            "name": "cotracker", "device": args.device,
            "detector": {"name": "groundingdino", "device": args.device}})))

    generator = GeneratorAgent(video_gen=video_gen)
    orchestrator = OrchestratorAgent(
        llm=llm, generator=generator, refiner=RefinerAgent(),
        image_edit=build_image_edit({"name": "wavespeed"}),  # 真实 keyframe 编辑(seedream-v4)
        skill_library=SkillLibrary(run_dir / "skills.jsonl"),
        max_turns=args.max_turns)
    # 长期 episode 记忆放稳定目录(跨 run 累积 —— 这正是它存在的意义)
    episode_memory = EpisodeMemory(base / "memory" / "episodes.jsonl")

    # Q-D 素材打标链:用户描述 > VLM caption > 文件名(真 VLM 才回填)
    from maestro.pipeline.window_loop import ensure_asset_descriptions
    n_cap = ensure_asset_descriptions(None, mllm)  # 无素材时为 0;接入素材库后生效
    if n_cap:
        print(f"  素材打标: VLM 补了 {n_cap} 条描述")

    _section("窗口式全片生成(§A 剧本 → §B' Image Plan → §C+§D 逐镜 → §E 合成 → §M 蒸馏)")
    res = generate_movie_windowed(
        args.prompt,
        board=ReviewBoard(critics=critics, metric_tool=MetricTool()),
        generator=generator, refiner=RefinerAgent(), verifier=VerifierAgent(judge=mllm),
        orchestrator=orchestrator, cache_dir=run_dir,
        screenwriter=ScreenwriterAgent(llm=llm), director=DirectorAgent(llm=llm),
        tournament=Tournament(judge=mllm),
        skill_library=orchestrator.skill_library,
        episode_memory=episode_memory, llm=llm,
        n_candidates=args.n_candidates, max_turns=args.max_turns,
        window_tail_s=args.tail_seconds,
        patience=args.patience, quality_bar=args.quality_bar)

    _section("brain 决策流水(§B keyframe + §C 条件;via=episode/llm/fallback)")
    for d in res.decisions:
        print(f"  [{d['stage']:9s}] {d['label']:<18s} → {d['strategy']:16s} "
              f"via={d['via']:8s} {d.get('reason', '')[:70]}")

    _section("台账终态(StoryboardMemory)")
    for e in res.storyboard.entries:
        last = e.reviews[-1] if e.reviews else {}
        print(f"  {e.label}: {e.status}  score={last.get('weighted_total')}  "
              f"kf={e.keyframe_source or '—'}  cond={e.condition.get('strategy')}  "
              f"stop={last.get('stop_reason', '?')}")

    _section("结果")
    print(f"  成片: {res.final_video or '(合成降级 — 单镜产物保留在台账里)'}")
    print(f"  episode: {res.episode_id}(长期记忆 {episode_memory.path}) ")
    print(f"  台账: {run_dir / 'storyboard.json'}")
    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
