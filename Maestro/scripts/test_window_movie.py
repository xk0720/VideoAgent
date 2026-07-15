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

# .env 自动加载(仓库根;已导出的环境变量优先——文件只是便利,不是权威)
from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

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
    ap.add_argument("--baseline-anchor", action="store_true",
                    help="开工先按用户指令一次直出锚点视频,收尾与成片盲测"
                         "对比(附加物,不影响正流程)")
    ap.add_argument("--anchor-duration", type=int, default=None,
                    help="锚点视频时长秒数(默认不传 = API 默认)")
    ap.add_argument("--prompt-enhancer", action="store_true",
                    help="启用 prompt 润色 agent(官方 prompt 技巧技能,"
                         "按条件事实重写每镜 video prompt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", action="append", default=[], metavar="PATH[::DESC]",
                    help="用户提供的图片素材(可多次);'::' 后接英文描述,"
                         "不给则由 VLM 自动打标(Q-D 链)")
    ap.add_argument("--video", action="append", default=[], metavar="PATH[::DESC]",
                    help="用户提供的视频素材(可多次);'::' 后接英文描述")
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
    # 任务 0:每次 WaveSpeed 调用(模型名+参数)落盘可核对
    vg_cfg = dict(models_cfg.get("video_gen") or {})
    vg_cfg.setdefault("call_log", str(run_dir / "wavespeed_calls.jsonl"))
    video_gen = build_video_gen(vg_cfg)
    # debug:brain 每次决策的原始输出 + 解析结果(核对策略→模型映射用)
    from maestro.logging_utils import set_brain_log
    set_brain_log(run_dir / "brain_calls.jsonl")
    print(f"配置: {args.config}")
    print(f"  brain LLM = {getattr(llm, 'model', '?')}  |  评审 VLM = "
          f"{getattr(mllm, 'model', '?')}  |  视频 = "
          f"{getattr(video_gen, 'model_id', '?')}")
    print(f"  调用日志: {vg_cfg['call_log']}")
    print(f"  brain 决策日志: {run_dir / 'brain_calls.jsonl'}")
    # ── 用户素材入库(--image/--video)→ AssetMemory ─────────────────────
    from maestro.tools.retrieval_tool import RetrievalTool
    from maestro.types import AssetMemory, Identity, Shot

    def _split_asset(arg: str) -> tuple[str, str]:
        path, _, desc = arg.partition("::")
        return path.strip(), desc.strip()

    asset_memory = AssetMemory()
    for i, a in enumerate(args.image):
        pth, desc = _split_asset(a)
        if not Path(pth).is_file():
            print(f"❌ 图片素材不存在: {pth}")
            return 2
        asset_memory.identity_anchors[f"img{i}"] = Identity(
            identity_id=f"img{i}", name=Path(pth).stem,
            description=desc, source=str(Path(pth).resolve()))
    for i, a in enumerate(args.video):
        pth, desc = _split_asset(a)
        vp = Path(pth)
        if not vp.is_file():
            print(f"❌ 视频素材不存在: {pth}")
            return 2
        # 时长用 ffprobe 探;探不到给 5s(只影响元数据,不影响抽帧)
        try:
            import subprocess as _sp
            out = _sp.run(["ffprobe", "-v", "error", "-show_entries",
                           "format=duration", "-of",
                           "default=noprint_wrappers=1:nokey=1", str(vp)],
                          capture_output=True, text=True, timeout=30)
            dur = float(out.stdout.strip())
        except Exception:
            dur = 5.0
        asset_memory.video_shots[f"vid{i}"] = Shot(
            shot_id=f"vid{i}", source_video=str(vp.resolve()),
            start_time=0.0, end_time=dur,
            caption=desc or vp.stem.replace("_", " "))
    retrieval = RetrievalTool(asset_memory) if (
        asset_memory.identity_anchors or asset_memory.video_shots) else None

    # Q-D 素材打标链:用户描述 > VLM caption > 文件名(真 VLM 才回填)
    from maestro.pipeline.window_loop import ensure_asset_descriptions
    n_cap = ensure_asset_descriptions(asset_memory, mllm)
    if asset_memory.identity_anchors or asset_memory.video_shots:
        print(f"  素材库: {asset_memory.summarize()}"
              + (f";VLM 补标 {n_cap} 条" if n_cap else ""))

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
        retrieval=retrieval,   # 修复工具 retrieve_replace 的素材入口
        max_turns=args.max_turns)
    # 长期 episode 记忆放稳定目录(跨 run 累积 —— 这正是它存在的意义)
    episode_memory = EpisodeMemory(base / "memory" / "episodes.jsonl")


    _section("窗口式全片生成(§A 剧本 → §B' Image Plan → §C+§D 逐镜 → §E 合成 → §M 蒸馏)")
    prompt_enhancer = None
    if args.prompt_enhancer:
        from maestro.agents.prompt_enhancer import PromptEnhancerAgent
        prompt_enhancer = PromptEnhancerAgent(llm=llm)
    res = generate_movie_windowed(
        args.prompt,
        board=ReviewBoard(critics=critics, metric_tool=MetricTool()),
        generator=generator, refiner=RefinerAgent(), verifier=VerifierAgent(judge=mllm),
        orchestrator=orchestrator, cache_dir=run_dir,
        asset_memory=asset_memory, retrieval=retrieval,
        screenwriter=ScreenwriterAgent(llm=llm), director=DirectorAgent(llm=llm),
        tournament=Tournament(judge=mllm),
        skill_library=orchestrator.skill_library,
        episode_memory=episode_memory, llm=llm,
        n_candidates=args.n_candidates, max_turns=args.max_turns,
        window_tail_s=args.tail_seconds,
        patience=args.patience, quality_bar=args.quality_bar,
        baseline_anchor=args.baseline_anchor,
        baseline_anchor_duration=args.anchor_duration,
        prompt_enhancer=prompt_enhancer)

    _section("brain 决策流水(§B keyframe + §C 条件;via=episode/llm/fallback)")
    for d in res.decisions:
        print(f"  [{d['stage']:9s}] {d.get('label', '—'):<18s} → "
              f"{str(d.get('strategy', d.get('route', '—'))):16s} "
              f"via={d.get('via', '—'):8s} {str(d.get('reason', ''))[:70]}")

    _section("台账终态(StoryboardMemory)")
    for e in res.storyboard.entries:
        last = e.reviews[-1] if e.reviews else {}
        print(f"  {e.label}: {e.status}  score={last.get('weighted_total')}  "
              f"kf={e.keyframe_source or '—'}  cond={e.condition.get('strategy')}  "
              f"stop={last.get('stop_reason', '?')}")

    _section("结果")
    print(f"  成片: {res.final_video or '(合成降级 — 单镜产物保留在台账里)'}")
    if res.baseline_anchor:
        a = res.baseline_anchor
        print(f"  基线锚点: {a['path']}(route={a['route']}, via={a['via']})")
        v = a.get("final_vs_anchor")
        if v:
            print(f"  成片 vs 锚点(盲测): {v.get('conclusion')} "
                  f"score={v.get('score'):+d} dims={v.get('dim_scores')}")
            print(f"    → {v.get('summary', '')}")
        else:
            print("  成片 vs 锚点: 判决不可用(无原生视频评委/合成降级)")
    print(f"  episode: {res.episode_id}(长期记忆 {episode_memory.path}) ")
    print(f"  台账: {run_dir / 'storyboard.json'}")
    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
