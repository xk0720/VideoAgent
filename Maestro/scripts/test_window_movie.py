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
    ap.add_argument("--mode", default="window",
                    choices=["window", "cinegraph"],
                    help="window=像素连续派(钉帧/矢量/桥);cinegraph="
                         "ViMax 式构图一致派(机位树/首帧派生/选图/"
                         "多视图注册表,2026-08-07)")
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
    ap.add_argument("--repair-severity", type=float, default=0.0,
                    help="最坏缺陷 severity 低于此值就不修(0=关闭,荐 0.6:"
                         "VLM 惯出的 0.5 级小项不再触发花钱修复)")
    ap.add_argument("--screenplay", default="",
                    help="用户自带剧本路径:.txt 纯文本;.json 走固定契约 "
                         "{content, role:{角色名→图片路径}}(智能识别,"
                         "角色图预填官方肖像,路径救援见 script_input)")
    ap.add_argument("--repair-mode", default="full",
                    choices=["full", "consistency", "classic"],
                    help="修复两段控制:consistency=只修一致性(菜单仅"
                         "转场/接受,其他 metrics 不修,accept 不被越权"
                         "改判);classic=只旧策略(regenerate/segment,"
                         "无转场);full=两段全开")
    ap.add_argument("--no-review", action="store_true",
                    help="M2:关闭评审/修复总开关(首选候选直接收货,"
                         "不评审不修复;episode 蒸馏同时停用)")
    ap.add_argument("--rl-group", type=int, default=0,
                    help="RL 组采样(semi-online GRPO):每镜同 state 采 "
                         "K 个条件决策各生成一候选,评审择优推进主干;"
                         "组记录落 run 目录 rl_steps.jsonl")
    ap.add_argument("--rl-temperature", type=float, default=0.9)
    ap.add_argument("--no-junction-agent", action="store_true",
                    help="缝合师回滚开关(2026-08-09):关掉 LLM 组稿,"
                         "派生双镜描述回确定性模板装配")
    ap.add_argument("--pin-gate-mad", type=float, default=0.0,
                    help="§G 钉帧完整性闸门(0=关闭,荐 8.0):钉了开场的"
                         "路线生成完立即测开场撕裂度(接点/帧0→1/帧1→2 "
                         "取最大;测量零成本),超阈当场重掷一次(重掷花"
                         "一次生成费,换掉更贵的评审→修复弯路)")
    ap.add_argument("--with-physics-measure", action="store_true",
                    help="启用 CoTracker+GroundingDINO 测量 critic(需 GPU)")
    ap.add_argument("--baseline-anchor", action="store_true",
                    help="开工先按用户指令一次直出锚点视频,收尾与成片盲测"
                         "对比(附加物,不影响正流程)")
    ap.add_argument("--anchor-duration", type=int, default=None,
                    help="锚点视频时长秒数(默认不传 = API 默认)")
    ap.add_argument("--transitions", action="store_true",
                    help="启用 add_transition 转场桥修复工具(2026-08-04 "
                         "用户令:默认关闭)")
    ap.add_argument("--bgm", action="store_true",
                    help="启用背景音乐(2026-08-05 用户令:默认取消)")
    ap.add_argument("--no-bgm", action="store_true",
                    help="关闭背景音乐(§F 配乐不跑);--audio 开着时仍保"
                         "对白原生音+口型同步——'人物说话有声、无 BGM'")
    ap.add_argument("--audio", action="store_true",
                    help="音频线:对白原生音频(口型)+ scene 级配乐 + 混音")
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

    # 剧本输入智能识别:.json 走固定契约(content + role 角色图绑定)
    _screenplay_text, _given_chars = None, None
    if args.screenplay:
        sp = Path(args.screenplay)
        if sp.suffix.lower() == ".json":
            from maestro.pipeline.script_input import parse_script_json
            parsed = parse_script_json(sp)
            _screenplay_text = parsed["content"]
            _given_chars = parsed["roles"]
            print(f"剧本 JSON: {len(_screenplay_text)} 字 | 角色绑定 "
                  f"{len(_given_chars)} 位")
            for n in parsed["notes"]:
                print(f"  ⚠ {n}")
        else:
            _screenplay_text = sp.read_text(encoding="utf-8")

    llm = build_llm(models_cfg.get("llm"))
    mllm = build_mllm(models_cfg.get("mllm"))
    # ViMax 分工(2026-08-06 用户令):六个 agent 各持【独立实例】,
    # 绝不共享对象;models.crew.<role> 可逐 agent 换不同型号,缺省
    # 沿用对应默认配置(LLM 角色 ← models.llm;VLM 角色 ← models.mllm)。
    _crew_cfg = dict(models_cfg.get("crew") or {})

    def _agent_llm(role: str):
        return build_llm(_crew_cfg.get(role) or models_cfg.get("llm"))

    def _agent_mllm(role: str):
        return build_mllm(_crew_cfg.get(role) or models_cfg.get("mllm"))

    llm_screenwriter = _agent_llm("screenwriter")     # 剧本扩写 LLM agent
    llm_scene_writer = _agent_llm("scene_writer")     # 分镜 LLM agent
    llm_video_brain = _agent_llm("video_brain")       # 视频 brain LLM agent
    llm_enhancer = _agent_llm("prompt_enhancer")      # 润色 LLM agent
    llm_stitcher = _agent_llm("junction_stitcher")    # 缝合师 LLM agent
                                                      # (2026-08-10 裁决a:
                                                      # 独立槽位,RL 冻结)
    mllm_reviewer = _agent_mllm("reviewer")           # 评审 VLM agent
    mllm_verifier = _agent_mllm("verifier")           # 验收 VLM agent
    # 任务 0:每次 WaveSpeed 调用(模型名+参数)落盘可核对
    vg_cfg = dict(models_cfg.get("video_gen") or {})
    vg_cfg.setdefault("call_log", str(run_dir / "wavespeed_calls.jsonl"))
    video_gen = build_video_gen(vg_cfg)
    # debug:brain 每次决策的原始输出 + 解析结果(核对策略→模型映射用)
    from maestro.logging_utils import set_brain_log
    from maestro.memory.character_library import CharacterLibrary
    set_brain_log(run_dir / "brain_calls.jsonl")
    print(f"配置: {args.config}")
    print("  agent 编成(ViMax 分工,独立实例):")
    print(f"    剧本扩写 LLM  = {getattr(llm_screenwriter, 'model', '?')}")
    print(f"    分镜     LLM  = {getattr(llm_scene_writer, 'model', '?')}")
    print(f"    视频brain LLM = {getattr(llm_video_brain, 'model', '?')}"
          "  (image_plan+window_generation+orchestrator)")
    print(f"    润色     LLM  = {getattr(llm_enhancer, 'model', '?')}")
    print(f"    评审     VLM  = {getattr(mllm_reviewer, 'model', '?')}")
    print(f"    验收     VLM  = {getattr(mllm_verifier, 'model', '?')}")
    print(f"    视频生成      = {getattr(video_gen, 'model_id', '?')}")
    print(f"  调用日志: {vg_cfg['call_log']}")
    print(f"  brain 决策日志: {run_dir / 'brain_calls.jsonl'}")

    # ── skill 自检(百分之百确定技能文件装载;每次 brain 调用另在
    # brain_calls.jsonl 记 skill_chars 作逐次证据)────────────────────────
    _section("skill 自检(进 brain prompt 的技能文件)")
    from maestro.skills.loader import load_skill
    _needed = ["scene_write", "image_plan", "window_generation", "orchestrator"]
    if args.prompt_enhancer:
        _needed.append("prompt_enhancer")
    _missing = []
    for _n in _needed:
        try:
            _sk = load_skill(_n)
        except Exception:
            _sk = None
        _ok = bool(_sk and _sk["body"].strip())
        print(f"  {'✓' if _ok else '✗'} {_n:20s} "
              f"{len(_sk['body']) if _ok else 0:6d} chars"
              + ("" if _ok else "   ← 缺失!该 brain 只有内联短指令"))
        if not _ok:
            _missing.append(_n)
    if _missing:
        print(f"  ⚠⚠⚠ {len(_missing)} 个技能没装载: {_missing} — 决策质量会差,"
              "先修再跑!")
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
        # caption 留给 Q-D 打标链:用户描述 > VLM 看中间帧 > 文件名兜底
        # (2026-07-16 裁决:用户大概率只给一个路径,必须兼容)
        asset_memory.video_shots[f"vid{i}"] = Shot(
            shot_id=f"vid{i}", source_video=str(vp.resolve()),
            start_time=0.0, end_time=dur, caption=desc)
    retrieval = RetrievalTool(asset_memory) if (
        asset_memory.identity_anchors or asset_memory.video_shots) else None

    # Q-D 素材打标链:用户描述 > VLM caption > 文件名(真 VLM 才回填)
    from maestro.pipeline.window_loop import ensure_asset_descriptions
    n_cap = ensure_asset_descriptions(asset_memory, mllm,
                                      cache_dir=run_dir / "asset_labels")
    if asset_memory.identity_anchors or asset_memory.video_shots:
        print(f"  素材库: {asset_memory.summarize()}"
              + (f";VLM 补标 {n_cap} 条" if n_cap else ""))

    critics = [SemanticCritic(mllm=mllm_reviewer),
               PhysicsCritic(mllm=mllm_reviewer)]
    if args.with_physics_measure:
        from maestro.critics.physics_consistency import PhysicsConsistencyCritic
        from maestro.physics.tracks import build_track_extractor
        critics.append(PhysicsConsistencyCritic(extractor=build_track_extractor({
            "name": "cotracker", "device": args.device,
            "detector": {"name": "groundingdino", "device": args.device}})))

    generator = GeneratorAgent(video_gen=video_gen)
    orchestrator = OrchestratorAgent(
        llm=llm_video_brain, generator=generator, refiner=RefinerAgent(),
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
        prompt_enhancer = PromptEnhancerAgent(llm=llm_enhancer)
    if args.mode == "cinegraph":
        from maestro.cinegraph import generate_movie_cinegraph
        res = generate_movie_cinegraph(
            args.prompt,
            cache_dir=run_dir, llm=llm,
            crew={"screenwriter": llm_screenwriter,
                  "scene_writer": llm_scene_writer,
                  "video_brain": llm_video_brain},
            mllm=mllm_reviewer, video_gen=video_gen,
            image_edit=build_image_edit({"name": "wavespeed"}),
            board=ReviewBoard(critics=critics, metric_tool=MetricTool()),
            asset_memory=asset_memory,
            screenwriter=ScreenwriterAgent(llm=llm_screenwriter,
                                           config=cfg.get("plan") or {}),
            screenplay=_screenplay_text,
            given_characters=_given_chars,
            character_library=CharacterLibrary(REPO_ROOT / "data"
                                               / "character_library"),
            prompt_enhancer=prompt_enhancer,
            plan_cfg=cfg.get("plan") or {},
            max_turns=args.max_turns,
            enable_audio=args.audio)
        _section("cinegraph 决策流水")
        for d in res.decisions:
            print(f"  [{d.get('stage', '—'):11s}] "
                  f"{str(d.get('label', '—')):<10s} → "
                  f"{str(d.get('strategy', d.get('via', '—'))):20s} "
                  f"{str(d.get('reason', ''))[:60]}")
        print("成片:", getattr(res, "movie_path", None))
        return
    res = generate_movie_windowed(
        args.prompt,
        crew={"screenwriter": llm_screenwriter,
              "scene_writer": llm_scene_writer,
              "video_brain": llm_video_brain,
              "junction_stitcher": llm_stitcher},
        board=ReviewBoard(critics=critics, metric_tool=MetricTool()),
        generator=generator, refiner=RefinerAgent(), verifier=VerifierAgent(judge=mllm_verifier),
        orchestrator=orchestrator, cache_dir=run_dir,
        # 空间圣经(2026-08-10 事故:漏传 → 三向视图与清场回流全部
        # 静默跳过)—— 窗口管线的图像编辑端
        image_edit=build_image_edit({"name": "wavespeed"}),
        asset_memory=asset_memory, retrieval=retrieval,
        screenwriter=ScreenwriterAgent(llm=llm_screenwriter,
                                       config=cfg.get("plan") or {}),
        director=DirectorAgent(llm=llm),
        tournament=Tournament(judge=mllm),
        skill_library=orchestrator.skill_library,
        episode_memory=episode_memory, llm=llm,
        n_candidates=args.n_candidates, max_turns=args.max_turns,
        window_tail_s=args.tail_seconds,
        patience=args.patience, quality_bar=args.quality_bar,
        repair_severity=args.repair_severity,
        pin_gate_mad=args.pin_gate_mad,
        screenplay=_screenplay_text,
        given_characters=_given_chars,
        enable_review=(not args.no_review),
        use_junction_agent=(not args.no_junction_agent),
        rl_group=args.rl_group,
        rl_temperature=args.rl_temperature,
        repair_mode=args.repair_mode,
        enable_bgm=args.bgm,
        enable_transitions=args.transitions,
        baseline_anchor=args.baseline_anchor,
        baseline_anchor_duration=args.anchor_duration,
        prompt_enhancer=prompt_enhancer,
        enable_audio=args.audio,
        character_library=CharacterLibrary(REPO_ROOT / "data"
                                           / "character_library"),
        mllm=mllm)   # 需求 ②:接点实况 VLM(gemini API / qwen-local 本地)

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
        print(f"  基线锚点: {a['path']}(route={a['route']}, via={a['via']}"
              f") —— 自己看片对比,不做机器裁决")
    print(f"  episode: {res.episode_id}(长期记忆 {episode_memory.path}) ")
    print(f"  台账: {run_dir / 'storyboard.json'}")
    print(f"\n📂 本次所有产物在: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
