"""RL agent loop —— 生产 generate_movie_windowed 的逐段移植(2026-08-19
用户令【训练=生产完全同构 + rl/ 自包含】)。

与生产的差异仅限用户明令三点:
  ① 每镜 generation-condition 同 state 采 K 个决策(v0 默认温度,其余
     rl_temperature),K 个候选各自走完整出门链并生成;
  ② 主干 = rl/reward skill 判官(文本判官 + action/physics/camera 三路
     排名 + 一致性对照)合成 reward 的 argmax —— 评审板/锦标赛/修复
     循环不进 RL(既定裁决:--max-turns 0、skill 判官换评审);
  ③ enhancer / episode 记忆 / BGM / 转场 / baseline anchor 关(生产 RL
     农场跑法本就如此)。

其余与生产同构(代码 = rl/env/window_core.py 的逐字移植件):
§A0 剧本 → §A1 角色提取 → §A 分镜(坏回复/语言/对白覆盖全闸)→
§A' 官方肖像 → §A2 背景板 + 资产保证闸 → 空间圣经(环视多视图+图注)
→ 逐镜三叉分诊(derive 派生缝合 / cut / continue 片尾报告)→ §B'
image plan(策略单采样,与生产一致)→ 条件菜单+菜单锁 → 槽位清单 →
K 组采样 → 出门链(剥标记/契约清洗/引用闸/正典/对白音频/名字终换)
→ 条件执行(同策略重试一次→降级留痕)→ 判官择主干 → rl_steps 组记录
→ 空间圣经实拍回流 → §E 拼接。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent.parent
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

import env.junction_stitcher as _js                           # noqa: E402
import env.window_core as W                                   # noqa: E402
from env.junction_stitcher import JunctionStitcherAgent       # noqa: E402
from env.language import set_output_lang                      # noqa: E402
from env.skills import skill_body                             # noqa: E402
from env.logging_utils import brain_log, get_logger, set_brain_log  # noqa: E402
from env.space_bible import (build_space_views, pick_space_view,    # noqa: E402
                             washed_frame_upgrade)
from env.storyboard import StoryboardMemory                   # noqa: E402

log = get_logger("maestro.rl_env")

# 组内并发上限(2026-08-20 用户令"轴 A"):三段网络等待各自的并发度。
# 可灵账号的并发配额未知 —— 环境变量可随时收紧,不用改代码。
_GEN_CONCURRENCY = int(os.environ.get("RL_GEN_CONCURRENCY", "4"))
_JUDGE_CONCURRENCY = int(os.environ.get("RL_JUDGE_CONCURRENCY", "6"))
_POLICY_CONCURRENCY = int(os.environ.get("RL_POLICY_CONCURRENCY", "4"))


def _run_concurrent(fn, items: list, workers: int) -> list:
    """并发映射,结果【按输入顺序】返回(顺序即候选下标,绝不能乱)。
    workers<=1 或只有一项 → 直接串行(测试/降级路径行为完全一致)。
    单项抛异常 → 原样上抛(调用方的 try 阶梯照旧生效)。"""
    if workers <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


@dataclass
class ShotSpec:
    """window_core 用到的 spec 三元组(生产 types.ShotSpec 的使用面)。"""
    shot_idx: int
    duration: float | None
    prompt: str


def _fallback_outline(text: str, n_shots: int = 3,
                      max_shots: int = 12) -> list[str]:
    """确定性拆条兜底(生产 ScreenwriterAgent.run 同法:分号切子句,
    循环填充,"Shot i:" 前缀)。scene_write 全败才会走到这里。"""
    n = max(1, min(n_shots, max_shots))
    clauses = [c.strip() for c in str(text or "").replace("；", ";")
               .split(";") if c.strip()]
    return [f"Shot {i + 1}: {clauses[i % len(clauses)] if clauses else text}"
            for i in range(n)]


def _brain_index(run_dir: Path) -> dict:
    """decision_id → brain_calls 记录(raw 的单一来源:生产 _decide 不
    回传 raw,原始输出只落 brain_calls.jsonl —— 组记录按 id 回取)。"""
    idx: dict = {}
    p = run_dir / "brain_calls.jsonl"
    if not p.exists():
        return idx
    for line in p.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        did = d.get("decision_id")
        if did and d.get("stage") == "window/generation-condition":
            idx[did] = d
    return idx


def _concat(clips: list, out_path: Path) -> Path:
    """§E 拼接(生产 VideoConcatTool 同律:ffmpeg 必备、假产物绝不
    出门;concat demuxer -c copy)。"""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("video_concat: ffmpeg is REQUIRED")
    bad = [str(p) for p in clips
           if not Path(p).exists() or Path(p).stat().st_size <= 1024]
    if bad:
        raise RuntimeError(f"video_concat: bad inputs {bad[:3]}")
    import subprocess
    lst = out_path.parent / "concat_list.txt"
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in clips))
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                        "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(out_path)], capture_output=True, timeout=600)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"concat failed: {r.stderr.decode()[:200]}")
    return out_path


# ── 判官(与收集器时代同一套 skill;评审在采样端)──────────────────
def build_judges(models_cfg: dict, log_path: Path):
    from reward.judges import (ConsistencyChecker, JudgeLog,
                               OpenAICompatChat, TextJudge, VideoRanker)
    MAAS = ("https://ws-ox5q19lbmn2u1drg.cn-beijing.maas.aliyuncs.com"
            "/compatible-mode/v1")
    DAS = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    IDE = "https://idealab-external.alibaba-inc.com/api/openai/v1"
    _BY_NAME = {"qwen-maas": (MAAS, "DASHSCOPE_API_KEY", None),
                "qwen": (DAS, "QWEN_API_KEY", "DASHSCOPE_API_KEY"),
                "qwen-vl": (DAS, "QWEN_API_KEY", "DASHSCOPE_API_KEY"),
                "idealab": (IDE, "IDEALAB_API_KEY", None),
                "idealab-gemini": (IDE, "IDEALAB_API_KEY", None)}

    def _client(spec, default_model):
        name = (spec or {}).get("name", "qwen")
        base, ev1, ev2 = _BY_NAME.get(
            name, _BY_NAME.get(name.split("-")[0], _BY_NAME["qwen"]))
        base = (spec or {}).get("base_url") or base
        key = ((spec or {}).get("api_key") or os.getenv(ev1)
               or (os.getenv(ev2) if ev2 else None))
        if not key:
            raise RuntimeError(f"judge({name}) 缺 key:设 {ev1}")
        return OpenAICompatChat(base, (spec or {}).get(
            "model", default_model), key,
            extra_body=(spec or {}).get("extra_body"))

    jlog = JudgeLog(log_path)
    return {"text": TextJudge(_client(models_cfg.get("llm"),
                                      "qwen-max"), log=jlog),
            "ranker": VideoRanker(_client(models_cfg.get("mllm"),
                                          "gemini-3.1-pro-preview"),
                                  log=jlog),
            "consistency": ConsistencyChecker(
                _client(models_cfg.get("mllm"), "gemini-3.1-pro-preview"),
                log=jlog)}


def judge_group(judges, context: dict, entry, storyboard,
                variants: list[dict], conds: list[dict], videos: list,
                run: str) -> tuple[list[dict], dict]:
    """组内双重评审(用户 reward v3 设计原样):文本判官逐候选(判词
    = brain 原始 video_prompt;衔接连续性仅 continue / 带空间视图的
    derive 可判)+ 三路排名(一组一调用)+ 一致性对照(参照 = 出场者
    肖像 + junction 空间视图)→ compose_rewards。分量失败剔除归一化;
    全失败 = reward 只剩 format 分(诚实,不编数)。"""
    from reward.judges import compose_rewards
    n = len(variants)
    fmt = [1.0 if v.get("via") == "llm" else 0.0 for v in variants]
    jm = getattr(entry, "junction_meta", None) or {}
    kind = jm.get("kind")
    continuity = bool(kind == "continue"
                      or (kind == "derive" and jm.get("space_view")))
    details: list[dict] = [{} for _ in range(n)]

    # 文本判官:逐候选独立 → 并发(2026-08-20 轴 A;结果按下标回填)
    def _text_one(iv):
        i, v = iv
        try:
            case = {
                "shot_script": entry.description,
                "cast_canon": context.get("cast") or {},
                "story_so_far": [{"label": r.get("label"),
                                  "description": r.get("description")}
                                 for r in (context.get("storyboard")
                                           or [])],
                "prev_end_state": (context.get("prev_shot")
                                   or {}).get("end_state", ""),
                "junction": {"kind": kind,
                             "continuity_applicable": continuity,
                             "handoff_required": kind == "derive"},
                "slots": (context.get("slots_by_strategy")
                          or {}).get(v.get("strategy"), []),
                "candidate_prompt": v.get("video_prompt", ""),
            }
            score, detail = judges["text"].score(
                case, tag={"run": run, "label": entry.label,
                           "candidate": i,
                           "decision_id": v.get("decision_id")})
            return score, detail
        except Exception as exc:
            print(f"[judge] text failed ({str(exc)[:120]})", flush=True)
            return None, {"error": str(exc)[:120]}

    _text = _run_concurrent(_text_one, list(enumerate(variants)),
                            _JUDGE_CONCURRENCY)
    text_scores = [t[0] for t in _text]
    for i, t in enumerate(_text):
        details[i]["judge_text"] = t[1]

    video_parts: dict = {"action": None, "physics": None,
                         "camera": None, "consistency": None}
    judge_video: dict = {}
    ok = [v for v in videos if v and Path(str(v)).exists()]
    if len(ok) == n and n >= 2:
        rank_ctx = {"shot_script": entry.description,
                    "camera_facing": getattr(entry, "camera_facing", ""),
                    "cast_canon": context.get("cast") or {}}
        # 三个排序维度互相独立 → 并发(每次调用自带整组视频)
        def _rank_one(dim):
            try:
                res = judges["ranker"].rank(
                    dim, rank_ctx, [str(v) for v in videos],
                    tag={"run": run, "label": entry.label})
                return dim, res, None
            except Exception as exc:
                print(f"[judge] rank {dim} failed ({str(exc)[:120]})",
                      flush=True)
                return dim, None, str(exc)[:120]

        for dim, res, err in _run_concurrent(
                _rank_one, ["action", "physics", "camera"],
                _JUDGE_CONCURRENCY):
            if err is None:
                video_parts[dim] = res["points"]
                judge_video[dim] = {"evidence": res.get("evidence"),
                                    "order": res.get("order")}
            else:
                judge_video[dim] = {"error": err}
        refs = []
        portraits = storyboard.portraits or {}
        for name in (context.get("cast_in_shot") or []):
            pth = portraits.get(name)
            if pth and Path(pth).exists():
                refs.append({"kind": f"portrait:{name}", "path": pth,
                             "note": (context.get("cast")
                                      or {}).get(name, "")[:120]})
        sv = jm.get("space_view") or {}
        if isinstance(sv, dict) and sv.get("path") \
                and Path(sv["path"]).exists():
            refs.append({"kind": "space_view", "path": sv["path"],
                         "note": (sv.get("caption") or "")[:200]})
        if refs:
            # 一致性对照:逐候选独立 → 并发
            def _cons_one(iv):
                i, v = iv
                try:
                    sc, detail = judges["consistency"].score(
                        str(v), refs,
                        {"shot_script": entry.description},
                        tag={"run": run, "label": entry.label,
                             "candidate": i})
                    return i, sc, detail
                except Exception as exc:
                    print(f"[judge] consistency failed "
                          f"({str(exc)[:120]})", flush=True)
                    return i, None, {"error": str(exc)[:120]}

            cons: dict = {}
            for i, sc, detail in _run_concurrent(
                    _cons_one, list(enumerate(videos)),
                    _JUDGE_CONCURRENCY):
                details[i]["judge_consistency"] = detail
                if sc is not None:
                    cons[i] = sc
            video_parts["consistency"] = cons or None
    composed = compose_rewards(fmt, text_scores, video_parts, n)
    for i in range(n):
        details[i].update(composed[i])
    return details, judge_video


# ── 主 driver(生产 generate_movie_windowed 的移植)──────────────────
def run_episode(*, task_text: str = "", screenplay: str | None = None,
                run_dir: Path, frozen_llm, policy, video_gen, image_edit,
                mllm, judges, group: int = 4, rl_temperature: float = 0.9,
                fps: int = 8, window_tail_s: float = 2.0,
                max_shots: int = 12, enable_audio: bool = False,
                use_junction_agent: bool = True) -> dict:
    """一条轨迹。task_text = idea/一句话;screenplay = 用户剧本原文
    (给了就跳过 §A0,与生产同)。"""
    # ── 硬预检(生产同款)────────────────────────────────────────
    _missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if _missing:
        raise RuntimeError(
            f"windowed pipeline PREFLIGHT failed: {_missing} not found "
            f"on PATH — install ffmpeg first.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_brain_log(run_dir / "brain_calls.jsonl")
    asset_memory = None                      # RL 无用户素材(生产同构:
    #                                          全部素材函数 None 安全)
    decisions: list[dict] = []
    guidance = {"replay_hints": [], "avoid": [], "n_episodes_matched": 0}
    replay_plan: dict = {}
    replay_cond: dict = {}

    llm_screenwriter = llm_scene_writer = llm = frozen_llm
    llm_video_brain = policy
    # ── 缝合师(2026-08-20 用户裁决:独立 agent → 冻结底座)────────
    # 独立性依据:自带上下文(上镜 end_state / 片尾报告 / 本镜 opening
    # / 槽位表,共 5 个字段 —— 看不到台账、菜单、决策历史)、自带四级
    # 校验、自带降级路径(判死退模板);产物只喂派生视频的 prompt,
    # 永不进 RL 训练目标。所以它钉在 qwen3.8-max 上,不跟被训策略漂移
    # —— 这正是生产 crew.junction_stitcher 槽位当初留下的意图。
    #
    # 技能手册预置:移植件的 _skill_body() 走生产的相对导入,在 rl/ 下
    # 必抛异常并被 except 吞成【空手册】(实测 0 字符 vs 生产 3179)——
    # 缝合师一直在裸奔。从外部把缓存喂上,移植件因此保持逐字不动
    # (整文件同构锁在岗)。
    _js._SKILL_CACHE.setdefault("junction_stitch",
                                skill_body("junction_stitch"))
    if use_junction_agent and not _js._SKILL_CACHE["junction_stitch"]:
        log.warning("junction_stitch 技能手册为空 —— 缝合师将裸奔")
    junction_stitcher = (JunctionStitcherAgent(llm=frozen_llm)
                         if use_junction_agent else None)

    # ── §A0 剧本 + 语言 + 声词词典 ───────────────────────────────
    asset_catalog0 = W._media_catalog(asset_memory)
    screenplay_text, sp_via = W._write_screenplay(
        llm_screenwriter, task_text, screenplay, asset_catalog0)
    decisions.append({"stage": "screenplay", "via": sp_via,
                      "chars": len(screenplay_text)})
    prompt_lang = W._prompt_lang(screenplay_text or task_text)
    set_output_lang(prompt_lang)
    W.set_run_sound_lexicon(screenplay_text or task_text)

    # ── §A1 角色提取 ────────────────────────────────────────────
    cast_canon, ce_via = (W._extract_characters(
        llm, screenplay_text, given=None, prompt_language=prompt_lang)
        if sp_via != "idea_passthrough" else ({}, "skipped"))
    decisions.append({"stage": "character_extract", "via": ce_via,
                      "characters": sorted(cast_canon)})

    # ── §A 分镜 ─────────────────────────────────────────────────
    outline, shot_durations, shot_end_states, script_meta, outline_via = \
        W._write_outline(
            llm_scene_writer, screenplay_text, asset_catalog0,
            episode_guidance=guidance, max_shots=max_shots,
            fallback_fn=lambda: _fallback_outline(
                screenplay_text or task_text, max_shots=max_shots),
            cast_canon=cast_canon, prompt_language=prompt_lang)
    decisions.append({"stage": "playwriting", "label": "outline",
                      "strategy": f"{len(outline)} shots",
                      "via": outline_via})
    specs = [ShotSpec(shot_idx=i,
                      duration=(float(d) if d is not None else None),
                      prompt=W._strip_markers(o))
             for i, (o, d) in enumerate(zip(outline, shot_durations))]
    storyboard = StoryboardMemory.from_outline(
        outline, path=run_dir / "storyboard.json")
    for i_, (entry_, end_) in enumerate(zip(storyboard.entries,
                                            shot_end_states)):
        entry_.end_state = end_
        vars_ = script_meta.get("variations") or []
        opens_ = script_meta.get("opening_frames") or []
        dlgs_ = script_meta.get("dialogues") or []
        spks_ = script_meta.get("dialogue_speakers") or []
        bgs_ = script_meta.get("bgs") or []
        entry_.variation = vars_[i_] if i_ < len(vars_) else ""
        entry_.opening_frame = opens_[i_] if i_ < len(opens_) else ""
        entry_.dialogue_speaker = spks_[i_] if i_ < len(spks_) else ""
        entry_.bg_id = bgs_[i_] if i_ < len(bgs_) else ""
        entry_.dialogue = dlgs_[i_] if i_ < len(dlgs_) else ""
        _fcs = script_meta.get("camera_facings") or []
        entry_.camera_facing = _fcs[i_] if i_ < len(_fcs) else ""
    storyboard.cast = dict(script_meta.get("cast", {}))
    storyboard.music_plan = dict(script_meta.get("music_plan", {}))
    storyboard.setting = str(script_meta.get("setting", ""))

    # ── §A' 官方肖像 + §A2 背景板 + 资产保证闸 + 空间圣经 ─────────
    decisions.extend(W._ensure_cast_portraits(
        storyboard, asset_memory, video_gen, run_dir,
        library=None, llm=llm))
    _caps0 = (video_gen.capabilities() if video_gen is not None else set())
    _bg_prompts: dict = {}
    _need_keys: list = []
    if video_gen is not None and "t2i" in _caps0 \
            and hasattr(video_gen, "text_to_image"):
        adir = run_dir / "anchors"
        _bg_keys: list = []
        for e in storyboard.entries:
            k = (getattr(e, "bg_id", "") or f"scene_{e.scene_idx}")
            if k not in _bg_keys:
                _bg_keys.append(k)
        _need_keys = [k for k in _bg_keys
                      if k not in (storyboard.backgrounds or {})]
        _bg_prompts, _bg_via = (W._write_bg_prompts(llm, storyboard,
                                                    _need_keys)
                                if _need_keys else ({}, "none"))
        if _need_keys:
            decisions.append({"stage": "scene_image", "via": _bg_via,
                              "backgrounds": sorted(_need_keys)})
        from env.cine import _spaced_retry
        for _bk in _need_keys:
            aprompt = _bg_prompts[_bk]
            adir.mkdir(parents=True, exist_ok=True)
            got = _spaced_retry(
                lambda: video_gen.text_to_image(aprompt,
                                                adir / f"bg_{_bk}.png"),
                tag=f"background asset {_bk}")
            storyboard.backgrounds[_bk] = {"path": str(got), "src": "t2i"}
            decisions.append({"stage": "background_asset", "bg": _bk,
                              "via": "t2i", "path": str(got)})
        storyboard._save()
        _missing_assets: list = []
        for _e in storyboard.entries:
            for _n in W._cast_in_shot(_e.description, storyboard.cast):
                _pp = (storyboard.portraits or {}).get(_n)
                if not _pp or not Path(_pp).exists():
                    _missing_assets.append(
                        f"shot {_e.shot_idx}: character {_n!r} 无肖像")
            _k = (getattr(_e, "bg_id", "") or f"scene_{_e.scene_idx}")
            _b = (storyboard.backgrounds or {}).get(_k) or {}
            if not _b.get("path") or not Path(str(_b["path"])).exists():
                _missing_assets.append(
                    f"shot {_e.shot_idx}: background {_k} 无场景板")
        if _missing_assets:
            raise RuntimeError("asset guarantee failed — "
                               + "; ".join(_missing_assets))
        decisions.extend(build_space_views(
            storyboard, image_edit, mllm, run_dir / "spaces",
            bg_descs=_bg_prompts if _need_keys else None,
            video_gen=video_gen))
    log.info("window: playwriting done via=%s — %s",
             outline_via, storyboard.summary())

    # ── §B' Image Plan(策略单采样决策,与生产一致)──────────────
    kf_dir = run_dir / "keyframes"
    asset_catalog = W._media_catalog(asset_memory)
    portrait_paths: set = set()
    for _pp in (storyboard.portraits or {}).values():
        try:
            portrait_paths.add(str(Path(_pp).resolve()))
        except Exception:
            portrait_paths.add(str(_pp))
    for entry, spec in zip(storyboard.entries, specs):
        menu = W._image_plan_menu(video_gen, asset_memory)
        d = W._decide(
            llm_video_brain, "image-plan", menu,
            {"shot": entry.to_brain_line(),
             "prompt_language": prompt_lang,
             "cast": storyboard.cast, "setting": storyboard.setting,
             "storyboard": storyboard.to_brain_json(),
             "asset_catalog": asset_catalog,
             "episode_guidance": guidance},
            replay_hint=replay_plan.get(entry.label),
            priority=W._PLAN_PRIORITY)
        decisions.append({"stage": "image_plan", "label": entry.label,
                          **d})
        _shot_cast_b = W._cast_in_shot(entry.description, storyboard.cast)
        plan_final, images, degraded_from = W._execute_image_plan(
            d, entry, video_gen, asset_memory, None, kf_dir,
            cast=storyboard.cast, portrait_paths=portrait_paths,
            has_portrait_cast=any(n in (storyboard.portraits or {})
                                  for n in _shot_cast_b))
        storyboard.set_image_plan(entry.shot_idx, plan_final, images,
                                  degraded_from=degraded_from)

    W.set_run_ambience(storyboard.setting)
    source_videos = W._prepared_source_videos(asset_memory,
                                              run_dir / "asset_labels")

    # ── §C 逐镜大循环 ────────────────────────────────────────────
    n_groups = 0
    aborted = False
    while True:
        entry = storyboard.next_pending()
        if entry is None:
            break
        spec = specs[entry.shot_idx]
        prev = storyboard.prev_generated(entry.shot_idx)
        shot_dir = run_dir / f"shot{entry.shot_idx:03d}"

        # 三叉分诊(生产原文:人物变→derive(退 cut);同人异景→cut;
        # 同人同景→derive(退 continue))
        _junction_kind = None
        _route_reason = ""
        _open_cast: list = []
        _derive_fallback = "cut"
        _bg_prev = _bg_cur = None
        if prev is not None and prev.video_path and entry.shot_idx > 0:
            _bg_prev = (getattr(prev, "bg_id", "")
                        or f"scene_{prev.scene_idx}")
            _bg_cur = (getattr(entry, "bg_id", "")
                       or f"scene_{entry.scene_idx}")
            _same_cast, _cast_reason, _open_cast = W._judge_junction_cast(
                llm_video_brain, prev, entry, storyboard.cast,
                storyboard.portraits, prompt_lang)
            if not _same_cast:
                _junction_kind = "derive"
                _derive_fallback = "cut"
            elif _bg_prev != _bg_cur:
                _junction_kind = "cut"
            else:
                _junction_kind = "derive"
                _derive_fallback = "continue"
            _route_reason = f"bg {_bg_prev}→{_bg_cur}; {_cast_reason}"
            log.info("window: %s junction → %s (%s)", entry.label,
                     _junction_kind, _route_reason[:200])

        junction_actual = None
        if _junction_kind == "derive":
            junction_actual = W._junction_state(
                mllm, prev, shot_dir, tail_s=window_tail_s,
                portraits=storyboard.portraits)
            _bgrec = (storyboard.backgrounds or {}).get(_bg_cur) or {}
            _sview = None
            if _bg_prev == _bg_cur:
                _sview = pick_space_view(
                    llm_video_brain, storyboard, _bg_cur,
                    (getattr(entry, "camera_facing", "") or
                     W._strip_markers(" ".join(
                         t for t in (entry.opening_frame,
                                     entry.description) if t))))
            _derived = W._derive_junction_frame(
                video_gen, mllm, llm_video_brain, prev, prev, entry,
                _open_cast, storyboard.cast, storyboard.portraits,
                (_bgrec.get("path") if _bg_prev != _bg_cur else None),
                shot_dir, prompt_lang,
                stitcher=junction_stitcher,
                tail_report=W._parse_tail_report(junction_actual),
                space_view=_sview)
            entry.junction_meta = {
                **(getattr(entry, "junction_meta", None) or {}),
                "kind": "derive", "route_reason": _route_reason[:300],
                "space_view": ({"view": _sview["view"],
                                "path": _sview["path"],
                                "caption": _sview.get("caption",
                                                      "")[:200]}
                               if _sview else None),
                "derived_frame": (str(_derived) if _derived else None),
            }
            if _derived is not None:
                entry.images = list(entry.images or []) + [{
                    "path": str(_derived), "role": "reference",
                    "source": "pin_frame",
                    "description": "derived junction first frame"}]
                decisions.append({"stage": "junction",
                                  "label": entry.label,
                                  "strategy": "derive",
                                  "reason": _route_reason[:160]})
            else:
                _junction_kind = _derive_fallback
                entry.junction_meta["fallback_to"] = _derive_fallback
                decisions.append({"stage": "junction",
                                  "label": entry.label,
                                  "strategy":
                                      f"derive→{_derive_fallback}",
                                  "reason": "派生失败/两拒 — 降级"
                                            f"{_derive_fallback}"})
        if _junction_kind == "derive":
            junction_ctx = {
                "junction_kind": "derive",
                "junction_note": (
                    "缝合策略:本镜首帧已由派生帧给定(清单末位 pin_frame "
                    "行,其提及由执行器负责,你绝不引用该槽位);按本镜"
                    "剧本全新书写画面与动作,禁止书写任何承接上一镜的"
                    "连续性语句。" if prompt_lang == "zh" else
                    "STITCH strategy: the opening frame is given by a "
                    "derived frame (the manifest's last pin_frame row; the "
                    "executor owns its mention — never reference that slot "
                    "yourself). Write the shot fresh from its own script; "
                    "do NOT write any continuity with the previous shot."),
                "required_end_state": entry.end_state or None,
            }
        elif _junction_kind == "cut":
            junction_ctx = {
                "junction_kind": "cut",
                "junction_note": (
                    "硬切换场:背景已变,本镜是全新构图;禁止书写任何承接"
                    "上一镜的连续性语句(不写承接/入场对齐/未尽动作);"
                    "人物与场景一致性由引用图保证。" if prompt_lang == "zh"
                    else
                    "HARD CUT: the background changed — this shot is a "
                    "FRESH composition. Do NOT write any continuity with "
                    "the previous shot (no carry-over, no entry alignment, "
                    "no unfinished action); character and location "
                    "consistency ride on the reference images."),
                "required_end_state": entry.end_state or None,
            }
            entry.junction_meta = {
                **(getattr(entry, "junction_meta", None) or {}),
                "kind": "cut", "route_reason": _route_reason[:300]}
            decisions.append({"stage": "junction", "label": entry.label,
                              "strategy": "cut",
                              "reason": _route_reason[:160]})
        elif _junction_kind == "continue":
            junction_actual = W._junction_state(
                mllm, prev, shot_dir, tail_s=window_tail_s,
                portraits=storyboard.portraits)
            _tail_rep = W._parse_tail_report(junction_actual)
            junction_ctx = {
                "junction_kind": "continue",
                "prev_tail_report": _tail_rep or (junction_actual
                                                  or None),
                "prev_end_state_script": (getattr(prev, "end_state", "")
                                          or None) if prev else None,
                "required_end_state": entry.end_state or None,
            }
            entry.junction_meta = {
                **(getattr(entry, "junction_meta", None) or {}),
                "kind": "continue", "route_reason": _route_reason[:300],
                "tail_report": _tail_rep}
            decisions.append({"stage": "junction", "label": entry.label,
                              "strategy": "continue",
                              "reason": _route_reason[:160]})
        else:
            junction_ctx = {"required_end_state": entry.end_state or None}

        shot_cast = W._cast_in_shot(entry.description, storyboard.cast)
        shot_portraits = {n: storyboard.portraits[n] for n in shot_cast
                          if n in (storyboard.portraits or {})}
        # 背景板前插(生产原文:恒为自有图第一位)
        if "first_frame_plus_refs" in (video_gen.capabilities()
                                       or set()):
            _bgkey = (getattr(entry, "bg_id", "")
                      or f"scene_{entry.scene_idx}")
            _bg = (storyboard.backgrounds or {}).get(_bgkey) \
                or ({"path": storyboard.scene_anchors.get(
                        entry.scene_idx), "src": "t2i"}
                    if (storyboard.scene_anchors
                        or {}).get(entry.scene_idx) else None)
            if _bg and _bg.get("path") and Path(_bg["path"]).exists() \
                    and not any(im.get("source") == "background"
                                or str(im.get("path")) == str(_bg["path"])
                                for im in (entry.images or [])):
                if _bg.get("src") == "frame":
                    _bgdesc = (f"the OFFICIAL look of background {_bgkey} "
                               f"(a real frame from an earlier shot of "
                               f"this film) — the shot MUST take place in "
                               f"this SAME space: identical architecture, "
                               f"floor, furniture and lighting; never "
                               f"invent a different hall; IGNORE the "
                               f"people in it — do not copy them or "
                               f"their positions")
                else:
                    _bgdesc = (f"the OFFICIAL look of background {_bgkey} "
                               f"— the shot MUST take place in this SAME "
                               f"space: identical architecture, floor, "
                               f"furniture and lighting; never invent a "
                               f"different hall; do not copy its empty "
                               f"framing")
                entry.images = [{
                    "path": str(_bg["path"]), "role": "reference",
                    "source": "background",
                    "description": _bgdesc}] + list(entry.images or [])

        menu = W._condition_menu(entry, prev, video_gen,
                                 portraits=shot_portraits)
        if _junction_kind is not None:
            _cut_only = [m for m in menu if m["name"] == "ref2v"]
            if _cut_only:
                menu = _cut_only
                log.info("window: %s junction=%s → menu locked to ref2v",
                         entry.label, _junction_kind)
        slots_by_strategy = {
            m["name"]: W._slot_manifest(m["name"], entry, prev,
                                        use_prev_tail=True,
                                        source_videos=source_videos,
                                        portraits=shot_portraits,
                                        video_gen=video_gen)
            for m in menu}
        _ns_best = max((W._name_slot_map(v)
                        for v in slots_by_strategy.values()),
                       key=len, default={})
        _junction_mapped = dict(junction_ctx)
        if isinstance(_junction_mapped.get("prev_tail_report"), dict):
            _junction_mapped["prev_tail_report"] = W._map_tail_report(
                _junction_mapped["prev_tail_report"], _ns_best,
                storyboard.cast, portraits=storyboard.portraits)
        for _k in ("prev_end_state_script", "required_end_state"):
            if _junction_mapped.get(_k):
                _junction_mapped[_k] = W._map_markers(
                    _junction_mapped[_k], _ns_best)
        _cond_context = {
            "shot": entry.to_brain_line(),
            "prompt_language": prompt_lang,
            "prev_shot": prev.to_brain_line() if prev else None,
            "junction": _junction_mapped,
            "cast": storyboard.cast, "setting": storyboard.setting,
            "cast_in_shot": sorted(shot_cast),
            "slots_by_strategy": slots_by_strategy,
            "storyboard": storyboard.to_brain_json(),
            "episode_guidance": guidance}
        # RL 组采样(生产 2026-08-10 版语义原样:v0 默认温度、其余带
        # rl_temperature)。2026-08-20 组内并发:K 个请求【同时】发给
        # vLLM —— 同 state 意味着 prompt 逐字相同,continuous batching
        # 会自动复用前缀 KV,4 路几乎等于 1 路的时间。
        rl_variants = _run_concurrent(
            lambda t: W._decide(
                llm_video_brain, "generation-condition", menu,
                _cond_context,
                replay_hint=replay_cond.get(entry.label),
                priority=W._CONDITION_PRIORITY,
                temperature=t),
            [None] + [rl_temperature] * max(0, group - 1),
            _POLICY_CONCURRENCY)
        d = rl_variants[0]
        rl_state = {"menu": [dict(m) for m in menu],
                    "context": _cond_context}
        decisions.append({"stage": "condition", "label": entry.label,
                          **d})
        entry.draft_prompt = str(d.get("video_prompt") or "")

        def _prompt_chain(d):
            """决策 → 出门 prompt 全链(生产原文;enhancer 恒 None →
            润色段自然跳过)。"""
            brain_prompt = W._scrub_setting_sentence(
                W._scrub_cast_labels(
                    W._strip_markers(d.get("video_prompt", "")),
                    storyboard.cast),
                storyboard.setting, d["strategy"])
            use_tail = bool(d.get("use_prev_tail_video", False))
            slots = W._slot_manifest(d["strategy"], entry, prev, use_tail,
                                     source_videos=source_videos,
                                     portraits=shot_portraits,
                                     video_gen=video_gen)
            # 承接句机器化(生产原文)
            _pin_row = next((r_ for r_ in (slots or [])
                             if r_.get("source") == "pin_frame"), None)
            if _pin_row and brain_prompt:
                brain_prompt = re.sub(
                    r"[^。]*(?:从首帧精确开始|从第一帧开始|从第一帧精确开始|"
                    r"首帧即上一镜|第一帧与上一镜|上一镜的最后一帧|"
                    r"starts? EXACTLY on the given first)[^。]*。\s*",
                    "", brain_prompt).strip()
                if _pin_row["slot"] not in brain_prompt:
                    _tok = _pin_row["slot"]
                    brain_prompt = (
                        f"画面从{_tok}所示的首帧继续。"
                        if prompt_lang == "zh"
                        else f"The video continues from the first frame "
                             f"shown in {_tok}. ") + brain_prompt
            # 正典逐字契约(无锚路线)
            if d["strategy"] not in W._ANCHORED_STRATEGIES:
                brain_prompt, canon_notes = W._enforce_cast_canon(
                    brain_prompt, shot_cast, storyboard.cast)
            else:
                canon_notes = []
            for cn in canon_notes:
                decisions.append({**cn, "label": entry.label})
            # 音频线(生产原文;RL 农场 enable_audio=False 时全部短路)
            want_audio = bool(enable_audio and entry.dialogue)
            if enable_audio and not want_audio and brain_prompt \
                    and re.search("(?:说道?|says?|喊道?|大喊|高喊|怒吼|低语|回应|"
                                  "问道?|轻声问?)[^\"“]{0,6}?"
                                  "[:：]?\\s*[\"“]",
                                  brain_prompt):
                want_audio = True
                log.warning("window: %s prompt carries spoken lines but "
                            "the dialogue field is EMPTY — enabling "
                            "native audio anyway", entry.label)
                if "无背景音乐" not in brain_prompt \
                        and "no background music" not in brain_prompt:
                    _snd_i = W._scripted_sounds(entry.description,
                                                entry.end_state)
                    if re.search(r"[一-鿿]", brain_prompt):
                        brain_prompt += (
                            f"音频:角色对白的人声与剧本写明的环境声"
                            f"({'、'.join(_snd_i)})——无背景音乐、无其他"
                            f"音效。" if _snd_i else
                            "音频:只有角色对白的人声——无背景音乐、无音效。")
                    else:
                        brain_prompt += (
                            " Audio: the characters' voices plus the "
                            "scripted ambient sound "
                            f"({', '.join(_snd_i)}) — no "
                            "background music, no other effects."
                            if _snd_i else
                            " Audio: only the characters' voices — no "
                            "background music, no sound effects.")
            if enable_audio and not want_audio and brain_prompt:
                _snd_shot = W._scripted_sounds(entry.description,
                                               entry.end_state)
                if _snd_shot:
                    want_audio = True
                    log.info("window: %s no dialogue but scripted sounds "
                             "%s — native audio ON (sfx shot)",
                             entry.label, _snd_shot)
                    if "无背景音乐" not in brain_prompt \
                            and "no background music" not in brain_prompt:
                        brain_prompt += (
                            f"音频:只有剧本写明的环境声"
                            f"({'、'.join(_snd_shot)})——无背景音乐、"
                            f"无人声。"
                            if re.search(r"[一-鿿]", brain_prompt) else
                            " Audio: only the scripted ambient sound "
                            f"({', '.join(_snd_shot)}) — no background "
                            "music, no voices.")
            # 引用出口闸
            if brain_prompt:
                fixed, audit = W.validate_references(brain_prompt, slots)
                if not audit["ok"]:
                    log.warning("window: %s prompt references unknown "
                                "slots %s (allowed: %s) — dropping it",
                                entry.label, audit["unknown"],
                                audit["allowed"])
                    decisions.append({"stage": "ref_validate",
                                      "label": entry.label,
                                      "strategy": d["strategy"],
                                      "via": "gate",
                                      "reason": "unknown refs "
                                                f"{audit['unknown']}"})
                    brain_prompt = ""
                else:
                    if audit["appended"]:
                        decisions.append({"stage": "ref_validate",
                                          "label": entry.label,
                                          "strategy": d["strategy"],
                                          "via": "gate",
                                          "reason": "appended mentions: "
                                                    f"{audit['appended']}"})
                    brain_prompt = fixed
            if want_audio:
                brain_prompt = W._with_dialogue(
                    brain_prompt or spec.prompt, entry, storyboard.cast,
                    name_to_slot=W._name_slot_map(slots))
            # 旁白剥除 + 名字终换闸(生产原文)
            if brain_prompt:
                brain_prompt = re.sub(
                    r"(?:画外)?旁白[:：]?\s*[\"“][^\"“”]*"
                    r"[\"”]。?\s*|(?:voice-?over|narration)\s*[:：]"
                    r"[^.\"]*[.\"]?\s*",
                    "", brain_prompt, flags=re.IGNORECASE).strip()
                brain_prompt = W._names_to_tokens(
                    brain_prompt, W._name_slot_map(slots))
                _noq = re.sub(r'["“][^"“”]*["”]', "", brain_prompt)
                _leak = [n for n in (storyboard.cast or {}) if n in _noq]
                if _leak:
                    log.warning("window: %s outgoing prompt still "
                                "carries SLOTLESS cast name(s) %s "
                                "outside quotes", entry.label, _leak)
                    decisions.append({"stage": "name_leak",
                                      "label": entry.label,
                                      "names": _leak})
            return brain_prompt, slots, use_tail, want_audio

        # ── 生成:K 个变体各走同一条链(生产 _gen_plan 原文)────────
        # 组内并发(2026-08-20 用户令):出门链【串行】跑完(纯字符串
        # 处理,不耗时;decisions 顺序与 brain_log 因此保持确定),只把
        # 网络等待为主的条件执行丢进线程池。三处隔离:
        #   ①每候选一个客户端副本 → generate_audio 开关线程私有;
        #   ②每候选一个工作目录 shotNNN/cK → 中间产物(上镜尾帧、
        #     尾段裁片)同名不再互相覆盖;
        #   ③结果按下标回填 → 顺序与串行版逐位一致。
        chains = [(_i, _v) + _prompt_chain(_v)
                  for _i, _v in enumerate(rl_variants)]

        def _gen_one(item):
            s, _rl_d, brain_prompt, slots, use_tail, want_audio = item
            vg = (video_gen.clone() if hasattr(video_gen, "clone")
                  else video_gen)
            cand_dir = shot_dir / f"c{s}"
            _old_ga = getattr(vg, "generate_audio", False)
            if want_audio:
                vg.generate_audio = True
            try:
                video_path, cond = W._generate_with_condition(
                    _rl_d["strategy"], entry, prev, spec, vg,
                    cand_dir, seed=s, fps=fps,
                    window_tail_s=window_tail_s,
                    brain_prompt=brain_prompt,
                    use_prev_tail_video=use_tail,
                    source_videos=source_videos,
                    portraits=shot_portraits)
            except Exception as exc:
                log.warning("window: conditioned generation failed (%s): "
                            "%s — retrying the SAME strategy once",
                            _rl_d["strategy"], exc)
                try:
                    video_path, cond = W._generate_with_condition(
                        _rl_d["strategy"], entry, prev, spec, vg,
                        cand_dir, seed=s, fps=fps,
                        window_tail_s=window_tail_s,
                        brain_prompt=brain_prompt,
                        use_prev_tail_video=use_tail,
                        source_videos=source_videos,
                        portraits=shot_portraits)
                    cond["retried_after"] = f"exception: {exc}"[:200]
                except Exception as exc2:
                    try:
                        video_path, cond = W._generate_with_condition(
                            "t2v", entry, prev, spec, vg,
                            cand_dir, seed=s, fps=fps,
                            window_tail_s=window_tail_s,
                            brain_prompt=(W._with_dialogue(
                                spec.prompt, entry, storyboard.cast)
                                if want_audio else ""))
                        cond["degraded_from"] = _rl_d["strategy"]
                        cond["degraded_reason"] = \
                            f"exception: {exc2}"[:200]
                    except Exception as exc3:
                        log.warning("window: %s c%d ALL routes failed "
                                    "(%s)", entry.label, s,
                                    str(exc3)[:160])
                        video_path, cond = None, {
                            "strategy": _rl_d["strategy"],
                            "degraded_from": _rl_d["strategy"],
                            "degraded_reason":
                                f"all routes failed: {exc3}"[:200]}
            finally:
                vg.generate_audio = _old_ga
            if want_audio:
                cond["generate_audio"] = True
            cond["seed"] = s
            cond["rl"] = {"k": s, "decision_id": _rl_d.get("decision_id"),
                          "via": _rl_d.get("via"),
                          "strategy": _rl_d.get("strategy")}
            cond["final_prompt"] = brain_prompt or cond.get(
                "final_prompt", "")
            return s, video_path, cond

        results = _run_concurrent(_gen_one, chains, _GEN_CONCURRENCY)
        videos = [r[1] for r in results]
        seed_conds = [r[2] for r in results]

        # ── 差异②:skill 判官择主干(评审板/锦标赛不存在)─────────
        rewards, judge_video = judge_group(
            judges, _cond_context, entry, storyboard, rl_variants,
            seed_conds, videos, run_dir.name)
        order = sorted(range(len(rl_variants)),
                       key=lambda i: (rewards[i].get("reward") or 0.0),
                       reverse=True)
        best_k = next((i for i in order if videos[i] is not None),
                      order[0])

        # 组记录(schema 与旧收集器契约一致;reward 内联;raw 从
        # brain_calls.jsonl 按 decision_id 回取 —— 生产单源)
        bidx = _brain_index(run_dir)
        samples = []
        for k, (v, cond, vid, rw) in enumerate(zip(rl_variants,
                                                   seed_conds, videos,
                                                   rewards)):
            completion = {k2: v2 for k2, v2 in v.items()
                          if k2 not in ("via", "decision_id")
                          and not k2.startswith("_")}
            call = bidx.get(v.get("decision_id")) or {}
            samples.append({
                "decision_id": v.get("decision_id"),
                "via": v.get("via"),
                "completion": json.dumps(completion, ensure_ascii=False),
                "raw": call.get("raw")
                       or json.dumps(completion, ensure_ascii=False),
                "usable": bool(call.get("usable",
                                        v.get("via") == "llm")),
                "strategy": cond.get("strategy", v.get("strategy")),
                "degraded_from": cond.get("degraded_from"),
                "final_prompt": cond.get("final_prompt"),
                "video": str(vid) if vid else None,
                "chosen": k == best_k, **rw})
        try:
            with open(run_dir / "rl_steps.jsonl", "a") as _f:
                _f.write(json.dumps({
                    "kind": "condition_group",
                    "run": run_dir.name,
                    "shot_idx": entry.shot_idx,
                    "label": entry.label,
                    "junction_kind": (entry.junction_meta or {}
                                      ).get("kind"),
                    "policy_version": os.environ.get(
                        "MAESTRO_POLICY_VERSION", "0"),
                    "group_size": len(rl_variants),
                    "menu": rl_state["menu"],
                    "context": rl_state["context"],
                    "judge_video": judge_video,
                    "samples": samples,
                }, ensure_ascii=False, default=str) + "\n")
        except Exception as _exc:
            log.warning("rl group record failed (%s)", str(_exc)[:120])
        n_groups += 1

        if videos[best_k] is None:
            log.warning("window: %s ALL candidates failed — episode "
                        "aborted after %d groups", entry.label, n_groups)
            aborted = True
            break

        # 台账归因(生产同律:按主干胜者的实际条件;分歧才展开 per_seed)
        winner_cond = dict(seed_conds[best_k])
        winner_cond["decided_strategy"] = rl_variants[0]["strategy"]
        winner_cond["decided_via"] = rl_variants[0]["via"]
        distinct = {json.dumps({k: v for k, v in c.items()
                                if k not in ("seed", "rl")},
                               sort_keys=True) for c in seed_conds}
        if len(distinct) > 1:
            winner_cond["per_seed"] = seed_conds
        storyboard.set_condition(entry.shot_idx, winner_cond)
        storyboard.set_result(entry.shot_idx, Path(videos[best_k]),
                              converged=False, repair_actions=[])
        brain_log("window/shot_outcome", {
            "label": entry.label, "shot_idx": entry.shot_idx,
            "converged": False, "stop_reason": "judge_trunk",
            "repair_turns": 0, "gen_calls": len(rl_variants),
            "condition_decision_id": rl_variants[best_k].get(
                "decision_id"),
            "decided_strategy": rl_variants[best_k]["strategy"],
            "decided_via": rl_variants[best_k]["via"],
            "trunk_k": best_k,
            "trunk_reward": rewards[best_k].get("reward"),
        })
        log.info("window: %s done — trunk=c%d reward=%s", entry.label,
                 best_k, rewards[best_k].get("reward"))

        # ③空间圣经·实拍回流(生产原文)
        best_vp = Path(videos[best_k])
        if best_vp.exists():
            _bgk3 = (getattr(entry, "bg_id", "")
                     or f"scene_{entry.scene_idx}")
            _tailf = W._last_frame(best_vp,
                                   shot_dir / "space_upgrade_tail.png")
            if _tailf is not None:
                _upv = washed_frame_upgrade(
                    storyboard, _bgk3, Path(_tailf), image_edit, mllm,
                    llm_video_brain, run_dir / "spaces",
                    entry.shot_idx)
                if _upv:
                    entry.junction_meta = {
                        **(getattr(entry, "junction_meta", None) or {}),
                        "frame_upgrade": {
                            "view": _upv,
                            "path": storyboard.spaces[_bgk3][_upv]
                            ["path"]}}
                    decisions.append({"stage": "space_view",
                                      "bg": _bgk3, "view": _upv,
                                      "via": "frame_upgrade",
                                      "label": entry.label})

    # ── §E 合成(生产同律:时长对账 + concat;音频/BGM 关)────────
    final = None
    if not aborted:
        clips, assemble_notes = W._final_cut(storyboard, run_dir)
        decisions.extend(assemble_notes)
        for e_ in storyboard.entries:
            if not e_.video_path or not Path(e_.video_path).exists():
                continue
            planned_ = (specs[e_.shot_idx].duration
                        if e_.shot_idx < len(specs) else None)
            if not planned_:
                continue
            actual_ = W._probe_seconds(Path(e_.video_path))
            if actual_ > 0 and abs(actual_ - float(planned_)) > max(
                    1.5, 0.3 * float(planned_)):
                log.warning("assemble AUDIT: %s runs %.1fs but the plan "
                            "says %.1fs", e_.label, actual_,
                            float(planned_))
        if clips:
            try:
                final = _concat(clips, run_dir / "movie.mp4")
            except Exception as exc:
                log.warning("window: FINAL MERGE FAILED (%s) — per-shot "
                            "clips remain", exc)
    (run_dir / "decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=1, default=str))
    return {"groups": n_groups, "run": run_dir.name,
            "movie": str(final) if final else None,
            "aborted": aborted}
