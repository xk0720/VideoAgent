"""RL agent loop(2026-08-19 用户令:整个训练环境重建进 rl/,主管线
零依赖)。精简版窗口式生成,只保留训练必需:

  §A  剧本→分镜(冻结 LLM + scene_write 技能,任务 prompt 与生产逐字
      同款 —— 冻结 agent 的输入分布不漂)
  §A' 资产(肖像 t2i 逐角色、背景板 t2i 逐 bg;资产保证闸:缺 = 硬停)
  §C  逐镜:junction(精简:同 bg=continue、换 bg=cut;不做 derive
      派生帧)→ 菜单(shot0=[t2v,ref2v],其余锁 [ref2v],与生产菜单锁
      一致)→ 策略 K 组采样(v0 默认温度,其余 rl_temperature)→ 每
      候选走同一条出门链(剥标记/契约清洗/引用闸/名字终换/对白音频)
      → 可灵生成
  §R  skill 判官组内评审(文本判官逐候选 + 三路视频排名 + 一致性对照
      —— rl/reward/judges.py 同一套技能文件)→ compose_rewards →
      argmax = 主干推进;全部判词落 JudgeLog
  §W  rl_steps.jsonl 组记录(schema 与旧版同,reward 已内联 ——
      收集器只聚合,不再二次评审)

刻意去掉的生产件(用户令"多余的全部去掉"):修复循环、语义/物理
critic、metric_tool、锦标赛、prompt enhancer、episode 记忆、derive 缝合、
空间圣经、BGM/转场、蒸馏。
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from env.skills import decision_prompt, extract_json, skill_body  # noqa: E402

# ── 生产同款常量/小闸(逐字移植,来源 window_loop.py)────────────────
_MARKER_RE = re.compile(r"<([^<>\n]{1,60})>")
_CAST_SPLIT_RE = re.compile(
    r"static[:：]\s*(.+?)\s*[;;,]\s*dynamic[:：].*",
    re.IGNORECASE | re.DOTALL)
_SHOT_PREFIX_RE = re.compile(
    r"^\s*shot\s+\d+\s*:\s*(scene\s+\d+\s*[—-]\s*)?", re.IGNORECASE)
_REF_TOKEN_RE = re.compile(
    r"@(?:Image|Video)\d+|reference image \d+|<<<(?:image|video)_\d+>>>",
    re.IGNORECASE)
_KLING_VARIANT_RE = re.compile(
    r"<{1,4}\s*(image|video)[\s_]?(\d+)\s*>{1,4}", re.IGNORECASE)
_PORTRAIT_SLOT_CONTENT = (
    "official portrait of {name} — identity ONLY: match the character's "
    "face/build/wardrobe to it; NEVER copy its pose, framing or background")
_CONDITION_PRIORITY = ["ref2v", "t2v"]        # 精简菜单的确定性兜底序


def strip_markers(text: str) -> str:
    return _MARKER_RE.sub(r"\1", text or "")


def static_half(desc: str) -> str:
    s = str(desc or "").strip()
    m = _CAST_SPLIT_RE.match(s)
    if m:
        return m.group(1).strip()
    cut = re.split(r"dynamic[:：]", s, maxsplit=1,
                   flags=re.IGNORECASE)[0]
    cut = re.sub(r"^\s*static[:：]\s*", "", cut, flags=re.IGNORECASE)
    return cut.strip().rstrip(";;,. ").strip()


def scrub_cast_labels(text: str, cast: dict) -> str:
    out = str(text or "")
    for v in (cast or {}).values():
        s = str(v or "").strip()
        m = _CAST_SPLIT_RE.match(s)
        if m and s in out:
            out = out.replace(s, m.group(1).strip())
    return re.sub(r"\bstatic[:：]\s*", "", out)


def cast_in_shot(description: str, cast: dict) -> dict:
    if not cast:
        return {}
    marks = {m.strip().lower()
             for m in _MARKER_RE.findall(description or "")}
    if not marks:
        return dict(cast)
    hit = {k: v for k, v in cast.items() if k.strip().lower() in marks}
    return hit or dict(cast)


def normalize_ref_tokens(prompt: str) -> str:
    return _KLING_VARIANT_RE.sub(
        lambda m: f"<<<{m.group(1).lower()}_{m.group(2)}>>>",
        str(prompt or ""))


def validate_references(prompt: str, slots: list[dict],
                        zh: bool) -> tuple[str, dict]:
    """出口引用闸(生产同款规则):清单外编号 → 整条作废;可引用槽位
    漏提 → 自动补一句。"""
    prompt = normalize_ref_tokens(prompt)
    allowed = {str(r.get("slot", "")).lower(): r for r in slots
               if r.get("referenceable")}
    found = {m.group(0) for m in _REF_TOKEN_RE.finditer(prompt)}
    unknown = sorted(t for t in found if t.lower() not in allowed)
    if unknown:
        return "", {"ok": False, "unknown": unknown,
                    "allowed": sorted(r["slot"] for r in slots
                                      if r.get("referenceable"))}
    mentioned = {t.lower() for t in found}
    appended = []
    fixed = prompt.rstrip()
    for r in slots:
        if not r.get("referenceable"):
            continue
        slot = str(r.get("slot", ""))
        if slot.lower() in mentioned:
            continue
        content = str(r.get("content", "")).strip()
        _cjk = any("一" <= ch <= "鿿" for ch in content)
        if zh and not _cjk:
            fixed += f"画面中包含{slot}所示之物,外观与其保持一致。"
        else:
            fixed += (f" {slot} shows: {content} — keep it consistent."
                      if content else f" Keep {slot} consistent.")
        appended.append(slot)
    return fixed, {"ok": True, "unknown": [], "appended": appended}


def names_to_tokens(text: str, name_to_slot: dict) -> str:
    if not text or not name_to_slot:
        return text
    parts = re.split(r'(["“][^"“”]*["”])', text)
    for i in range(0, len(parts), 2):
        for n, tok in name_to_slot.items():
            if n in parts[i]:
                parts[i] = parts[i].replace(n, tok)
    return "".join(parts)


def with_dialogue(prompt: str, entry: dict, name_to_slot: dict,
                  zh: bool) -> str:
    """对白子句 + 无 BGM 压制句(生产规则的精简版:台词已在场就只补
    压制句;不在场则按说话人记号补一句)。"""
    line = str(entry.get("dialogue") or "").strip()
    if not line or not prompt:
        return prompt
    audio_zh = "音频:只有角色说这句台词的人声——无背景音乐、无音效。"
    audio_en = ("Audio: only the character's voice speaking the line — "
                "no background music, no sound effects.")

    def ensure_audio(p_):
        if "无背景音乐" in p_ or "no background music" in p_:
            return p_
        return f"{p_} {audio_zh}" if zh else f"{p_} {audio_en}"
    if line in prompt:
        return ensure_audio(prompt)
    who = (entry.get("dialogue_speaker") or "").strip() or "the character"
    subj = name_to_slot.get(who, who)
    if zh:
        return ensure_audio(f'{prompt} {subj}说:"{line}"。')
    return ensure_audio(f'{prompt} {subj} says: "{line}".')


# ── §A 分镜(scene_write,任务 prompt 与生产逐字同款)────────────────
def scene_write_prompt(task_text: str, prompt_language: str,
                       max_shots: int) -> str:
    skill_text = skill_body("scene_write")
    return (
        skill_text
        + "\n\nTHIS TASK (JSON):\n"
        + json.dumps({"user_prompt": task_text,
                      "prompt_language": prompt_language,
                      "cast_canon": {},
                      "asset_catalog": [],
                      "episode_guidance": {"past_task_shapes": []},
                      "max_shots_hard_cost_cap": max_shots},
                     ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"cast": {"<entity name>": "<10-20 '
          'word CANONICAL appearance descriptor (species/build, coat/'
          'wardrobe with colors, distinctive marks) — every shot prompt '
          'will restate it VERBATIM>"}, "setting": "<one canonical '
          'set-dressing + lighting sentence for the (main) scene>", '
          '"shots": [{"description": "Shot 1: '
          '<detailed filmable description — mark every cast character '
          'as <name> in angle brackets, names copied from cast keys>", '
          '"duration_s": <int 4-10>, '
          '"end_state": "<one sentence: at the CUT, who/what is where, '
          'moving or still, in which direction — PLUS the camera\'s '
          'own state (static / pushing in / tracking right at walking '
          'pace ...)>", '
          '"variation": "large|medium|small (expected first-to-last '
          'frame change inside this shot)", '
          '"opening_frame": "<ONLY for the first shot and scene cuts: '
          'a purely STATIC opening snapshot (no ongoing actions); omit '
          'for continuing shots>", '
          '"dialogue": {"speaker": "<the cast key of WHO SPEAKS — copied '
          'verbatim from cast>", "line": "<ONE spoken line of at most '
          '6 words>"} — include ONLY when a cast character visibly '
          'speaks on screen (medium close-up or closer); omit '
          'otherwise, '
          '"bg": "<background id like bg_1 — keep the SAME id while '
          'the shot happens in the same physical space (the master '
          'background is unchanged); switch to a NEW id ONLY when the '
          'action moves to a different space. Predicting this drives '
          'which background reference image the generator receives>"}, '
          '...], "music_plan": {"scene 1": "<ONE music description '
          'for the whole scene: mood, genre, tempo/BPM — all shots in '
          'a scene share one track; omit a scene (or the whole field) '
          'for silence>"}} '
          "— each description 15-40 words (subject + action + "
          "setting + camera), scene N stated when the location changes. "
          "YOU decide the shot count AND each shot's duration_s (4-10 "
          "seconds, from how long the action NEEDS) from the story "
          "itself (use past_task_shapes as experience from similar "
          f"past tasks); max_shots ({max_shots}) is only a COST ceiling, "
          "never a target — never pad by repeating a shot. HANDOFF LAW: "
          "each shot's opening must continue the PREVIOUS shot's "
          "end_state exactly (position AND motion); to hand motion to "
          "the next shot, do NOT let the mover stop before the cut; a "
          "resting object may only move again if a NEW force/event acts "
          "on it (write that event into the description). SCRIPT LANGUAGE LAW: "
        + ("EVERYTHING (descriptions, end_state, opening_frame, "
           "cast descriptors, setting) MUST be in CHINESE, "
           "EXCERPTING the screenplay's own action and performance "
           "wording verbatim wherever it exists — translation is "
           "loss (image-model strings are translated downstream). "
           if prompt_language == "zh" else
           "cast descriptors, setting, descriptions, end_state, "
           "variation and opening_frame MUST be ENGLISH (they feed "
           "image/video models directly). ")
        + "Entity NAMES in cast keys and dialogue lines always "
          "stay in the user's language. CAST CANON: when the task "
          "JSON carries a non-empty cast_canon, adopt those names and "
          "descriptors VERBATIM in your cast output — you may only ADD "
          "characters it missed, never rename or rewrite them."
    )


def build_storyboard(frozen_llm, task_text: str, prompt_language: str,
                     max_shots: int = 16) -> dict:
    """§A:scene_write → {"cast","setting","shots":[entry…]}。坏回复
    重试 2 次;仍不可用 → 硬停(训练环境没有确定性拆条的价值)。"""
    prompt = scene_write_prompt(task_text, prompt_language, max_shots)
    data = None
    for attempt in range(3):
        try:
            raw = frozen_llm.complete(prompt, max_tokens=16384)
            data = extract_json(raw)
        except Exception as exc:
            print(f"[env] scene_write call failed: {str(exc)[:200]}",
                  flush=True)
            data = None
        if isinstance(data, dict) and isinstance(data.get("shots"), list) \
                and len(data["shots"]) >= 2:
            break
        data = None
    if data is None:
        raise RuntimeError("scene_write unusable after retries — "
                           "rollout aborted (no deterministic fallback "
                           "in the RL env)")
    shots = []
    for i, s in enumerate(data["shots"][:max_shots]):
        if not isinstance(s, dict) or not str(s.get("description",
                                                    "")).strip():
            continue
        dlg = s.get("dialogue") or {}
        shots.append({
            "shot_idx": len(shots),
            "label": f"shot {len(shots) + 1}",
            "description": str(s["description"]).strip(),
            "duration_s": s.get("duration_s"),
            "end_state": str(s.get("end_state", "")).strip(),
            "variation": str(s.get("variation", "")).strip(),
            "opening_frame": str(s.get("opening_frame", "")).strip(),
            "dialogue": str(dlg.get("line", "")).strip()
            if isinstance(dlg, dict) else "",
            "dialogue_speaker": str(dlg.get("speaker", "")).strip()
            if isinstance(dlg, dict) else "",
            "bg_id": str(s.get("bg", "") or "bg_1").strip() or "bg_1",
            "status": "pending", "video": None,
        })
    return {"cast": {str(k): str(v) for k, v in
                     (data.get("cast") or {}).items()},
            "setting": str(data.get("setting", "")).strip(),
            "shots": shots}


# ── §A' 资产 ──────────────────────────────────────────────────────
def ensure_assets(sb: dict, frozen_llm, t2i, run_dir: Path) -> dict:
    """肖像逐角色 + 背景板逐 bg(t2i);资产保证闸:任一缺失 = 硬停。
    返回 {"portraits": {name: path}, "backgrounds": {bg_id: path}}。"""
    portraits, backgrounds = {}, {}
    setting = sb["setting"]
    for name, desc in sb["cast"].items():
        slug = re.sub(r"[^\w一-鿿]+", "_", name).strip("_") or "cast"
        out = run_dir / "portraits" / f"{slug}.png"
        if not out.exists():
            bg = (f"Background: {setting} — the character stands inside "
                  f"this exact scene, lit by its natural light."
                  if setting else "Neutral studio backdrop.")
            t2i.text_to_image(
                f"full-body portrait of {name}: {static_half(desc)}. "
                f"Standing, natural pose, facing the camera, whole "
                f"figure visible. {bg} cinematic still, high detail",
                out)
        portraits[name] = str(out)
    for bg_id in sorted({e["bg_id"] for e in sb["shots"]}):
        out = run_dir / "anchors" / f"bg_{bg_id}.png"
        if not out.exists():
            shots_here = [strip_markers(e["description"])
                          for e in sb["shots"] if e["bg_id"] == bg_id][:3]
            bg_prompt = None
            try:
                raw = frozen_llm.complete(
                    skill_body("scene_image")
                    + "\n\nTHIS TASK (JSON):\n"
                    + json.dumps({"bg_id": bg_id, "setting": setting,
                                  "sample_shots": shots_here},
                                 ensure_ascii=False)
                    + '\n\nSTRICT JSON only: {"prompt": "<one ENGLISH '
                      "t2i prompt for this location's EMPTY master "
                      'plate>"}')
                d = extract_json(raw)
                bg_prompt = (d or {}).get("prompt")
            except Exception as exc:
                print(f"[env] scene_image failed ({str(exc)[:120]}) — "
                      f"deterministic bg prompt", flush=True)
            if not bg_prompt:
                bg_prompt = f"{setting} — wide establishing view."
            bg_prompt = str(bg_prompt) + (
                " Empty scene: no people, no characters, no animals — "
                "architecture, furniture and lighting only.")
            t2i.text_to_image(bg_prompt, out)
        backgrounds[bg_id] = str(out)
    missing = [n for n in sb["cast"] if not Path(portraits[n]).exists()]
    missing += [b for b in backgrounds
                if not Path(backgrounds[b]).exists()]
    if missing:
        raise RuntimeError(f"asset guarantee failed — missing: {missing}")
    return {"portraits": portraits, "backgrounds": backgrounds}


# ── §C 逐镜状态构造 ───────────────────────────────────────────────
def junction_of(entry: dict, prev: dict | None) -> str | None:
    """精简 junction:shot0=None;换 bg=cut;同 bg=continue。
    (生产的 derive 派生帧机器不进 RL 环境 —— 用户令去多余。)"""
    if prev is None or not prev.get("video"):
        return None
    return "cut" if entry["bg_id"] != prev["bg_id"] else "continue"


def junction_ctx(kind: str | None, entry: dict, prev: dict | None,
                 zh: bool) -> dict:
    if kind == "cut":
        return {"junction_kind": "cut",
                "junction_note": (
                    "硬切换场:背景已变,本镜是全新构图;禁止书写任何承接"
                    "上一镜的连续性语句(不写承接/入场对齐/未尽动作);"
                    "人物与场景一致性由引用图保证。" if zh else
                    "HARD CUT: the background changed — this shot is a "
                    "FRESH composition. Do NOT write any continuity with "
                    "the previous shot (no carry-over, no entry alignment, "
                    "no unfinished action); character and location "
                    "consistency ride on the reference images."),
                "required_end_state": entry["end_state"] or None}
    if kind == "continue":
        return {"junction_kind": "continue",
                "prev_tail_report": None,
                "prev_end_state_script": (prev or {}).get("end_state")
                or None,
                "required_end_state": entry["end_state"] or None}
    return {"required_end_state": entry["end_state"] or None}


def build_menu(kind: str | None, has_refs: bool) -> list[dict]:
    """精简菜单(与生产的可灵菜单锁一致:非首镜锁 ref2v)。"""
    menu = [{"name": "t2v", "description": "Text only — nothing else "
             "is available (last resort)."}]
    if has_refs:
        menu.append({"name": "ref2v",
                     "description": "Reference-to-video: every planned "
                                    "reference image and official "
                                    "portrait rides the reference "
                                    "channel (<<<image_N>>>). THE "
                                    "route for scene cuts with "
                                    "characters. Mention every slot "
                                    "with its content."})
    if kind is not None:
        cut_only = [m for m in menu if m["name"] == "ref2v"]
        if cut_only:
            menu = cut_only
    return menu


def slot_manifest(strategy: str, bg_path: str | None,
                  shot_portraits: dict) -> list[dict]:
    """ref2v 槽位清单:背景板 → 肖像(按名排序);编号 = refer 装配序。"""
    if strategy != "ref2v":
        return []
    rows = []
    n = 1
    if bg_path:
        rows.append({"slot": f"<<<image_{n}>>>",
                     "content": "the location's empty master plate — "
                                "match its space layout and lighting",
                     "referenceable": True})
        n += 1
    for name in sorted(shot_portraits):
        rows.append({"slot": f"<<<image_{n}>>>",
                     "content": _PORTRAIT_SLOT_CONTENT.format(name=name),
                     "referenceable": True, "name": name})
        n += 1
    return rows


def brain_line(e: dict) -> dict:
    """台账行(键集与生产 to_brain_line 对齐,策略看到同形状态)。"""
    return {"label": e["label"], "description": e["description"],
            "end_state": e["end_state"], "variation": e["variation"],
            "opening_frame": e["opening_frame"], "dialogue": e["dialogue"],
            "status": e["status"], "image_plan": None,
            "images": ([{"role": "reference", "source": "background",
                         "description": "location master plate"}]
                       if e.get("_bg_attached") else []),
            "keyframe": None, "keyframe_source": None,
            "video": e.get("video"), "condition_strategy":
            e.get("condition_strategy"), "last_score": e.get("last_score"),
            "open_defects": []}


def build_context(sb: dict, entry: dict, prev: dict | None,
                  junction: dict, slots_by_strategy: dict,
                  shot_cast: dict, prompt_language: str) -> dict:
    return {"shot": brain_line(entry),
            "prompt_language": prompt_language,
            "prev_shot": brain_line(prev) if prev else None,
            "junction": junction,
            "cast": sb["cast"], "setting": sb["setting"],
            "cast_in_shot": sorted(shot_cast),
            "slots_by_strategy": slots_by_strategy,
            "storyboard": [brain_line(e) for e in sb["shots"]],
            "episode_guidance": {"replay_hints": [], "avoid": [],
                                 "n_episodes_matched": 0}}


# ── 策略采样(_decide 精简版:LLM 严格 JSON → 确定性兜底)────────────
def decide(policy, menu: list[dict], context: dict,
           temperature=None) -> dict:
    valid = {m["name"] for m in menu}
    prompt = decision_prompt(skill_body("window_generation"), menu, context)
    raw = ""
    try:
        raw = policy.complete(prompt, temperature=temperature)
        data = extract_json(raw)
    except Exception as exc:
        raw = raw or f"<complete() raised: {exc}>"
        data = None
    out = None
    if isinstance(data, dict) and str(data.get("strategy", "")) in valid:
        out = {"strategy": str(data["strategy"]),
               "reason": str(data.get("reason", ""))}
        if isinstance(data.get("video_prompt"), str) \
                and data["video_prompt"].strip():
            out["video_prompt"] = data["video_prompt"].strip()
        out["via"] = "llm"
    if out is None:
        name = next((n for n in _CONDITION_PRIORITY if n in valid),
                    sorted(valid)[0])
        out = {"strategy": name, "via": "fallback",
               "reason": "deterministic priority (brain reply unusable)"}
    out["decision_id"] = uuid.uuid4().hex[:16]
    out["_raw"] = raw
    return out


def sample_group(policy, menu, context, k: int,
                 temperature: float) -> list[dict]:
    """K 个决策同一 state:v0 默认温度,v1..k-1 带 rl_temperature。"""
    group = [decide(policy, menu, context)]
    for _ in range(max(0, k - 1)):
        group.append(decide(policy, menu, context,
                            temperature=temperature))
    return group


# ── 出门链(prompt_chain 精简版)──────────────────────────────────
def outgoing_prompt(d: dict, entry: dict, slots: list[dict], cast: dict,
                    zh: bool) -> tuple[str, bool]:
    """决策 → 出门 prompt:剥标记 → 契约清洗 → 引用闸(错编号弃用,
    落剧本兜底)→ 名字终换 → 对白+无BGM。返回 (prompt, want_audio)。"""
    p = scrub_cast_labels(strip_markers(d.get("video_prompt", "")), cast)
    fallback = _SHOT_PREFIX_RE.sub("", strip_markers(entry["description"]))
    if p:
        fixed, audit = validate_references(p, slots, zh)
        p = fixed if audit["ok"] else ""
    if not p:
        p, _ = validate_references(fallback, slots, zh)
        p = p or fallback
    name_to_slot = {r["name"]: r["slot"] for r in slots
                    if r.get("name") and r.get("referenceable")}
    p = names_to_tokens(p, name_to_slot)
    want_audio = bool(entry.get("dialogue"))
    if want_audio:
        p = with_dialogue(p, entry, name_to_slot, zh)
    return p, want_audio


# ── §R 组内评审(skill 判官;择优即主干)──────────────────────────
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


def judge_group(judges, context: dict, entry: dict, assets: dict,
                variants: list[dict], videos: list, run: str,
                junction: dict) -> tuple[list[dict], dict]:
    """组内双重评审(文本逐候选 + 三路排名 + 一致性)→ compose。
    返回 (per-candidate dicts, group-level judge_video)。分量失败剔除
    归一化;全失败 = reward 只剩 format 分(诚实,不编数)。"""
    from reward.judges import compose_rewards
    n = len(variants)
    fmt = [1.0 if v.get("via") == "llm" else 0.0 for v in variants]
    kind = junction.get("junction_kind")
    text_scores: list = []
    details: list[dict] = [{} for _ in range(n)]
    for i, v in enumerate(variants):
        try:
            case = {
                "shot_script": entry["description"],
                "cast_canon": context.get("cast") or {},
                "story_so_far": [{"label": r.get("label"),
                                  "description": r.get("description")}
                                 for r in (context.get("storyboard")
                                           or [])],
                "prev_end_state": (context.get("prev_shot")
                                   or {}).get("end_state", ""),
                "junction": {"kind": kind,
                             "continuity_applicable":
                                 kind == "continue",
                             "handoff_required": False},
                "slots": (context.get("slots_by_strategy")
                          or {}).get(v.get("strategy"), []),
                "candidate_prompt": v.get("video_prompt", ""),
            }
            score, detail = judges["text"].score(
                case, tag={"run": run, "label": entry["label"],
                           "candidate": i,
                           "decision_id": v.get("decision_id")})
            details[i]["judge_text"] = detail
            text_scores.append(score)
        except Exception as exc:
            print(f"[judge] text failed ({str(exc)[:120]})", flush=True)
            details[i]["judge_text"] = {"error": str(exc)[:120]}
            text_scores.append(None)

    video_parts: dict = {"action": None, "physics": None,
                         "camera": None, "consistency": None}
    judge_video: dict = {}
    ok_videos = [v for v in videos if v and Path(str(v)).exists()]
    if len(ok_videos) == n and n >= 2:
        rank_ctx = {"shot_script": entry["description"],
                    "camera_facing": entry.get("camera_facing", ""),
                    "cast_canon": context.get("cast") or {}}
        for dim in ("action", "physics", "camera"):
            try:
                res = judges["ranker"].rank(
                    dim, rank_ctx, [str(v) for v in videos],
                    tag={"run": run, "label": entry["label"]})
                video_parts[dim] = res["points"]
                judge_video[dim] = {"evidence": res.get("evidence"),
                                    "order": res.get("order")}
            except Exception as exc:
                print(f"[judge] rank {dim} failed ({str(exc)[:120]})",
                      flush=True)
                judge_video[dim] = {"error": str(exc)[:120]}
        refs = []
        for name in (context.get("cast_in_shot") or []):
            pth = (assets.get("portraits") or {}).get(name)
            if pth and Path(pth).exists():
                refs.append({"kind": f"portrait:{name}", "path": pth,
                             "note": (context.get("cast")
                                      or {}).get(name, "")[:120]})
        bgp = (assets.get("backgrounds") or {}).get(entry["bg_id"])
        if bgp and Path(bgp).exists():
            refs.append({"kind": "space:master_plate", "path": bgp,
                         "note": (context.get("setting") or "")[:200]})
        if refs:
            cons: dict = {}
            for i, v in enumerate(videos):
                try:
                    sc, detail = judges["consistency"].score(
                        str(v), refs,
                        {"shot_script": entry["description"]},
                        tag={"run": run, "label": entry["label"],
                             "candidate": i})
                    cons[i] = sc
                    details[i]["judge_consistency"] = detail
                except Exception as exc:
                    print(f"[judge] consistency failed "
                          f"({str(exc)[:120]})", flush=True)
                    details[i]["judge_consistency"] = {
                        "error": str(exc)[:120]}
            video_parts["consistency"] = cons or None
    composed = compose_rewards(fmt, text_scores, video_parts, n)
    for i in range(n):
        details[i].update(composed[i])
    return details, judge_video


# ── §W 组记录 ────────────────────────────────────────────────────
def write_step_record(run_dir: Path, entry: dict, junction: dict,
                      menu: list, context: dict, variants: list[dict],
                      videos: list, rewards: list[dict],
                      judge_video: dict, best_k: int):
    samples = []
    for k, (v, vid, rw) in enumerate(zip(variants, videos, rewards)):
        # completion = 训练目标:只留决策语义字段(与 STRICT JSON 输出
        # 契约同形);内部簿记(_ 前缀)与机械字段绝不入内
        completion = {k2: v2 for k2, v2 in v.items()
                      if k2 not in ("via", "decision_id")
                      and not k2.startswith("_")}
        samples.append({"decision_id": v.get("decision_id"),
                        "via": v.get("via"),
                        "completion": json.dumps(completion,
                                                 ensure_ascii=False),
                        "raw": v.get("_raw", ""),
                        "usable": v.get("via") == "llm",
                        "strategy": v.get("strategy"),
                        "degraded_from": None,
                        "final_prompt": v.get("_final_prompt"),
                        "video": str(vid) if vid else None,
                        "chosen": k == best_k,
                        **rw})
    rec = {"kind": "condition_group", "run": run_dir.name,
           "shot_idx": entry["shot_idx"], "label": entry["label"],
           "junction_kind": junction.get("junction_kind"),
           "policy_version": os.environ.get("MAESTRO_POLICY_VERSION",
                                            "0"),
           "group_size": len(variants), "menu": menu,
           "context": context, "judge_video": judge_video,
           "samples": samples}
    with open(run_dir / "rl_steps.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def save_storyboard(run_dir: Path, sb: dict, assets: dict):
    tmp = run_dir / "storyboard.json.tmp"
    tmp.write_text(json.dumps(
        {"cast": sb["cast"], "setting": sb["setting"],
         "portraits": assets.get("portraits", {}),
         "backgrounds": assets.get("backgrounds", {}),
         "entries": sb["shots"]}, ensure_ascii=False, indent=1,
        default=str))
    tmp.replace(run_dir / "storyboard.json")


# ── 主循环 ───────────────────────────────────────────────────────
def run_episode(*, task_text: str, run_dir: Path, frozen_llm, policy,
                kling, t2i, judges, group: int = 4,
                rl_temperature: float = 0.9, max_shots: int = 16) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    zh = bool(re.search(r"[一-鿿]", task_text))
    lang = "zh" if zh else "en"
    print(f"[env] §A scene_write ({lang}) …", flush=True)
    sb = build_storyboard(frozen_llm, task_text, lang, max_shots)
    print(f"[env] storyboard: {len(sb['shots'])} shots, "
          f"cast={list(sb['cast'])}", flush=True)
    assets = ensure_assets(sb, frozen_llm, t2i, run_dir)
    save_storyboard(run_dir, sb, assets)
    prev = None
    n_groups = 0
    for entry in sb["shots"]:
        kind = junction_of(entry, prev)
        jctx = junction_ctx(kind, entry, prev, zh)
        shot_cast = cast_in_shot(entry["description"], sb["cast"])
        shot_portraits = {n: assets["portraits"][n] for n in shot_cast
                          if n in assets["portraits"]}
        bg_path = assets["backgrounds"].get(entry["bg_id"])
        entry["_bg_attached"] = bool(bg_path)
        menu = build_menu(kind, has_refs=bool(bg_path or shot_portraits))
        slots_by = {m["name"]: slot_manifest(m["name"], bg_path,
                                             shot_portraits)
                    for m in menu}
        context = build_context(sb, entry, prev, jctx, slots_by,
                                shot_cast, lang)
        print(f"[env] {entry['label']} junction={kind} "
              f"menu={[m['name'] for m in menu]} — sampling K={group}",
              flush=True)
        variants = sample_group(policy, menu, context, group,
                                rl_temperature)
        shot_dir = run_dir / f"shot{entry['shot_idx']:03d}"
        videos: list = []
        for k, v in enumerate(variants):
            prompt, want_audio = outgoing_prompt(
                v, entry, slots_by.get(v["strategy"], []), sb["cast"], zh)
            v["_final_prompt"] = prompt
            out = shot_dir / f"shot{entry['shot_idx']:03d}_w_s{k}.mp4"
            refs = ([bg_path] if (bg_path and v["strategy"] == "ref2v")
                    else [])
            refs += [shot_portraits[n] for n in sorted(shot_portraits)] \
                if v["strategy"] == "ref2v" else []
            try:
                kling.generate(prompt, entry.get("duration_s"), out,
                               reference_images=refs or None,
                               audio=want_audio)
                videos.append(out)
            except Exception as exc:
                print(f"[env] {entry['label']} c{k} generation FAILED: "
                      f"{str(exc)[:200]}", flush=True)
                videos.append(None)
        rewards, judge_video = judge_group(
            judges, context, entry, assets, variants, videos,
            run_dir.name, jctx)
        order = sorted(range(len(variants)),
                       key=lambda i: (rewards[i].get("reward") or 0.0),
                       reverse=True)
        best_k = next((i for i in order if videos[i] is not None),
                      order[0])
        write_step_record(run_dir, entry, jctx, menu, context, variants,
                          videos, rewards, judge_video, best_k)
        n_groups += 1
        if videos[best_k] is None:
            print(f"[env] {entry['label']}: ALL candidates failed — "
                  f"episode aborted after {n_groups} groups", flush=True)
            break
        entry["video"] = str(videos[best_k])
        entry["status"] = "generated"
        entry["condition_strategy"] = variants[best_k]["strategy"]
        entry["last_score"] = rewards[best_k].get("reward")
        prev = entry
        save_storyboard(run_dir, sb, assets)
        print(f"[env] {entry['label']} trunk=c{best_k} "
              f"reward={rewards[best_k].get('reward')}", flush=True)
    return {"groups": n_groups, "run": run_dir.name}
