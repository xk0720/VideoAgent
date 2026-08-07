"""cinegraph 总编排(ViMax script2video_pipeline 生成环的移植)。

流程:§A 剧本/正典/分镜(复用 window 的帮手与全部闸门)→ CG 结构
→ 机位树 → 多视图注册表 → 逐机位首帧(根=选图生图;子=双镜切派生
+换人不换景)→ 逐镜末帧(variation≥medium)→ 逐镜视频(可灵:
ff i2v / ff+lf flf2v)+ 我们的台词/音效法 + 评审重掷 → 拼装。

短期记忆 = 台账为骨(StoryboardMemory)+ 工件为肉(working_dir 文件,
skip-if-exists 幂等续跑)+ frame_registry(帧历史,选图 agent 工作集)。
镜与镜之间不追像素连续 —— 一致性在图像层焊死。
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from ..language import set_output_lang
from ..memory.storyboard import StoryboardMemory
from ..pipeline.audio_stage import any_audio, normalize_for_concat
from ..pipeline.window_loop import (_cast_in_shot, _ensure_cast_portraits,
                                    _extract_characters, _names_to_tokens,
                                    _prompt_lang, _scripted_sounds,
                                    _strip_markers, _write_outline,
                                    _write_screenplay, set_run_ambience)
from ..tools.video_concat import VideoConcatTool
from ..types import CandidateClip, ShotSpec
from .camera_tree import construct_camera_tree
from .first_frame_factory import (NEW_CAMERA_NOTE, derive_new_camera,
                                  generate_frame)
from .interfaces import CGCamera, CGShot
from .portrait_registry import build_portrait_registry, registry_pairs
from .reference_selector import select_reference_images

log = get_logger("maestro.cinegraph")

_SHOT_PREFIX_RE = re.compile(r"^\s*shot\s+\d+\s*:\s*"
                             r"(scene\s+\d+\s*[—-]\s*)?", re.IGNORECASE)
_NARRATION_RE = re.compile(
    r"(?:画外)?旁白[:：]?\s*[\"“][^\"“”]*[\"”]。?\s*")


def _audio_clause(shot: CGShot, zh: bool) -> str:
    if shot.dialogue:
        if shot.sounds:
            return (f"音频:角色说这句台词的人声与剧本写明的环境声"
                    f"({'、'.join(shot.sounds)})——无背景音乐、无其他音效。"
                    if zh else
                    " Audio: the character's voice plus the scripted "
                    f"ambient sound ({', '.join(shot.sounds)}) — no "
                    "background music, no other effects.")
        return ("音频:只有角色说这句台词的人声——无背景音乐、无音效。"
                if zh else
                " Audio: only the character's voice speaking the line — "
                "no background music, no sound effects.")
    if shot.sounds:
        return (f"音频:只有剧本写明的环境声({'、'.join(shot.sounds)})"
                "——无背景音乐、无人声。" if zh else
                " Audio: only the scripted ambient sound "
                f"({', '.join(shot.sounds)}) — no background music, "
                "no voices.")
    return ""


def generate_movie_cinegraph(
    user_prompt: str,
    *,
    cache_dir: Path,
    llm=None,
    crew: Optional[dict] = None,
    mllm=None,
    video_gen=None,
    image_edit=None,
    board=None,
    asset_memory=None,
    screenwriter=None,
    screenplay: Optional[str] = None,
    given_characters: Optional[dict] = None,
    character_library=None,
    prompt_enhancer=None,
    plan_cfg: Optional[dict] = None,
    max_turns: int = 2,
    accept_bar: float = 0.60,
    enable_audio: bool = True,
    portraits_as_refer: bool = True,
    fps: int = 24,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    decisions: list = []
    _crew = dict(crew or {})
    llm_screenwriter = _crew.get("screenwriter") or llm
    llm_scene_writer = _crew.get("scene_writer") or llm
    llm_video_brain = _crew.get("video_brain") or llm

    # ── §A 剧本 → 语言 → 钦定角色(在场闸+图注)→ 正典 → 分镜 ──
    screenplay_text, sp_via = _write_screenplay(
        llm_screenwriter, user_prompt, screenplay, [])
    decisions.append({"stage": "screenplay", "via": sp_via})
    prompt_lang = _prompt_lang(screenplay_text or user_prompt)
    set_output_lang(prompt_lang)
    zh = prompt_lang == "zh"
    given_caps: dict = {}
    for _gn, _gp in (given_characters or {}).items():
        if _gn not in str(screenplay_text or user_prompt or ""):
            log.warning("cinegraph: given character %r appears NOWHERE in "
                        "the screenplay — binding SKIPPED", _gn)
            continue
        cap = ""
        if _gp and Path(_gp).exists() and mllm is not None:
            _fn = getattr(mllm, "caption_identity", None) \
                or getattr(mllm, "caption_image", None)
            if _fn is not None:
                try:
                    cap = str(_fn(Path(_gp)) or "").strip()
                except Exception as exc:
                    log.warning("cinegraph: caption for %r failed (%s)",
                                _gn, exc)
        given_caps[_gn] = cap
    cast_canon, ce_via = _extract_characters(
        llm_screenwriter, screenplay_text, given=(given_caps or None),
        prompt_language=prompt_lang)
    outline, durs, ends, meta, outline_via = _write_outline(
        llm_scene_writer, screenplay_text, [],
        episode_guidance={}, max_shots=int((plan_cfg or {}).get(
            "max_shots", 12)),
        fallback_fn=lambda: (screenwriter.run(screenplay_text or
                                              user_prompt, asset_memory)
                             if screenwriter is not None else [
                                 screenplay_text or user_prompt]),
        cast_canon=cast_canon, prompt_language=prompt_lang)
    decisions.append({"stage": "playwriting",
                      "strategy": f"{len(outline)} shots",
                      "via": outline_via})

    storyboard = StoryboardMemory.from_outline(
        outline, path=cache_dir / "storyboard.json")
    for i_, (entry_, end_) in enumerate(zip(storyboard.entries, ends)):
        entry_.end_state = end_
        for key, attr in (("variations", "variation"),
                          ("opening_frames", "opening_frame"),
                          ("dialogues", "dialogue"),
                          ("dialogue_speakers", "dialogue_speaker"),
                          ("bgs", "bg_id")):
            vals = meta.get(key) or []
            setattr(entry_, attr, vals[i_] if i_ < len(vals) else "")
    storyboard.cast = dict(meta.get("cast", {}))
    storyboard.setting = str(meta.get("setting", ""))
    storyboard.music_plan = dict(meta.get("music_plan", {}))
    set_run_ambience(storyboard.setting)
    for _gn, _gp in (given_characters or {}).items():
        if _gn in given_caps and _gp and Path(_gp).exists():
            storyboard.portraits[_gn] = str(_gp)
    storyboard._save()
    decisions.extend(_ensure_cast_portraits(
        storyboard, asset_memory, video_gen, cache_dir,
        library=character_library, llm=llm_screenwriter))

    # ── CG 结构:镜表 + 机位分组(scene_write 的 camera 标注;缺 →
    #    每镜自成机位,树 agent 负责组织)──
    cam_meta = meta.get("cameras") or []
    shots: list = []
    for i_, entry_ in enumerate(storyboard.entries):
        desc = _SHOT_PREFIX_RE.sub("", str(entry_.description or "")).strip()
        dial = entry_.dialogue or ""
        if isinstance(dial, dict):
            dial = str(dial.get("line") or "")
        ff_desc = str(entry_.opening_frame or "") or desc
        lf_desc = str(entry_.end_state or "")
        s = CGShot(
            idx=i_,
            cam_idx=(int(cam_meta[i_]) if i_ < len(cam_meta)
                     and str(cam_meta[i_]).strip().isdigit() else i_),
            visual_desc=desc, ff_desc=ff_desc, lf_desc=lf_desc,
            variation=str(entry_.variation or "small") or "small",
            duration=(float(durs[i_]) if i_ < len(durs)
                      and durs[i_] is not None else None),
            dialogue=str(dial), speaker=str(entry_.dialogue_speaker or ""),
            sounds=_scripted_sounds(entry_.description, entry_.end_state),
            ff_chars=sorted(_cast_in_shot(ff_desc, storyboard.cast)),
            lf_chars=sorted(_cast_in_shot(lf_desc or desc,
                                          storyboard.cast)))
        # 对白镜强制 ff-only(flf2v 无引用通道,说话人记号无处挂)
        if s.dialogue and s.variation in ("medium", "large"):
            log.info("cinegraph: shot %d has dialogue — forcing ff-only "
                     "(variation %s ignored for the frame pair)",
                     i_, s.variation)
            s.variation = "small"
        shots.append(s)
    order: list = []
    for s in shots:
        if s.cam_idx not in order:
            order.append(s.cam_idx)
    remap = {c: k for k, c in enumerate(order)}
    for s in shots:
        s.cam_idx = remap[s.cam_idx]
    cameras = [CGCamera(idx=k, active_shot_idxs=[
        s.idx for s in shots if s.cam_idx == k]) for k in range(len(order))]
    decisions.append({"stage": "cameras",
                      "strategy": f"{len(cameras)} cameras"})

    # ── 机位树 + 注册表 ──
    cameras = construct_camera_tree(llm_video_brain, cameras, shots)
    state = {"cameras": [vars(c) for c in cameras]}
    registry = build_portrait_registry(
        storyboard.portraits, image_edit, cache_dir / "registry")
    state["registry"] = registry

    def _t2i(prompt, out):
        if video_gen is not None and hasattr(video_gen, "text_to_image"):
            return video_gen.text_to_image(prompt, out)
        raise RuntimeError("no t2i backend")

    # ── 首帧审查闸(我们的加笔):mllm 图注 × llm 裁决,一次重掷 ──
    def _frame_ok(frame: Path, want_desc: str) -> bool:
        if mllm is None or llm_video_brain is None:
            return True
        _cap = getattr(mllm, "caption_image", None)
        if _cap is None:
            return True
        try:
            got = str(_cap(Path(frame)) or "")[:600]
            from ..pipeline.window_loop import _extract_json
            raw = llm_video_brain.complete(
                "Does the generated frame match the intended description "
                "well enough to animate? Judge composition, characters "
                "present, and setting.\nINTENDED: " + want_desc[:600]
                + "\nGENERATED (caption): " + got
                + '\nSTRICT JSON only: {"ok": true|false, '
                  '"reason": "<one sentence>"}')
            d = _extract_json(raw) or {}
            if not bool(d.get("ok", True)):
                log.warning("cinegraph: frame review REJECTED — %s",
                            d.get("reason"))
                return False
        except Exception as exc:
            log.warning("cinegraph: frame review errored (%s) — "
                        "fail-open, frame accepted", str(exc)[:160])
            return True
        return True

    def _make_frame(available, want_desc, out_path, tag):
        out_path = Path(out_path)
        if out_path.exists():
            return out_path
        for _try in range(2):
            sel = select_reference_images(llm_video_brain, available,
                                          want_desc)
            frame = generate_frame(image_edit, _t2i, sel, out_path)
            if _frame_ok(frame, want_desc):
                return frame
            frame.unlink(missing_ok=True)
            log.warning("cinegraph: regenerating frame %s (review reject)",
                        tag)
        # 二连拒 → 用最后一次产物(诚实降级,绝不空手)
        if not out_path.exists():
            sel = select_reference_images(llm_video_brain, available,
                                          want_desc)
            generate_frame(image_edit, _t2i, sel, out_path)
        return out_path

    # ── 逐机位首帧(树的拓扑序:根先行)──
    shots_by_idx = {s.idx: s for s in shots}
    cam_by_idx = {c.idx: c for c in cameras}
    cam_ff: dict = {}
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    def _emit_camera_ff(cam: CGCamera):
        if cam.idx in cam_ff:
            return
        if cam.parent_cam_idx is not None:
            _emit_camera_ff(cam_by_idx[cam.parent_cam_idx])
        first_idx = cam.active_shot_idxs[0]
        first = shots_by_idx[first_idx]
        out = frames_dir / f"shot{first_idx:03d}_ff.png"
        if out.exists():
            cam_ff[cam.idx] = out
            return
        available = registry_pairs(registry, first.ff_chars)
        if cam.parent_cam_idx is not None:
            parent_cam = cam_by_idx[cam.parent_cam_idx]
            p_shot = shots_by_idx.get(
                cam.parent_shot_idx
                if cam.parent_shot_idx is not None
                else parent_cam.active_shot_idxs[0])
            derived = derive_new_camera(
                video_gen, cam_ff[parent_cam.idx], p_shot.visual_desc,
                first.visual_desc, frames_dir, tag=f"cam{cam.idx}")
            if derived is not None and cam.is_parent_fully_covers_child \
                    and not cam.missing_info:
                shutil.copy(derived, out)      # 全覆盖 → 直接采用派生帧
                cam_ff[cam.idx] = out
                decisions.append({"stage": "camera_ff", "label": str(cam.idx),
                                  "via": "derived_copy"})
                return
            if derived is not None:
                available = available + [(str(derived), NEW_CAMERA_NOTE.format(
                    missing_info=cam.missing_info or "characters"))]
        _make_frame(available, first.ff_desc, out, f"cam{cam.idx}_ff")
        cam_ff[cam.idx] = out
        decisions.append({"stage": "camera_ff", "label": str(cam.idx),
                          "via": ("derived_edit" if cam.parent_cam_idx
                                  is not None else "root_gen")})

    for cam in cameras:
        _emit_camera_ff(cam)

    # ── 逐镜帧:非首镜 ff;variation≥medium 的 lf ──
    frame_of: dict = {}
    for cam in cameras:
        first_idx = cam.active_shot_idxs[0]
        cam_pair = (str(cam_ff[cam.idx]),
                    shots_by_idx[first_idx].ff_desc)
        frame_of[(first_idx, "ff")] = cam_ff[cam.idx]
        for si in cam.active_shot_idxs:
            s = shots_by_idx[si]
            if si != first_idx:
                out = frames_dir / f"shot{si:03d}_ff.png"
                avail = registry_pairs(registry, s.ff_chars) + [cam_pair]
                frame_of[(si, "ff")] = _make_frame(
                    avail, s.ff_desc, out, f"shot{si}_ff")
            if s.variation in ("medium", "large") and s.lf_desc:
                out = frames_dir / f"shot{si:03d}_lf.png"
                avail = registry_pairs(registry, s.lf_chars) + [cam_pair]
                frame_of[(si, "lf")] = _make_frame(
                    avail, s.lf_desc, out, f"shot{si}_lf")

    # ── 逐镜视频(可灵)+ 音频法 + 评审重掷 ──
    clips: list = []
    for s in shots:
        outp = cache_dir / f"shot{s.idx:03d}" / "video.mp4"
        outp.parent.mkdir(exist_ok=True)
        ff = frame_of[(s.idx, "ff")]
        lf = frame_of.get((s.idx, "lf"))
        # 运动 prompt:描述(剥标记)+ 台词逐字 + 压制句;旁白剥除
        motion = _strip_markers(s.visual_desc)
        motion = _NARRATION_RE.sub("", motion).strip()
        refs = []
        tokmap = {}
        if portraits_as_refer and lf is None:
            for j, n in enumerate(s.ff_chars):
                p = storyboard.portraits.get(n)
                if p and Path(p).exists():
                    refs.append(Path(p))
                    tok = (video_gen.ref_token(j + 1)
                           if hasattr(video_gen, "ref_token")
                           else f"<<<image_{j + 1}>>>")
                    tokmap[n] = tok
        if tokmap:
            motion = _names_to_tokens(motion, tokmap)
        if s.dialogue and s.dialogue not in motion:
            spk = tokmap.get(s.speaker, s.speaker or "")
            motion += ((f' {spk}说:"{s.dialogue}"。' if zh
                        else f' {spk} says: "{s.dialogue}".'))
        clause = _audio_clause(s, zh) if enable_audio else ""
        if clause and "无背景音乐" not in motion \
                and "no background music" not in motion:
            motion = motion.rstrip() + " " + clause.strip()
        if prompt_enhancer is not None:
            try:
                enhanced = prompt_enhancer.run(
                    _strip_markers(s.visual_desc), strategy="i2v_first",
                    conditions=[{"kind": "state", "role": "prompt_language",
                                 "description": prompt_lang}],
                    base_prompt=motion, label=f"shot {s.idx}")
                if enhanced:
                    from ..pipeline.window_loop import _is_mostly_chinese
                    if not zh or _is_mostly_chinese(enhanced):
                        motion = enhanced
            except Exception as exc:
                log.warning("cinegraph: enhancer failed (%s) — draft kept",
                            exc)
        want_audio = bool(enable_audio and (s.dialogue or s.sounds))
        best = None
        for turn in range(max(1, max_turns)):
            vout = outp if turn == 0 else \
                outp.with_name(f"video_r{turn}.mp4")
            if not vout.exists():
                old_a = getattr(video_gen, "generate_audio", False)
                video_gen.generate_audio = want_audio
                try:
                    if lf is not None:
                        video_gen.frame_to_frame(
                            prompt=motion, first_frame=ff, last_frame=lf,
                            out_path=vout, duration=s.duration or 5,
                            seed=turn)
                    else:
                        video_gen.generate(
                            motion, s.duration, vout, fps=fps, seed=turn,
                            first_frame=ff,
                            reference_images=(refs or None))
                finally:
                    video_gen.generate_audio = old_a
            cand = CandidateClip(shot_idx=s.idx, video_path=Path(vout),
                                 revision=turn)
            score = 1.0
            if board is not None:
                spec = ShotSpec(shot_idx=s.idx, duration=s.duration,
                                prompt=_strip_markers(s.visual_desc))
                try:
                    board.review(cand, spec, asset_memory, fps)
                    score = float(cand.metric_scores.get(
                        "weighted_total", 0.0))
                except Exception as exc:
                    log.warning("cinegraph: review failed (%s) — "
                                "accepting unreviewed", exc)
            if best is None or score > best[0]:
                best = (score, cand)
            if score >= accept_bar:
                break
            log.warning("cinegraph: shot %d turn %d score %.3f < %.2f — "
                        "re-rolling", s.idx, turn, score, accept_bar)
        clips.append(best[1].video_path)
        decisions.append({"stage": "shot_video", "label": f"shot {s.idx}",
                          "strategy": ("flf2v" if lf is not None
                                       else "i2v_ff"),
                          "reason": f"score={best[0]:.3f}"})
        log.info("cinegraph: shot %d done (score %.3f)", s.idx, best[0])

    # ── 拼装(硬切;构图一致性由机位树保证)──
    state["shots"] = [vars(s) for s in shots]
    (cache_dir / "cinegraph_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=1, default=str))
    movie = None
    if clips:
        concat_in = normalize_for_concat(clips, cache_dir / "concat_norm") \
            if any_audio(clips) else clips
        movie = VideoConcatTool().run(concat_in, cache_dir / "movie.mp4")
        log.info("cinegraph: movie assembled → %s", movie)

    class _Res:
        pass
    res = _Res()
    res.decisions = decisions
    res.storyboard = storyboard
    res.movie_path = movie
    res.cameras = cameras
    return res
