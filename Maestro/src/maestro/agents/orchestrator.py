"""OrchestratorAgent — the BRAIN of the agentic repair loop (v0.4).

UniVA / NEWTON hand "review verdict → repair" to an ephemeral Act-LLM that
re-decides a tool every run with NO grounding: it sees a textual goal, not a
MEASURED review, and nothing rejects a regression. Maestro keeps the LLM in the
driver's seat (genuine function-calling over a tool registry) but GROUNDS it on
two hard rails:

  • the structured review the clip just received — the SEMANTIC failed items
    (question / fix_instruction / kind) + the MEASURED/JUDGED physics verdicts
    (mode / severity / frame_range / suggested_intervention / source) + the
    metric suite — so the brain reads evidence, not a vibe;
  • the monotonic Verifier ("brain proposes, gate disposes"): whatever tool the
    brain calls, the result only sticks if `verifier.is_better` accepts it. A
    rejected action is fed back to the brain next turn so it does NOT repeat it.

The deterministic `RepairRouter` (agents/repair_router.py) is the SAFETY-NET:
when the brain returns garbage / an out-of-menu tool / a tool whose capability
or asset is missing, `decide` returns the `INVALID` sentinel and the loop falls
back to the router's one deterministic action — so the loop never stalls.

The brain talks through `BaseLLMClient.complete(prompt) -> str` and STRICT JSON
only (provider-agnostic; no OpenAI/Anthropic SDK dep) — the reply is parsed with
the SAME tolerant `_extract_json` the real VLM critics use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..critics.board import ReviewBoard
from ..logging_utils import brain_log, get_logger
from ..models.image_edit import BaseImageEditClient
from ..models.llm import BaseLLMClient, MockLLMClient
from ..models.mllm_backends import _extract_json  # reuse the exact JSON extractor
from ..types import CandidateClip, ShotSpec

log = get_logger(__name__)

# Sentinel `decide` returns when the brain's reply is unusable (unparseable,
# out-of-menu tool, or a tool whose capability/asset is missing): the caller
# falls back to the deterministic RepairRouter for this turn.
INVALID = {"tool": "__invalid__", "args": {}, "reason": "brain reply unusable"}


class OrchestratorAgent:
    """The LLM orchestrator: reads the review, decides ONE tool call, executes it.

    Tools (executable handles, all optional except generator):
      generator   — GeneratorAgent.run (regenerate / keyframe-anchored re-gen)
      video_gen   — BaseVideoGenClient (edit_video / extend), capability-gated
      refiner     — RefinerAgent (kept for parity; the brain may consult its plan)
      image_edit  — BaseImageEditClient (keyframe local edit)
      retrieval   — RetrievalTool (retrieve_source_shots), asset-gated
    """

    def __init__(
        self,
        llm: Optional[BaseLLMClient] = None,
        *,
        generator=None,
        video_gen=None,
        refiner=None,
        image_edit: Optional[BaseImageEditClient] = None,
        retrieval=None,
        skill_library=None,
        sim_client=None,
        max_turns: int = 4,
        logger=None,
    ):
        self.llm = llm or MockLLMClient()
        self.generator = generator
        # Optional physics-sim backend (GenesisSimClient, NEWTON-style): when
        # wired AND the video backend has the "ref_video" channel, the brain
        # gains `simulate_reference` — simulate a physics-correct reference
        # clip and regenerate conditioned on it.
        self.sim_client = sim_client
        # Optional SkillLibrary: when set, decide() RETRIEVES a learned repair
        # workflow (retrieve_repair) and replays it BEFORE re-reasoning with the
        # LLM — UniVA's repair workflows are hand-coded; ours are learned skills.
        self.skill_library = skill_library
        # The brain's video tools come from the generator's backend by default
        # (single source of truth for capabilities), but allow an override.
        self.video_gen = video_gen or (generator.video_gen if generator else None)
        self.refiner = refiner
        self.image_edit = image_edit
        self.retrieval = retrieval
        self.max_turns = int(max_turns)
        self.logger = logger
        # The brain's instructions live in its predefined skill file
        # (skills/brain_skills/orchestrator.md) — like UniVA's plan.txt but for
        # repair: role, review reading, tool-by-modality logic, output format.
        # Falls back to a terse inline default if the file is missing.
        self._skill_prompt = self._load_skill_prompt()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _log(self, action: str, action_input: dict, observation: dict) -> None:
        if self.logger:
            self.logger.append(self.name, action, action_input, observation)

    # ─────────────────────────────────────────────────────────────────────
    # Tool MENU — the brain's registry, gated by real capabilities + assets.
    # ─────────────────────────────────────────────────────────────────────
    def available_actions(self, video_gen=None, asset_memory=None) -> list[dict]:
        """The tool menu as JSON-schema-ish dicts {name, description, args}.

        `regenerate`, `keyframe_edit` and `accept` are ALWAYS offered.
        `edit_clip` / `extend_clip` appear only if the backend declares the
        matching capability; `retrieve_replace` only if AssetMemory holds
        source shots. This is the brain's tool registry — it can only call
        what is actually executable, so an honest decision is always possible.
        """
        vg = video_gen or self.video_gen
        caps = vg.capabilities() if vg is not None else set()
        has_shots = bool(asset_memory and getattr(asset_memory, "video_shots", None))

        menu: list[dict] = [
            {
                "name": "regenerate",
                "description": "Re-generate the whole shot from the original prompt "
                "plus an anti-defect hint appended to the generation prompt. The "
                "broadest fix; use when the defect is global or no narrower tool fits.",
                "args": {"hint": "str — anti-defect instruction to append"},
            },
            {
                "name": "keyframe_edit",
                "description": "Locally edit ONE keyframe image, then re-generate "
                "anchored on the edited frame. Cheapest; use for a localized visual "
                "defect at a specific keyframe.",
                "args": {
                    "keyframe_idx": "int — index into the clip's keyframes",
                    "edit_instruction": "str — what to change in that frame",
                },
            },
        ]
        # ── LOCALIZED, propagated tools (v0.4) ──────────────────────────────
        # All three route through pipeline.timeline.propagate_repair: repair the
        # segment the defect overlaps, then re-anchor downstream segments until
        # continuity reconverges. They are offered whenever ANY video capability
        # exists (the timeline degrades to a whole-clip no-op when the clip has
        # no decodable frames or ffmpeg is missing — execute() guards that).
        if vg is not None and caps:
            menu.append({
                "name": "regenerate_segment",
                "description": "THE frame-precise repair: physically CUTS the "
                "clip at the defect's frame range, re-generates only that span "
                "anchored on the good boundary frames, then re-anchors "
                "downstream until continuity converges. The ONLY family of "
                "tools that truly consumes a frame range (the cut is by frame). "
                "PREFER this for any span-localized defect — including a "
                "defect at the END of the clip (cut before it, regrow the "
                "tail from the last good frame; never extend_clip there).",
                "args": {
                    "frame_start": "int — defect span start frame",
                    "frame_end": "int — defect span end frame",
                    "hint": "str — anti-defect instruction for the regenerated span",
                },
            })
            menu.append({
                "name": "keyframe_edit_propagate",
                "description": "Edit the corrected keyframe at a frame, re-generate "
                "the segment anchored on it, then propagate forward. Use when one "
                "frame's CONTENT is wrong and the fix should ripple downstream.",
                "args": {
                    "frame_idx": "int — frame whose segment to repair",
                    "edit_instruction": "str — what to correct in that frame",
                },
            })
            if "flf2v" in caps:
                menu.append({
                    "name": "frame_to_frame",
                    "description": "Re-generate a segment double-anchored on its "
                    "boundary frames (flf2v), then propagate forward. Strongest "
                    "continuity for a MOTION defect on a specific frame range.",
                    "args": {
                        "frame_start": "int — defect span start frame",
                        "frame_end": "int — defect span end frame",
                        "hint": "str — corrected-motion instruction",
                    },
                })
        if "edit" in caps and self.video_gen is not None:
            menu.append({
                "name": "edit_clip",
                "description": "Whole-clip prompt-guided re-render (edit model). "
                "It has NO frame addressing: NEVER write frame numbers into the "
                "prompt — describe the EVENT MOMENT ('as the glass leaves the "
                "table edge...') plus a DETAILED correction (subject, scene, "
                "what was wrong, what correct motion looks like, what must stay "
                "unchanged). Use for GLOBAL motion/look problems; for a "
                "span-localized defect prefer regenerate_segment/frame_to_frame.",
                "args": {
                    "prompt": "str — edit instruction describing the corrected motion",
                    "backend": "str — 'seedance' (default, best free-form) | "
                               "'runway' (free-form) | 'vace' (structure-guided)",
                },
            })
        if "depth" in caps and self.video_gen is not None:
            menu.append({
                "name": "depth_edit",
                "description": "Depth-guided foreground/background edit (VACE). "
                "Suitable for a BACKGROUND or FOREGROUND defect — the wrong scene "
                "behind/in front of the subject — while preserving the rest of the "
                "motion and layout.",
                "args": {"prompt": "str — what the corrected fg/bg should look like"},
            })
        if "style" in caps and self.video_gen is not None:
            menu.append({
                "name": "style_edit",
                "description": "Artistic style transfer (runway). Suitable for a "
                "STYLE defect — wrong palette / texture / look — keeping the same "
                "content and motion, re-rendered in the requested style.",
                "args": {"prompt": "str — the target artistic style to render in"},
            })
        if "extend" in caps and self.video_gen is not None:
            menu.append({
                "name": "extend_clip",
                "description": "Continue the clip FROM ITS CURRENT ENDING. Only "
                "for a clip that is genuinely incomplete but whose ending is "
                "GOOD. If the ending itself is the defect (e.g. the subject "
                "vanished in the final frames), extending REPRODUCES the broken "
                "state — use regenerate_segment on the tail instead.",
                "args": {"prompt": "str — what should happen in the continuation"},
            })
        if self.sim_client is not None and "ref_video" in caps:
            menu.append({
                "name": "simulate_reference",
                "description": "Run a RIGID-BODY physics simulation of the "
                "scene (you write the scene_spec), producing a physics-CORRECT "
                "reference video; the whole shot is then re-generated "
                "conditioned on that reference motion. Strongest fix for a "
                "MEASURED physics violation (gravity_inertia / collision / "
                "conservation) that survived other repairs — the generator is "
                "SHOWN the correct motion instead of being told about it.",
                "args": {
                    "scene_spec": "dict — {objects: [{id, type: sphere|box|"
                                  "cylinder, pos [x,y,z] m, radius|size, "
                                  "init_velocity [vx,vy,vz] m/s, fixed: bool}],"
                                  " gravity: [0,0,-9.81]}; floor at z=0 is "
                                  "implicit; SI units; RIGID bodies only",
                    "hint": "str — anti-defect instruction for the regeneration",
                },
            })
        if has_shots and self.retrieval is not None:
            menu.append({
                "name": "retrieve_replace",
                "description": "Replace the clip with a matching REAL source shot "
                "from uploaded footage. Best for a semantic 'missing element' "
                "defect that real footage can ground.",
                "args": {"query": "str — what to retrieve from the asset memory"},
            })
        menu.append({
            "name": "accept",
            "description": "Stop repairing and accept the current clip. Choose this "
            "only when no tool is likely to strictly improve it.",
            "args": {},
        })
        return menu

    # ─────────────────────────────────────────────────────────────────────
    # DECIDE — build the grounded prompt, call the brain, parse + validate.
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _review_payload(clip: CandidateClip) -> dict:
        """Serialize the structured review the brain must reason over."""
        return {
            "failed_items": [
                {"question": it.question, "kind": it.kind,
                 "fix_instruction": it.fix_instruction, "mode": it.mode}
                for it in clip.checklist.failed_items
            ],
            "physics_verdicts": [
                {"mode": v.mode.value, "severity": round(v.severity, 3),
                 "frame_range": list(v.frame_range),
                 "suggested_intervention": v.suggested_intervention,
                 "source": v.source}
                for v in clip.physics_verdicts
            ],
            "metric_scores": {k: round(float(val), 4)
                              for k, val in clip.metric_scores.items()},
        }

    # Terse fallback if prompts/orchestrator.txt is missing (keeps the brain
    # functional even without the file). The file is the source of truth.
    _INLINE_SKILL = (
        "You are a video-repair orchestrator. Read the review, pick ONE tool from "
        "`tools` that best fixes the WORST localized defect (by its fix_modality), "
        "prefer a localized tool over a whole reroll, never repeat a REJECTED "
        "action (see history), and do not accept while defects remain. Respond "
        'STRICT JSON only: {"tool": str, "args": {...}, "reason": str}.'
    )

    def _load_skill_prompt(self) -> str:
        from .base import load_prompt
        from ..skills.loader import load_skill

        skill = load_skill("orchestrator")
        if skill and skill["body"].strip():
            log.info("brain skill LOADED: orchestrator (%d chars) → goes "
                     "into every repair decide prompt", len(skill["body"]))
            return skill["body"]
        # legacy location, then the terse inline fallback
        path = Path(__file__).resolve().parent.parent / "prompts" / "orchestrator.txt"
        legacy = load_prompt(path)
        log.warning("orchestrator SKILL.md missing/empty — repair brain falls "
                    "back to %s",
                    f"prompts/orchestrator.txt ({len(legacy)} chars)"
                    if legacy else "the terse INLINE instruction")
        return legacy or self._INLINE_SKILL

    def _build_prompt(self, clip, spec, menu, history, defect_report=None,
                      review_brief=None) -> str:
        user = {
            "shot_prompt": spec.prompt,
            # A consolidated, prioritized brief produced by the review-summarizer
            # (when wired); empty until then — the brain still has the raw review
            # + localized defects below.
            "review_brief": review_brief or "",
            "localized_defects": (defect_report.to_brain_json()
                                  if defect_report is not None else []),
            "review": self._review_payload(clip),
            "tools": menu,
            "history": [
                {"tool": d.get("tool"), "args": d.get("args", {}),
                 "outcome": outcome, "new_total": new_total,
                 # 盲测评委指出的具体问题(有则给):被拒时它说明差在哪,
                 # brain 下一步据此换打法而不是瞎试。
                 **({"verifier_issues": d["verifier_issues"]}
                    if d.get("verifier_issues") else {})}
                for (d, outcome, new_total) in history
            ],
        }
        return (
            self._skill_prompt
            + "\n\nTHIS TURN (JSON):\n"
            + json.dumps(user, ensure_ascii=False, indent=2)
            + '\n\nRespond with STRICT JSON only: {"tool": ..., "args": {...}, "reason": ...}'
        )

    # ─────────────────────────────────────────────────────────────────────
    # RETRIEVE-FIRST skill execution path (the headline connection).
    # Before re-reasoning, replay a LEARNED, VERIFIED repair workflow if one
    # matches this defect — Voyager-style reuse, but the workflow was distilled
    # from a real convergence and admitted by "skill CI" (UniVA's repair
    # workflows are hand-coded; ours are retrieved learned skills).
    # ─────────────────────────────────────────────────────────────────────
    def _skill_step(self, defect_report, menu, history) -> Optional[dict]:
        """If a repair skill matches the defect, return its NEXT untried
        workflow step as a decision dict tagged via="skill"; else None.

        "Next untried" = the first step whose tool is in the current menu and
        that has NOT already been replayed-and-rejected this episode (we scan
        `history` for prior via="skill" decisions of THIS skill and advance past
        them). When every step is exhausted/rejected we return None so the loop
        falls to full-palette LLM reasoning."""
        if self.skill_library is None or defect_report is None:
            return None
        skill = self.skill_library.retrieve_repair(defect_report)
        if skill is None or not skill.repair_workflow:
            return None
        valid_names = {m["name"] for m in menu}
        # How many of THIS skill's steps have already been replayed this episode.
        tried = sum(
            1 for (d, _outcome, _total) in history
            if isinstance(d, dict) and d.get("via") == "skill"
            and d.get("skill_id") == skill.skill_id
        )
        for idx in range(tried, len(skill.repair_workflow)):
            step = skill.repair_workflow[idx] or {}
            tool = str(step.get("tool", ""))
            if tool not in valid_names:
                continue   # menu no longer offers this tool (cap/asset gone)
            args = dict(step.get("args_template") or {})
            return {
                "tool": tool,
                "args": args,
                "reason": f"replaying learned repair workflow '{skill.name}' "
                          f"step {idx + 1}/{len(skill.repair_workflow)}",
                "via": "skill",
                "skill_id": skill.skill_id,
                "step_idx": idx,
            }
        return None

    def decide(self, clip, spec, menu, history, defect_report=None,
               review_brief=None) -> dict:
        """Decide ONE tool call. RETRIEVE-FIRST: if a learned repair workflow
        matches this defect, replay its next untried step (via="skill") instead
        of re-reasoning. Only when no skill matches OR its steps are exhausted do
        we call the LLM for fresh full-palette reasoning (via="llm").

        `review_brief` is the ReviewSummarizerAgent's consolidated dict (ranked
        issues + provenance + progress + do_not_repeat) — placed first in the
        prompt per the skill file's "read this first".

        Returns the validated decision dict {tool, args, reason, via, ...}, or the
        INVALID sentinel (unparseable reply / tool not in the menu) so the caller
        falls back to the deterministic RepairRouter."""
        skill_decision = self._skill_step(defect_report, menu, history)
        if skill_decision is not None:
            log.info("orchestrator REPLAY learned repair skill=%s step=%d tool=%s",
                     skill_decision["skill_id"], skill_decision["step_idx"],
                     skill_decision["tool"])
            brain_log("repair/decide", {
                "shot_idx": spec.shot_idx, "via": "skill",
                "parsed": {k: skill_decision[k]
                           for k in ("tool", "args", "skill_id", "step_idx")
                           if k in skill_decision},
                "usable": True})
            self._log("decide", {"shot_idx": spec.shot_idx},
                      {"valid": True, "via": "skill",
                       "skill_id": skill_decision["skill_id"],
                       "tool": skill_decision["tool"]})
            return skill_decision

        prompt = self._build_prompt(clip, spec, menu, history, defect_report,
                                    review_brief=review_brief)
        reply = self.llm.complete(prompt)
        data = _extract_json(reply)

        valid_names = {m["name"] for m in menu}
        if not isinstance(data, dict) or str(data.get("tool", "")) not in valid_names:
            log.info("orchestrator brain reply invalid/out-of-menu → fallback "
                     "(reply=%.120r)", reply)
            brain_log("repair/decide", {
                "shot_idx": spec.shot_idx, "menu": sorted(valid_names),
                "raw": reply, "parsed": data if isinstance(data, dict) else None,
                "usable": False, "skill": "orchestrator",
                "skill_chars": len(self._skill_prompt)})
            self._log("decide", {"shot_idx": spec.shot_idx},
                      {"valid": False, "raw": (reply or "")[:200]})
            return dict(INVALID)

        decision = {
            "tool": str(data["tool"]),
            "args": data.get("args", {}) if isinstance(data.get("args"), dict) else {},
            "reason": str(data.get("reason", "")),
            "via": "llm",   # fresh full-palette reasoning (no skill matched)
        }
        log.info("orchestrator brain decided tool=%s reason=%s",
                 decision["tool"], decision["reason"])
        # debug 日志(2026-07-14 用户令):修复 brain 的 tool call 原文全量留底
        brain_log("repair/decide", {
            "shot_idx": spec.shot_idx, "menu": sorted(valid_names),
            "raw": reply, "parsed": dict(decision), "usable": True,
            "skill": "orchestrator", "skill_chars": len(self._skill_prompt)})
        self._log("decide", {"shot_idx": spec.shot_idx},
                  {"valid": True, "via": "llm", "tool": decision["tool"],
                   "args": decision["args"], "reason": decision["reason"]})
        return decision

    # ─────────────────────────────────────────────────────────────────────
    # LOCALIZED PROPAGATED tools — build a timeline from `best`, match the
    # brain's frame range to a Defect, run propagate_repair, wrap the result.
    # Returns None on any degrade (mock clip / no ffmpeg / bad args) so the
    # caller treats it as a no-op.
    # ─────────────────────────────────────────────────────────────────────
    def _execute_propagated(
        self, tool, args, best, spec, cache_dir, r, fps
    ) -> Optional[CandidateClip]:
        from ..pipeline.timeline import ClipTimeline, propagate_repair
        from .defect_report import Defect, build_defect_report

        cache_dir = Path(cache_dir)
        vg = self.video_gen
        if vg is None:
            return None

        # Resolve the target frame range + a fix modality for the chosen tool.
        if tool == "keyframe_edit_propagate":
            try:
                idx = int(args.get("frame_idx", 0))
            except (TypeError, ValueError):
                return None
            frame_range = (idx, idx + 1)
            modality = "content"
            hint = str(args.get("edit_instruction", "")) or "correct this frame"
            image_edit = self.image_edit
        else:
            try:
                fs = int(args.get("frame_start", 0))
                fe = int(args.get("frame_end", 0))
            except (TypeError, ValueError):
                return None
            if fe <= fs:
                fe = fs + 1
            frame_range = (fs, fe)
            modality = "motion"
            hint = str(args.get("hint", "")) or "regenerate this span faithfully"
            image_edit = None

        # Prefer the matching real Defect (carries the right modality/entity);
        # fall back to a synthetic one built from the brain's args.
        report = build_defect_report(best, spec, fps)
        defect = None
        for d in report.sorted_by_severity():
            lo, hi = d.frame_range
            if min(hi, frame_range[1]) > max(lo, frame_range[0]):  # overlap
                defect = d
                break
        if defect is None:
            defect = Defect(kind="physics", entity="", frame_range=frame_range,
                            severity=0.5, fix_modality=modality, note=tool)

        prop_dir = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_{tool}"
        timeline = ClipTimeline.from_clip(
            best, prop_dir, n_segments=3,
            duration_s=float(getattr(spec, "duration", 0.0) or 0.0),
        )
        spliced = propagate_repair(
            timeline, defect, video_gen=vg, image_edit=image_edit,
            hint=hint, cache_dir=prop_dir,
        )
        if spliced is None:
            return None
        return CandidateClip(shot_idx=spec.shot_idx, video_path=Path(spliced),
                             revision=r)

    # ─────────────────────────────────────────────────────────────────────
    # EXECUTE — map the chosen tool to the real call; review the output.
    # ─────────────────────────────────────────────────────────────────────
    def execute(
        self, decision, best, spec, cache_dir, r, board: ReviewBoard,
        asset_memory=None, fps: int = 8,
    ) -> Optional[CandidateClip]:
        """Run the brain's chosen tool, wrap the output in a fresh reviewed
        CandidateClip, and RETURN it (the loop's Verifier decides accept).

        `accept` → None (brain chose to stop). Any guard failure (missing
        capability/asset/keyframe) returns None too — the caller treats that as
        a no-op and falls through, never crashing on a malformed decision."""
        cache_dir = Path(cache_dir)
        tool = decision.get("tool")
        args = decision.get("args", {}) or {}

        if tool == "accept":
            return None

        if tool == "regenerate":
            hint = str(args.get("hint", "")) or "improve physical plausibility and "\
                "match the prompt's key elements"
            cand = self.generator.run(
                spec, cache_dir, revision=r, seed=300 + r,
                extra_prompt=hint, fps=fps,
            )

        elif tool == "keyframe_edit":
            try:
                idx = int(args.get("keyframe_idx", 0))
            except (TypeError, ValueError):
                return None
            if self.image_edit is None or not best.keyframes \
                    or not (0 <= idx < len(best.keyframes)):
                return None
            instruction = str(args.get("edit_instruction", "")) or "fix the defect"
            first_frame = self.image_edit.edit(
                best.keyframes[idx], instruction,
                cache_dir / f"shot{spec.shot_idx:03d}_r{r}_brain_editkf.txt",
            )
            cand = self.generator.run(
                spec, cache_dir, revision=r, seed=400 + r,
                extra_prompt=instruction, first_frame=first_frame, fps=fps,
            )

        elif tool == "edit_clip":
            if self.video_gen is None or "edit" not in self.video_gen.capabilities():
                return None
            out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_brain_edit.mp4"
            prompt = str(args.get("prompt", "")) or "correct the implausible motion"
            backend = str(args.get("backend", "seedance")) or "seedance"
            if backend not in ("seedance", "runway", "vace"):
                backend = "seedance"
            video_path = self.video_gen.edit_video(
                prompt=prompt, video_path=best.video_path, out_path=out,
                backend=backend,
            )
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)

        elif tool == "depth_edit":
            if self.video_gen is None or "depth" not in self.video_gen.capabilities():
                return None
            out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_brain_depth.mp4"
            prompt = str(args.get("prompt", "")) or "correct the foreground/background"
            video_path = self.video_gen.depth_modify(
                prompt=prompt, video_path=best.video_path, out_path=out,
            )
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)

        elif tool == "style_edit":
            if self.video_gen is None or "style" not in self.video_gen.capabilities():
                return None
            out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_brain_style.mp4"
            prompt = str(args.get("prompt", "")) or "match the intended visual style"
            video_path = self.video_gen.style_transfer(
                prompt=prompt, video_path=best.video_path, out_path=out,
            )
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)

        elif tool == "extend_clip":
            if self.video_gen is None or "extend" not in self.video_gen.capabilities():
                return None
            out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_brain_extend.mp4"
            prompt = str(args.get("prompt", "")) or "continue the shot; keep all "\
                "entities present on one continuous trajectory"
            video_path = self.video_gen.extend(
                prompt=prompt, video_path=best.video_path, out_path=out,
                duration=(None if spec.duration is None
                          else max(1, int(round(spec.duration)))),
            )
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)

        elif tool in ("regenerate_segment", "keyframe_edit_propagate", "frame_to_frame"):
            cand = self._execute_propagated(
                tool, args, best, spec, cache_dir, r, fps
            )
            if cand is None:
                return None

        elif tool == "simulate_reference":
            # NEWTON-style (arXiv:2605.18396): simulate the physics → a
            # physics-correct reference clip → regenerate conditioned on it
            # (seedance-2.0 reference_videos channel). Guarded like every
            # other tool: missing sim client / channel / bad spec → no-op.
            if (self.sim_client is None or self.video_gen is None
                    or "ref_video" not in self.video_gen.capabilities()):
                return None
            scene_spec = args.get("scene_spec")
            hint = str(args.get("hint", "")) or "follow the reference video's motion exactly"
            try:
                sim = self.sim_client.run(scene_spec)
            except (ValueError, TypeError) as exc:
                # invalid brain-written spec — log + no-op (the rejection lands
                # in history, so the brain can rewrite the spec next turn)
                log.info("simulate_reference: invalid scene_spec — %s", exc)
                self._log("execute",
                          {"shot_idx": spec.shot_idx, "tool": tool},
                          {"error": f"invalid scene_spec: {exc}"})
                return None
            out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_brain_simref.mp4"
            prompt = spec.prompt + ". " + hint
            if sim.get("scene_desc"):
                # ground truth from the SPEC (NEWTON's ref_video_desc): the
                # generator is told the true object count of the reference.
                prompt += f" (reference video: {sim['scene_desc']})"
            video_path = self.video_gen.generate(
                prompt=prompt, duration=spec.duration, out_path=out,
                fps=fps, seed=500 + r, reference_video=sim["video_path"],
            )
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path,
                                 revision=r)

        elif tool == "retrieve_replace":
            if self.retrieval is None:
                return None
            query = str(args.get("query", "")) or spec.prompt
            shot_ids = self.retrieval.retrieve_source_shots(query=query)
            src: Optional[Path] = None
            for sid in shot_ids:
                shot = self.retrieval.memory.video_shots.get(sid)
                if shot is None or not shot.source_video:
                    continue
                p = Path(shot.source_video)
                if p.exists():
                    src = p
                    break
            if src is None:
                return None
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=src, revision=r)

        else:  # unknown tool (should be caught in decide) — treat as no-op
            return None

        # 修复出的候选继承本 shot 的生成条件(评审要拿条件对照评贴合度)。
        if getattr(cand, "conditioning", None) is None:
            cand.conditioning = getattr(best, "conditioning", None)
        board.review(cand, spec, asset_memory, fps)
        self._log("execute",
                  {"shot_idx": spec.shot_idx, "tool": tool, "args": args},
                  {"video_path": str(cand.video_path),
                   "weighted_total": cand.metric_scores.get("weighted_total", 0.0)})
        return cand
