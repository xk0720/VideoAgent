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
from ..logging_utils import get_logger
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
        max_turns: int = 4,
        logger=None,
    ):
        self.llm = llm or MockLLMClient()
        self.generator = generator
        # The brain's video tools come from the generator's backend by default
        # (single source of truth for capabilities), but allow an override.
        self.video_gen = video_gen or (generator.video_gen if generator else None)
        self.refiner = refiner
        self.image_edit = image_edit
        self.retrieval = retrieval
        self.max_turns = int(max_turns)
        self.logger = logger

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
        if "edit" in caps and self.video_gen is not None:
            menu.append({
                "name": "edit_clip",
                "description": "Edit the rendered clip IN PLACE (preserves good "
                "parts; no full reroll). Best for a physics MOTION defect "
                "(gravity/inertia, collision, conservation, penetration).",
                "args": {
                    "prompt": "str — edit instruction describing the corrected motion",
                    "backend": "str — 'runway' (free-form) or 'vace' (structure-guided)",
                },
            })
        if "extend" in caps and self.video_gen is not None:
            menu.append({
                "name": "extend_clip",
                "description": "Continue the clip past its last frame. Best for an "
                "incomplete / too-short clip or an object_permanence defect.",
                "args": {"prompt": "str — what should happen in the continuation"},
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

    def _build_prompt(self, clip, spec, menu, history) -> str:
        system = (
            "You are a video-repair orchestrator. The clip was reviewed; here are "
            "its DEFECTS (semantic failed items + measured/judged physics verdicts "
            "with mode/severity/frame-range/intervention) and METRIC scores. Here "
            "are the TOOLS you may call. Choose ONE tool call that best fixes the "
            "WORST defect. A monotonic verifier will REJECT your result unless it "
            "strictly improves — if a prior action was rejected (see history), DO "
            "NOT repeat it; try a different tool or accept. Respond STRICT JSON "
            'only: {"tool": str, "args": {...}, "reason": str}.'
        )
        user = {
            "shot_prompt": spec.prompt,
            "review": self._review_payload(clip),
            "tools": menu,
            "history": [
                {"tool": d.get("tool"), "args": d.get("args", {}),
                 "outcome": outcome, "new_total": new_total}
                for (d, outcome, new_total) in history
            ],
        }
        return (
            system
            + "\n\nREVIEW + TOOLS + HISTORY (JSON):\n"
            + json.dumps(user, ensure_ascii=False, indent=2)
            + '\n\nRespond with STRICT JSON only: {"tool": ..., "args": {...}, "reason": ...}'
        )

    def decide(self, clip, spec, menu, history) -> dict:
        """Ask the brain for ONE tool call; validate against the menu.

        Returns the validated decision dict {tool, args, reason}, or the INVALID
        sentinel (unparseable reply / tool not in the menu) so the caller falls
        back to the deterministic RepairRouter."""
        prompt = self._build_prompt(clip, spec, menu, history)
        reply = self.llm.complete(prompt)
        data = _extract_json(reply)

        valid_names = {m["name"] for m in menu}
        if not isinstance(data, dict) or str(data.get("tool", "")) not in valid_names:
            log.info("orchestrator brain reply invalid/out-of-menu → fallback "
                     "(reply=%.120r)", reply)
            self._log("decide", {"shot_idx": spec.shot_idx},
                      {"valid": False, "raw": (reply or "")[:200]})
            return dict(INVALID)

        decision = {
            "tool": str(data["tool"]),
            "args": data.get("args", {}) if isinstance(data.get("args"), dict) else {},
            "reason": str(data.get("reason", "")),
        }
        log.info("orchestrator brain decided tool=%s reason=%s",
                 decision["tool"], decision["reason"])
        self._log("decide", {"shot_idx": spec.shot_idx},
                  {"valid": True, "tool": decision["tool"], "args": decision["args"],
                   "reason": decision["reason"]})
        return decision

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
            backend = str(args.get("backend", "runway")) or "runway"
            if backend not in ("runway", "vace"):
                backend = "runway"
            video_path = self.video_gen.edit_video(
                prompt=prompt, video_path=best.video_path, out_path=out,
                backend=backend,
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
                duration=max(1, int(round(spec.duration))),
            )
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)

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

        board.review(cand, spec, asset_memory, fps)
        self._log("execute",
                  {"shot_idx": spec.shot_idx, "tool": tool, "args": args},
                  {"video_path": str(cand.video_path),
                   "weighted_total": cand.metric_scores.get("weighted_total", 0.0)})
        return cand
