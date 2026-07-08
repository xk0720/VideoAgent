# NEWTON (arXiv:2605.18396) — Code-Level Analysis + What Maestro Borrowed (July 2026)

> Read in full: loop/run_loop.py (956 L), tools/video_verifier/core.py (398 L),
> tools/simulator/core.py (780 L), loop/memory.py, planner_skills/*.md.
> Repo: /Users/kevin/Desktop/Kevin/repositories/NEWTON.

## 1. What NEWTON actually is

Planner (gpt-5.5 function-calling) → Executer (tools) → Verifier loop.
Video generation is DEMOTED to one tool among many; the planner builds
CONDITIONING (sim reference video, keyframes, real photos, refined prompt) and
the loop re-plans on verifier feedback. Core paper claim: text prompts are a
LOSSY COMPRESSION of the physical world — no prompt wording can specify
dynamics, so SHOW the generator the correct motion instead.

## 2. The simulator is CONDITIONING, not a verifier

`tools/simulator/core.py`: planner-written scene_spec → Genesis scene (floor
implicit, snap-to-support fixes cm-level floating), 1.0 s physics @ dt 0.01,
physics-only PRE-PASS computes the swept AABB → camera framed to the WHOLE
motion, 72 frames rendered, played back over 3 s @ 24 fps (3x slow-mo).
Outputs: reference VIDEO (→ Seedance reference_videos channel = motion
conditioning) + per-object trajectories + text summary (→ planner). The sim
NEVER scores a generated video.

## 3. Verification = two VLM gates (no absolute scores anywhere)

1. `judge_condition` (PRE-generation): Gemini sees the baseline clip + the
   proposed conditioning package (sim video / keyframes / prompt) and rules
   whether it is sound enough to be WORTH a generation. Sim reference judged
   ONLY on motion/object count; appearance explicitly ignored ("the sim is
   abstract; the prompt supplies the real appearance"). Ground-truth object
   counts come from the SPEC (`ref_video_desc`), never from watching the clip.
   Rejected conditioning → planner revises, no money spent.
2. `verify_relative` (POST-generation): BLIND A/B vs a fixed text-only
   baseline — randomized order, judge never told which is which, signed score
   in [-10,+10], STOP at ≥ +5. The judge "does NOT recommend tools — that
   decision belongs to another agent."

## 4. Other mechanics worth noting

- Skills: planner_skills/*.md with YAML frontmatter; PROGRESSIVE DISCLOSURE —
  system prompt carries only name+description, the planner calls read_skill()
  before using a tool. Four solver skills teach scene_spec authoring.
- Conditioning CARRIES OVER turns (Staged.clone → committed): a tool call
  REPLACES its piece, not calling KEEPS it — amend incrementally, never
  rebuild. System prompt encodes the diagnosis rules ("sim passed pre-check
  but video still wrong → the sim is NOT the problem, add a stronger visual
  constraint").
- Only trainable part: the planner (Flow-GRPO) — IRRELEVANT to us
  (training-free mandate); the whole inference loop is training-free.

## 5. What Maestro borrowed (commit-level)

| Borrow | Where | Why |
|---|---|---|
| Pre-generation condition gate | `timeline._same_shot` gates the flf2v double-anchor (HSV mean-diff 27/255, PySceneDetect's only battle-tested threshold); dissimilar anchors → i2v fallback | flf2v models insert a CUT on dissimilar anchors (Kling-documented) — check before paying |
| Blind pairwise confirmation | `VerifierAgent(judge=…, margin=0.02)`: marginal metric wins must survive a bidirectional MLLM compare; judge can VETO a marginal win, never rescue a loss | weighted_total comes from noisy VLM absolute scores; NEWTON's stance: only relative, blind comparisons are trustworthy |
| Sim as repair conditioning | `physics/sim_backends.GenesisSimClient` (RIGID-only port) + brain tool `simulate_reference` (gated on sim client + "ref_video" cap) + `WaveSpeedClient.generate(reference_video=…)` → seedance-2.0 reference_videos | our MEASURED verdicts (law_verifier) tell the brain WHEN to simulate — NEWTON's planner has to guess; text can't specify dynamics, so show the motion |
| Ground truth from the spec | `describe_scene_spec` rides the regen prompt ("reference video: exactly 1 sphere…") | judges/generators miscount rendered clips; the spec IS the truth |

## 6. Explicitly NOT borrowed

- Fixed text-only baseline as the bar: our Verifier compares against the
  CURRENT BEST (a rising bar) — stronger for repair.
- Progressive skill disclosure via read_skill: our per-agent skill files are
  small enough to inline, and our LEARNED repair skills are auto-retrieved
  (stronger than hoping the LLM remembers to read).
- Particle solvers (SPH/MPM/PBD): v1 sim client is rigid-only — matches the
  law families our verifier measures; liquids degrade honestly (ValueError).

## 7. Beyond NEWTON (open idea, not implemented)

Sim trajectories as a VERIFICATION reference: we already extract pixel tracks
(CoTracker); comparing generated-video tracks against the sim's ground-truth
trajectory (shape/DTW distance) would be true "simulation as verifier" — a
step NEWTON itself never takes. Candidate C6 extension; needs a study of
camera-projection alignment first.
