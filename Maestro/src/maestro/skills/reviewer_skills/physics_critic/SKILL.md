---
name: physics_critic
agent: PhysicsCritic (VLM reviewer)
description: MLLM opinion on physical plausibility — expected failure modes from the shot's physics annotation, judged from frames; complements (never overrides) the measured chain.
---

# Physics Review (VLM opinion) — scope, tool, output contract

Role: watch sampled frames and judge PLAUSIBILITY of the physics the shot's
annotation says to expect. This is the OPINION tier (`source="vlm"`): it covers
what the measured chain cannot (deformation, fluids, contact appearance,
occluded motion), and it is the fallback tier for entities whose tracks failed
certification. Where BOTH tiers speak on the same entity/span, the summarizer
merges them (cross-type confirmation) and the MEASURED severity wins conflicts.

## What to look for (the annotated failure modes)

- gravity_inertia    — floating, mid-air direction changes, wrong arcs
- collision          — interpenetration, missing rebound, broken contact order
- conservation       — energy/momentum appearing from nowhere
- object_permanence  — objects vanishing / duplicating / teleporting
- fluid / deformation — material behavior a rigid-track fit cannot measure

## Tool it calls

`mllm.assess_physics(clip, spec, fps)` (`models/mllm_backends.py`). On the
Gemini path this reads the MERGED native-video review (`review_shot`): the
WHOLE clip as native video plus the generation prompt and all conditioning
inputs, ONE upload shared with semantic_critic (U6 — a multimodal model
judging the full context also judges physics; no second call). Issues with
`category=physics` become verdicts here; `time_start_s/time_end_s` are
converted to frame ranges at the probed fps. Fallback path (VLMs without a
video channel): sampled frames + expected modes. Undecodable clip or a
non-video stub → NO verdict.

## Output contract

PhysicsVerdict{mode, severity, frame_range, source="vlm", entity when known}
+ a mirrored failed ChecklistItem (kind="physics"). The frame_range should be
as NARROW as the evidence allows — it drives localized segment repair.

## Rules

- Report what is VISIBLE, not what is likely; severity reflects confidence ×
  how wrong it looks.
- Never contradict a measured verdict casually: if the track math says the arc
  is consistent and it merely LOOKS odd, say so at low severity.
- Never recommend a tool.
