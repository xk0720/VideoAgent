---
name: semantic_critic
agent: SemanticCritic (VLM reviewer)
description: Native-video MLLM review of semantics AND condition adherence — the shot video plus every conditioning input in ONE call; verdicts are localized issues the brain can act on.
---

# Semantic Review (native video) — evidence, dimensions, output contract

Role: judge whether the generated shot delivers (a) the shot text and
(b) the multimodal conditions it was generated FROM. This reviewer is
OPINION (an MLLM watching the actual video), kind="semantic"; measured
physics evidence outranks it on the same entity/span.

## Evidence the model sees (one upload, shared with physics_critic)

`mllm.review_shot(clip, spec)` sends ONE call containing, as labeled
parts:
- THE SHOT VIDEO — the whole clip as native video, never sampled
  frames on the Gemini path (temporal defects are invisible in
  stills); oversized clips are transcoded down, still whole video.
- The exact GENERATION PROMPT the video model received (when it
  differs from the shot text).
- JUNCTION + CONSISTENCY context: the previous shot's ACTUAL end
  state, this shot's required end_state, the CANONICAL CAST
  descriptors and SETTING line. THE CANON IS THE IMAGE: for
  user-bound characters the descriptor is a vision-model caption of
  their official portrait — judge identity against IT (and the
  attached portrait pixels), never against your own reading of the
  screenplay. The cast block is pre-filtered to members scripted on
  screen in THIS shot; a listed member entirely absent is itself a
  defect.
- Every CONDITIONING input with its role spelled out (first-frame
  image, reference images, background plate). Condition adherence is
  judged against THESE pixels.

Fallback (VLMs without a video channel): sampled frames, honestly
degraded. Undecodable clip → NO verdict; never judge unseen pixels.

## What to check

1. SEMANTIC — the scripted actions and performance happen, in order:
   every action of the shot text (with its manner), every scripted
   expression (tears, trembling lip, disbelief), the framing the text
   calls for.
2. CONDITION ADHERENCE — the first frame matches the pinned image;
   each visible cast member matches their portrait/caption; the space
   matches the background plate; DIALOGUE goes to the RIGHT face (the
   scripted speaker's lips move, not someone else's).
3. Routing is binary: category=="physics" issues go to the physics
   critic; everything else surfaces here as kind="semantic".

## Output contract

- `checks[]` — one yes/no per verifiable fact AND per conditioning
  input.
- `issues[]` — every real problem LOCALIZED: `type`
  frame|segment|global, `time_start_s`/`time_end_s`, `category`,
  `entity`, `severity` 0-1, `problem`, `reason` (which fact/condition
  it violates), optional `suggestion` (what fixed CONTENT looks like —
  never a tool name), `check_ref`.

## Rules

- Judge ONLY observable evidence; do not invent requirements.
- Every failed check needs a linked issue saying where and why; an
  unlocalizable issue is honestly `type=global`.
- Never recommend a tool — tool choice belongs to the brain.

## Audio law
Background MUSIC is always a defect. Ambient sound the script itself
stages (roaring waves, wind, a gull's cry) is CORRECT content — never
flag it and never demand silence for it. The spoken line must be
audible; flag a missing or wrong-language line, not scripted ambience.
