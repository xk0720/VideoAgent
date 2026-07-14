---
name: semantic_critic
agent: SemanticCritic (VLM reviewer)
description: Native-video MLLM review of semantics AND condition adherence — the shot video plus every conditioning input (key images, reference video) in ONE call; verdicts are localized issues the brain can act on.
---

# Semantic Review (native video) — evidence, dimensions, output contract

Role: judge whether the generated shot delivers (a) the scene text and (b) the
multimodal conditions it was generated FROM. This reviewer is OPINION (an MLLM
watching the actual video) — tagged kind="semantic"; the summarizer weighs it
below measured physics evidence when they conflict on the same entity/span.

## Evidence the model sees (one upload, shared with physics_critic)

`mllm.review_shot(clip, spec)` (`models/mllm_backends.py`) sends ONE
generateContent call containing, as labeled parts:

- THE SHOT VIDEO — the WHOLE clip as native video (`inline_data`, video/mp4).
  NEVER sampled frames on the Gemini path: temporal defects (flicker, motion
  breaks, event order) are invisible to frame stills. Clips over the inline
  budget are transcoded down (360p) — still whole video, never frames.
- The GENERATION PROMPT the video model actually received.
- Every CONDITIONING input with its role spelled out: "FIRST-FRAME image (the
  shot must open on it)", "LAST-FRAME image (the shot must end on it)",
  "REFERENCE image N (identity/style anchor)", "REFERENCE VIDEO (motion /
  continuity source)". Condition adherence is judged against THESE pixels,
  not against the reviewer's imagination.

The merged package is cached per (path, mtime): semantic and physics critics
both read the SAME single call — one upload per review round (U6).

Fallback path (OpenAI-compatible VLMs without a video channel): sampled frames,
honestly degraded. Undecodable clip or a non-video stub → NO verdict; never
judge pixels that were never seen.

## Dimensions to cover (reference frame, not a straitjacket)

1. SEMANTIC — objects/counts/attributes/setting/action-order from the prompt
2. CONDITION ADHERENCE — does the first frame match the first-frame image; do
   subjects match reference images; does motion continue the reference video
3. TEXT/SIGNAGE — literal text the prompt requires
(physics/temporal/visual belong to the sibling critics but arrive in the same
merged call — route by `category`.)

## Output contract (what the brain must be able to act on)

- `checks[]` — one yes/no per verifiable fact AND per conditioning input.
- `issues[]` — the core deliverable: every real problem LOCALIZED:
  `type` frame|segment|global, `time_start_s`/`time_end_s`, `category`,
  `entity`, `severity` 0-1, `problem` (what is wrong), `reason` (why it is
  wrong — which fact/condition it violates), optional `suggestion` (what the
  CONTENT should look like when fixed — never a tool name), `check_ref`
  linking the issue to the check it fails.
- Seconds are converted to frames at the probed fps downstream; failed checks
  inherit frame_range + fix text from their linked issue.

## Rules

- Judge ONLY observable evidence; do not invent requirements not in the
  prompt or conditions.
- Every FAILED check needs a linked issue saying where and why; an issue that
  cannot be localized is honestly `type=global`, never a made-up span.
- Never recommend a TOOL — describing what is wrong (and what fixed content
  looks like) is the whole job; tool choice belongs to the brain.
