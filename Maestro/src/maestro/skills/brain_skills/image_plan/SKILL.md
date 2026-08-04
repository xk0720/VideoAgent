---
name: image_plan
agent: window brain (§B' per-shot image decision in pipeline/window_loop.py)
description: The character-image skill — decide whether THIS shot needs its own image(s) and write portrait-grade t2i prompts when one must be generated. Strict JSON output.
---

# Image Plan — character images and per-shot frames

## Role
Two duties. (a) Portrait craft: when a character has no user image,
their official portrait is generated from a t2i prompt you write —
that portrait then anchors the character's identity in EVERY shot.
(b) Per-shot planning: decide from the menu whether this shot needs
its own image(s) beyond what rides automatically.

## What rides automatically (plan around it, never duplicate it)

- Official character portraits attach as reference images on every
  reference-carrying route.
- The scene's empty background plate attaches per the storyboard's
  `bg` id.
- In-scene continuation shots are hard-pinned to the previous shot's
  final frame at the API level.
With all three in place, a continuing shot almost always needs
NOTHING: choose "none" unless a concrete gap remains.

## Portrait prompt craft (when a portrait must be generated)

- The canon's `static:` half becomes the prompt verbatim — every
  color, garment and mark — plus: neutral standing pose, plain
  background, even lighting, front three-quarter view, full or
  half body.
- One character per image; no props unless the canon names them; no
  scene dressing (the portrait must not smuggle a location).
- ENGLISH, concrete visual words only.

## Role → video-model family lock

An image's ROLE decides the downstream generation family: a
first_frame image locks an i2v-style route; a first+last pair locks
flf2v; reference images ride the reference channel. Plan roles for the
route the shot actually needs — a wrong role locks the wrong family.

## Per-shot rules

1. NO TEXT-DRAWN FIRST FRAMES: when the shot's cast have official
   portraits, never plan a t2i first_frame image of them — a freshly
   drawn face contradicts the portrait and splits identity. Asset
   images (real user pixels) are exempt.
2. A planned image must close a REAL gap: a brand-new location's
   establishing look, a scripted prop the references don't show, a
   scene-cut opening that nothing pins. Name the gap in `reason`.
3. pair_first_last only when the script fixes BOTH boundary moments.
4. Every t2i description is a complete standalone prompt (subject +
   setting + lighting + style), ENGLISH, no character names — use the
   canon's visual words instead.

## Output format (STRICT JSON — output this and nothing else)

{"strategy": "<one plan name from the menu>",
 "images": [{"source": "t2i"|"asset_image"|"video_extract",
             "description": "<full t2i prompt, or the retrieval query>"}],
 "reason": "<one short sentence>"}

`images` count MUST match the plan (single_* = 1, pair_* = 2,
none = 0).
