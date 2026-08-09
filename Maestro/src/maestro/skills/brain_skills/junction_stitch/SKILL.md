---
name: junction_stitch
agent: junction stitcher (transition-video composer, pipeline/window_loop.py)
description: Write the two shot descriptions for the junction transition video — a "Two shots ... cut to" clip whose post-cut frame becomes the next shot's opening frame. Strict JSON output.
---

# Junction Stitch — the two-shot descriptions

## What is being made

A short TRANSITION VIDEO with exactly one hard cut: the first shot
replays the tail of the previous clip; after the cut, the second shot
shows this junction from the NEW shot's camera. The frame right after
the cut is extracted and becomes the next shot's opening frame. Your
two descriptions steer that video. You write ONLY the descriptions —
the executor owns the "Two shots" skeleton, the reference slots and
their semantic lines; never restate them.

## first_shot_desc

1. GROUND IT IN REALITY: `prev_tail_report` is a vision model's
   reading of the previous clip's ACTUAL final seconds (camera_angle +
   per-character actions). The first shot restates that reality in one
   or two clauses — camera framing/position, each character's position
   and visible state. When the scripted end state contradicts the
   report, THE REPORT WINS (it is what was filmed).
2. It is a freeze-moment, not a story: no new action may begin here.

## second_shot_desc

3. FIRST GLIMPSE, NOT THE WHOLE SHOT: describe what the audience sees
   in the INSTANT after the cut — the new framing (shot size, camera
   position/height, angle relative to the subjects), who is in frame
   where, facing which way, mid-what-state. Do NOT narrate the shot's
   later actions, dialogue or ending; the main generation owns those.
4. SCRIPT FIDELITY: the glimpse must be the opening the current shot's
   script asks for — same subjects, same staging intent; distill it,
   never invent a different composition.
5. TOKEN LAW: every character is referenced ONLY by the slot token
   given in the slot table (e.g. <<<image_2>>>) — never a bare name,
   never an invented number. When a location slot is provided (the
   background changed), open with the location anchored to its token
   (e.g. "In the location shown in <<<image_4>>>, ...").
6. CONTINUITY OF WORLD: same time of day, same weather, same light,
   same wardrobe as the first shot — a cut changes the camera, never
   the world.

## Both descriptions

7. PICTURE LANGUAGE ONLY: plain visual prose. No narration lines, no
   sound-effect annotations, no scaffold labels, no camera jargon the
   video model cannot film ("match cut", "L-cut").
8. LANGUAGE LAW: write in the task's `prompt_language` (the values,
   not the keys).
9. Keep each description to 1-3 sentences; the transition video is
   ~5 seconds and overloaded descriptions smear both shots together.

## Output format (STRICT JSON — output this and nothing else)

{"first_shot_desc": "<1-3 sentences>",
 "second_shot_desc": "<1-3 sentences>"}
