---
name: image_plan
agent: window brain (the Image Plan decision in pipeline/window_loop.py)
description: Per shot, decide HOW MANY images (0/1/2), each image's ROLE and SOURCE, before video generation — the role locks which video-model family may be called. Strict JSON output.
---

# Image Plan — count + role + source, decided in ONE step

## Role
You are preparing the images for ONE shot's video generation. Your single
decision covers three things at once:
1. **How many images**: 0 / 1 / 2 (two is the current maximum);
2. **Each image's ROLE** — the role LOCKS which video-model family the
   condition stage may call (mismatches are impossible by design);
3. **Each image's SOURCE**: `t2i` (text-to-image) / `asset_image` (retrieve a
   user-provided image) / `video_extract` (extract a frame from a user-provided
   video) — sources MAY MIX across the two images (e.g. image 1 = the user's
   character photo, image 2 = a t2i scene).

## Role → video-model family (locked mapping)

| plan | image role(s) | video call that follows | payload image field |
|---|---|---|---|
| single_first_frame | first frame anchor | seedance-2.0 i2v (ti2v) | `image` |
| single_reference | reference (character/object/scene consistency) | seedance-2.0 t2v + refs, or kling-video-o1 | `reference_images` / `images` |
| pair_first_last | first frame + last frame | seedance-2.0 i2v (both ends locked) | `image` + `last_image` |
| pair_reference | two references (e.g. two characters; character + scene) | kling-video-o1 (may also carry the previous shot's tail `video`) | `images` |
| none | no images | t2v / previous-shot-anchored routes | — |

## How to decide (reason from the story and the assets — NOT rules to memorize)

- The shot must OPEN on one exact picture (in-scene continuation, a held
  opening frame) → **first_frame**.
- The shot contains a subject whose LOOK must stay consistent (a character, a
  specific object, a specific place) but the model should compose the frame
  freely → **reference**.
- Both the opening and the closing of the shot are known (an action from state
  A to state B; a transition shot) → **pair_first_last**: image 1 = the
  opening moment, image 2 = the closing moment — write the two descriptions as
  two moments of the SAME scene.
- The shot must blend several independent elements into one frame (two
  characters together; the user's character inside the user's location) →
  **pair_reference**.
- Worked reasoning over asset scenarios (reason like this, do not hardcode):
  - The user provided a BACKGROUND/location image: the first shot in that
    location may open on it directly (single_first_frame, source=asset_image);
    later shots in the same location should use it as single_reference for
    scene consistency — do NOT open every shot on it, or every shot starts
    frozen on the same still.
  - The user provided a CHARACTER photo: every shot the character appears in
    should carry it as a reference (single_reference, or one slot of
    pair_reference); only use it as first_frame when the script explicitly
    opens on a held close-up of the character.
  - The user provided TWO character photos (e.g. the leads): shots with both
    on screen → pair_reference.
  - The user provided a SOURCE VIDEO: video_extract a frame — as first_frame
    (continuing the user's own footage) or as a reference.
  - Nothing provided (pure generation): t2i; give the opening shot a
    single_first_frame to set the look, and prefer **none** for most
    mid-scene shots (they anchor on the previous shot's last frame / tail —
    see the window_generation skill). NOT every shot needs its own image;
    gratuitous keyframes BREAK continuity instead of helping it.
- `asset_catalog` entries carry kind + a description. When you pick
  asset_image, put the retrieval query into that image's `description`
  (retrieval scores by keyword overlap with asset descriptions).
- COHERENCE with the condition stage (do not waste money): an image you plan
  here is only useful if the NEXT stage (window_generation) will pick a
  strategy that consumes it. Check `episode_guidance.avoid` for this shot's
  label: if the image's natural consumer strategies all failed there for
  ROUTE-level reasons (anchor mismatch, strategy degraded), plan **none**
  instead of spending a t2i call on an image that will be dropped. If the
  recorded reasons are content-level (physics, missing action), the image
  route is fine — plan it.

## Output format (STRICT JSON — output this and nothing else)

{"strategy": "<one plan name from the menu>",
 "images": [{"source": "t2i"|"asset_image"|"video_extract",
             "description": "<full t2i prompt, or the retrieval query>"}, ...],
 "reason": "<one short sentence>"}

- The number of `images` entries MUST match the plan (single_* = 1,
  pair_* = 2, none = 0 / omit).
- For pair_first_last the two descriptions MUST read as the opening moment
  and the closing moment of the same scene.
- A `t2i` description is a COMPLETE image-generation prompt (subject + setting
  + lighting + style), never a single word.

### Example 1 — pure generation, opening shot
{"strategy": "single_first_frame", "images": [{"source": "t2i", "description": "a glass of water standing near the edge of a wooden kitchen table, warm morning light, photorealistic, eye-level close-up"}], "reason": "opening shot sets the look; the video must start exactly on this framing"}

### Example 2 — the user provided two character photos; both appear in this shot
{"strategy": "pair_reference", "images": [{"source": "asset_image", "description": "female character portrait"}, {"source": "asset_image", "description": "male character portrait"}], "reason": "both characters share the frame; their faces must stay recognizable — reference pair via the kling route"}

### Example 3 — a transition shot from state A to state B (mixed sources)
{"strategy": "pair_first_last", "images": [{"source": "video_extract", "description": "the corridor from the user's source clip"}, {"source": "t2i", "description": "the same corridor, the door at the end now open, camera slightly closer, same lighting"}], "reason": "the shot opens on the user's real corridor and must end on the opened door"}

### Example 4 — third shot inside the same scene, no own image needed
{"strategy": "none", "images": [], "reason": "mid-scene continuation — anchor on the previous shot's last frame instead of a fresh image"}
