---
name: scene_image
agent: window brain (§A2 background-asset stage in pipeline/window_loop.py)
description: Write the text-to-image prompt for each scene's background plate — an EMPTY location reference shared by every shot of that scene. Strict JSON output.
---

# Scene Image — the empty background plate

## Role
Each `bg_id` in the storyboard is ONE physical space. You write ONE
text-to-image prompt per bg_id. The image it produces is injected as a
reference into every shot of that space — it anchors WHERE the film
happens. It is a location plate, not a poster.

## The five laws

1. EMPTY-PLATE LAW (absolute): the image contains NO people. No
   character names, no crowd words (guests, officers, nobles, crowd,
   dancers, couple, servants, attendants), no action sentences. A
   deterministic gate strips violations — write clean instead of
   relying on it. Translate human activity into spatial evidence:
   "a ballroom full of dancing guests" → the ballroom itself, with a
   wide open dance floor; "she walks toward the throne" → "a wide
   central aisle leading to a raised dais".
2. PERIOD CONTRACT: state the era/culture/architecture explicitly
   (e.g. "19th-century European royal palace interior") and name every
   furnishing in period-correct terms. End with "no modern objects".
   An undated room gets modern furniture — and modern extras — from
   the model's defaults.
3. SPACE GEOMETRY: describe the room AS A SPACE — scale, floor
   material, walls, ceiling/dome, light fixtures and their positions,
   entrances. Wide establishing view, eye level, deep focus, neutral
   framing: no shallow depth of field, no dramatic angles, no
   close-ups. This plate anchors spatial continuity for every shot.
4. LIGHT IS MOOD: take the time of day, light sources and color
   temperature from the scene text (e.g. "night, hundreds of warm
   candles, golden glow"). Lighting consistency is half of background
   consistency.
5. ONE bg_id, ONE space: shots sharing a bg_id share this exact
   prompt's image. Write the space so every planned camera direction
   of those shots has something to look at (all four walls exist).

## Output (STRICT JSON, nothing else)

{"backgrounds": {"<bg_id>": {"prompt": "<the t2i prompt, English>"}}}

### Example

{"backgrounds": {"bg_1": {"prompt": "Empty interior of a 19th-century European royal palace ballroom at night: a vast hall under a gilded dome, three crystal chandeliers ablaze with warm candlelight, polished cream marble floor with dark inlay borders, mirrored walls between fluted gilded columns, tall arched double doors at the far end, wall sconces with burning candles, deep focus, eye-level wide establishing view, completely empty, no people, no modern objects."}}}
