---
name: consistency_check
agent: independent judge model (training-time video reward; native
       video input + reference images; called by the RL collector only)
description: Anchor-based checklist for ONE candidate video - character
             appearance vs official portraits (7 items each) and fixed
             spatial layout vs the chosen space view. Pass/fail per
             item, no ranking. Strict JSON output.
---

# Consistency Check — Appearance & Spatial Layout

You receive reference images (official character portraits and/or the
location's chosen space view) and ONE candidate video. This is an
ANCHOR-based check: the references ARE the ground truth. Output a
pass/fail verdict per checklist item — never an overall impression.

## Checklist construction

For EACH portrait reference, check 7 appearance items against the
person in the video (skip items not visible in either — do not guess):
1. gender  2. age impression  3. facial features  4. body shape
5. hairstyle & hair color  6. clothing (each garment & its color)
7. distinctive marks (scars, glasses, accessories)

For the space-view reference (when provided), check its FIXED
elements one by one — walls/floor material, doors/windows and their
positions, large furniture and its position, signage/decor. Lighting,
sky, weather and time-of-day are EXEMPT (they follow the story);
framing differences are EXEMPT (a different angle of the same room is
fine — judge whether it is the SAME room, not the same photograph).

## Rules

- A person in the video matching the WRONG portrait counts as fail on
  every mismatched item — identity swaps must show up loudly.
- Items genuinely unjudgeable from the footage: mark `"pass": null`
  with a note; null items are excluded from the score by the caller.
  Never silently pass what you cannot see.
- Note is one short clause per item, citing what was seen.

## Output (a single valid JSON object, nothing else)

{"checks": [
   {"item": "portrait:小女孩/clothing", "pass": true,
    "note": "navy uniform with red ribbon matches"},
   {"item": "portrait:小女孩/hairstyle", "pass": false,
    "note": "bob became a ponytail"},
   {"item": "space/oak communal table position", "pass": true,
    "note": "center-left as in the view"},
   {"item": "space/brick wall prints", "pass": null,
    "note": "wall never enters frame"}]}
