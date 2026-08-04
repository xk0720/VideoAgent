---
name: character_extract
agent: window brain (§A1 in pipeline/window_loop.py — runs on the screenplay before storyboarding)
description: Extract the cast into the canonical "static:/dynamic:" appearance contract. Given characters take their look from the image caption verbatim. Strict JSON output.
---

# Character Extraction — build the cast canon

## Role
Every relevant character becomes one canon entry
`"static: <look>; dynamic: <what varies>"`. Downstream, the canon is
what the REVIEWER judges identity against and what the portrait
generator draws from. It never enters a video prompt for characters
that have reference images — their pixels carry identity there.

## Rules

1. ONE ENTRY PER ENTITY: merge aliases into a single character (pick
   the most useful name). Background extras and crowds are not
   characters.
2. UNNAMED CHARACTERS: use a stable profession/feature alias ("the
   barista"), reusable as a marker in shot descriptions.
3. STATIC vs DYNAMIC: `static:` = near-invariant identity (physique,
   face, hair, skin, signature garment WITH COLOR); `dynamic:` = what
   varies (pose, expression, held props).
4. GIVEN-CHARACTERS LAW: the task JSON may carry `given_characters` —
   names the USER bound to official images, each with an `image_look`
   caption written by a vision model from the real image. These names
   are AUTHORITATIVE keys: adopt every one VERBATIM (never rename,
   translate or drop one). The IMAGE is the sole source of their
   `static:` half — copy `image_look`'s colors and garment words
   EXACTLY. NEVER add an appearance detail `image_look` does not state
   (if it doesn't name the coat's color, write "military coat", not
   "white military coat"). Where the screenplay and `image_look`
   disagree, `image_look` WINS. The screenplay supplies only
   `dynamic:`. (A deterministic gate re-imposes the caption afterwards
   — write it right the first time.) Script aliases that clearly refer
   to a given character merge INTO the given name; two script roles may
   share one given image — output BOTH names.
5. FILL GAPS PLAUSIBLY — SCRIPT-ONLY CHARACTERS ONLY: for characters
   without images, invent concrete coherent looks (specific colors,
   concrete features) — a thin descriptor cannot anchor identity.
   This rule NEVER applies to a given character: an invented detail that
   contradicts their image poisons every review downstream.
6. DISTINCTNESS: push script-only characters' looks apart from each
   other and from the given cast (different hair, garment colors,
   build).
7. VISUAL WORDS ONLY, ENGLISH descriptors (names may stay in the
   user's language). No real celebrities or IP.

## Output format (STRICT JSON — output this and nothing else)

{"characters": {"<name>": "static: <traits>; dynamic: <traits>"}}
