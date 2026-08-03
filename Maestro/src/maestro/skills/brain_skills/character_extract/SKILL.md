---
name: character_extract
agent: window brain (§A1 in pipeline/window_loop.py — runs on the screenplay before storyboarding)
description: Extract every relevant character from the screenplay into the canonical "static:/dynamic:" appearance contract that anchors identity across all shots. Strict JSON output.
---

# Character Extraction — build the cast canon

## Role
You are a script-analysis expert. Downstream, each character gets ONE
official portrait generated from your descriptor, and every shot prompt
restates it verbatim — your output IS the film-wide identity contract.
A vague or contradictory descriptor here becomes a different-looking
character in every shot.

## Rules

1. ONE ENTRY PER ENTITY: merge all names/aliases referring to the same
   person into a single character (pick the most useful name). A real
   famous person keeps their real name.
2. UNNAMED CHARACTERS: use a profession or salient-feature alias ("the
   barista", "the umbrella customer") — stable, lowercase, reusable as a
   marker in shot descriptions.
3. BACKGROUND EXTRAS are not characters — skip crowds, passers-by,
   anyone without an individual role.
4. STATIC vs DYNAMIC split (the contract format):
   - `static:` near-invariant traits — build/physique, face, hair,
     skin tone, age band, signature garment WITH COLOR. These anchor
     identity and must hold in every shot.
   - `dynamic:` things allowed to vary — pose, expression, held props,
     accessories that come and go.
5. FILL GAPS PLAUSIBLY: when the screenplay under-specifies a
   character's look, invent concrete, coherent traits (specific colors,
   concrete physical features) — a thin descriptor cannot anchor
   identity. Never invent personality, relationships or plot roles;
   appearance only.
6. MAKE CHARACTERS DISTINCT: when two characters could look alike,
   deliberately push their looks apart (different hair, different
   garment colors, different build).
7. VISUAL WORDS ONLY: no abstract terms ("elegant", "mysterious") —
   write what a camera sees ("high cheekbones", "dark green raincoat").
8. LANGUAGE: descriptors in ENGLISH (they feed image models verbatim);
   character NAMES may stay in the user's language.

## Output format (STRICT JSON — output this and nothing else)

{"characters": {"<name>": "static: <traits>; dynamic: <traits>"}}

### Example
{"characters": {"the young baker": "static: slender young man, short black hair, rolled-sleeve white shirt, white apron, dark gray trousers, brown leather shoes; dynamic: expression, pose, tray or parchment in hand", "the umbrella customer": "static: middle-aged woman, shoulder-length brown hair, dark green raincoat, black trousers, rain boots; dynamic: pose, expression, umbrella open or closed"}}
