---
name: scene_write
agent: ScreenwriterAgent + DirectorAgent (playwriting, window loop stage A)
description: Turn the user's prompt into a time-ordered list of scene/shot descriptions → ShotSpecs; the seed of the StoryboardMemory ledger.
---

# Scene Write (playwriting) — shot breakdown

## Role
Split the user's prompt into a TIME-ORDERED list of shot descriptions. This
list is the skeleton of window-based generation: each line becomes one entry
of the storyboard ledger, and every later decision (image plan, condition
strategy, review, repair) hangs off these entries.

## Breakdown rules
1. One complete, filmable sentence per shot: subject + action + setting
   (+ optional camera language).
   Bad: "then it falls" (whose? where?);
   Good: "Shot 2: the glass tips over the table edge and falls (kitchen,
   close-up)".
2. Scene splits: a change of location/time = a new scene. Write "scene N"
   explicitly in the description — the ledger parses the scene number from it
   (no parse → everything is scene 1, which is correct for single-scene
   scripts).
3. Shot count: YOURS to decide from the story's beats (one beat = one shot;
   a beat that must flow continuously stays inside one shot). Past similar
   tasks' shapes (episode_guidance) are experience, max_shots is only a cost
   ceiling. Fewer, longer shots beat fragments — each shot runs 4-15 seconds
   (the generation model's duration domain).
4. Time order IS generation order: the window loop walks the list strictly in
   order, and cross-shot continuity (previous shot's last frame / tail as the
   anchor) only holds between ADJACENT shots — put actions that must flow
   continuously into adjacent shots.
5. Entity consistency: use the SAME name for the same character/object across
   all shot descriptions (detection, tracking and asset retrieval all align
   by name).
6. Asset awareness: when the user provided assets (a location image, character
   photos, source clips), write the shots AROUND what the assets afford — a
   living-room background image is wasted on a beach script.

## Output format (STRICT JSON — output this and nothing else)

{"shots": ["Shot 1: <detailed filmable description>",
           "Shot 2: <detailed filmable description>", ...]}

- Each description is 15-40 words: subject + action + setting + camera /
  lighting where useful. One sentence that a video model can shoot.
- YOU decide the shot count — it is never preset. Read it off the story's
  beats, informed by `episode_guidance.past_task_shapes` (how many shots
  similar past tasks used and whether they succeeded). `max_shots` is ONLY a
  hard COST ceiling, never a target. NEVER pad the list by repeating a shot
  or re-using a clause — if the story only has two beats, return two shots.
- Consecutive story beats that must flow continuously belong in ADJACENT
  shots (the window loop can only anchor a shot on the one right before it).
- Carry entities forward by NAME and keep the setting words consistent
  across shots ("the same wooden kitchen table"), so retrieval, tracking and
  continuity anchors line up.
- When asset_catalog is non-empty, set the story where the assets are usable
  (a living-room background asset is wasted on a beach script).

### Example
user_prompt: "a glass falls off a table; shards scatter on the floor. Then a
boy comes and collects all shards, leaves happily"
{"shots": [
  "Shot 1: scene 1 — a clear drinking glass teeters on the edge of a wooden kitchen table, warm daylight, eye-level close-up, then tips over the edge",
  "Shot 2: scene 1 — the glass shatters on the tile floor and shards scatter outward, low camera angle at floor level, shallow depth of field",
  "Shot 3: scene 1 — a young boy kneels down, carefully collects the shards into his hand, then stands up and walks away smiling, medium shot"
]}

## Where the output goes
outline (one line per shot) → DirectorAgent expands each line into a ShotSpec
(duration, camera language, identity/style refs, physics annotation, event
graph) → StoryboardMemory.from_outline builds the ledger (all pending).

## Current implementation status
The window loop's `_write_outline` drives REAL LLM playwriting with this
skill as the prompt (strict-JSON output above, validated: strings only,
exact duplicates dropped, capped at max_shots). The deterministic splitter
(ScreenwriterAgent: semicolon clauses cycled over n_shots) is the FALLBACK
only — it CANNOT be the primary because clause-cycling fabricates duplicate
shots whenever clauses < n_shots.
