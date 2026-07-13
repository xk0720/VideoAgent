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
3. Shot count: follow the config (plan.n_shots, default 3; capped by
   max_shots); when the asset library carries a music profile, follow its
   section count. Fewer, longer shots beat fragments — each shot runs 4-15
   seconds (the generation model's duration domain).
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

## Where the output goes
outline (one line per shot) → DirectorAgent expands each line into a ShotSpec
(duration, camera language, identity/style refs, physics annotation, event
graph) → StoryboardMemory.from_outline builds the ledger (all pending).

## Current implementation status (honest note)
ScreenwriterAgent is currently a deterministic splitter (semicolon clauses,
n_shots cycling); its LLM call is a placeholder. This file is the operating
manual for the real LLM playwriting upgrade — the rules above are the
contract.
