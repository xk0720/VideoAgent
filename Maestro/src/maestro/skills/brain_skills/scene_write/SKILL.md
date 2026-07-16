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
   COMPLETE-ACTION LAW (2026-07-16): every shot contains COMPLETE action
   units — never cut a one-off action halfway. A jump LANDS inside the
   shot that started it; a fall IMPACTS inside its shot. Handing motion
   across a cut is only allowed with SUSTAINABLE motion (walking,
   trotting, rolling, a camera move) — "frozen mid-air at the cut" is
   both unfilmable (models freeze the subject) and uncontinuable.
   Bad:  shot 1 "...and leaps down" / end_state "mid-air below the sill"
   Good: shot 1 "...leaps down and LANDS, already trotting" / end_state
         "trotting across the floor toward the bowl, mid-stride".
   LINKING NARRATION: each description (except the first) opens with how
   it takes over from the previous end_state and closes with how it hands
   off — a reader should see the seam ("...lands and, without pausing,
   trots toward..."). Thin descriptions produce thin videos.
2. Scene splits: a change of location/time = a new scene. Write "scene N"
   explicitly in the description — the ledger parses the scene number from it
   (no parse → everything is scene 1, which is correct for single-scene
   scripts).
3. Shot count: YOURS to decide from the story's beats (one beat = one shot;
   a beat that must flow continuously stays inside one shot). Past similar
   tasks' shapes (episode_guidance) are experience, max_shots is only a cost
   ceiling. Fewer, longer shots beat fragments — each shot runs 4-10 seconds
   (the fixed planning range; the executor maps it onto each generation
   model's own duration domain).
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
7. ASSET MENTION LAW: when the user's task names a provided asset ("the cat
   from the photo appears", "use my living-room image"), the shot
   description(s) where it appears MUST state that requirement explicitly —
   using the asset's IDENTITY WORDS from the catalog (species, coat,
   markings, clothing: "the orange-and-white cat (from the user's photo)").
   NEVER paste the whole catalog caption: its pose/scene words describe
   the PHOTO, not this shot ("sleeping on a windowsill" inside a shot
   where the cat trots on the floor is a contradiction — field bug
   2026-07-16). One natural mention per shot is enough. Do NOT write
   @Image/@Video references here — numbering happens downstream (the slot
   manifest). A user-named asset that no shot description mentions is a
   script BUG (the executor warns loudly).

## Output format (STRICT JSON — output this and nothing else)

{"shots": [{"description": "Shot 1: <detailed filmable description>",
            "duration_s": <int 4-10>,
            "end_state": "<one sentence: at the cut, who/what is where,
                          moving or still, in which direction>"},
           {"description": "Shot 2: <detailed filmable description>",
            "duration_s": <int 4-10>,
            "end_state": "..."}, ...]}

- Each description is 15-40 words: subject + action + setting + camera /
  lighting where useful. One sentence that a video model can shoot.
- `duration_s` is YOUR call per shot (integer seconds, 4-10 — the fixed
  planning range): a quick impact beat wants 4-5s, a slow establishing pan
  or a multi-step action wants 8-10s. Judge from how long the described
  action NEEDS. It is never preset by config; the executor maps your
  seconds onto each generation model's own duration domain. If you omit
  duration_s the executor sends NO duration and the model's own default
  applies — so always state it.
- `end_state` is the HANDOFF BATON (the window loop shoots shots in order
  and shot N+1 is generated FROM shot N's final frame — your baton is what
  makes that cut physically possible):
  1. State the exact freeze-frame at the cut: every key subject's position,
     whether it is MOVING or AT REST, and its direction.
  2. The next shot's description must OPEN from that exact state — same
     place, same motion. Never write an opening that contradicts the
     previous end_state.
  3. MOTION HANDOFF LAW: if the next shot needs the subject in motion, do
     NOT let it come to rest at this shot's cut — keep it moving through
     the cut (or let it exit frame moving). An object AT REST cannot start
     moving again by itself; if the story needs that, a NEW force/event
     (a push, a gust, a character's touch) must be written INTO the next
     shot's description, or the story is physically wrong.
  4. SUSTAINABLE-MOTION-ONLY at the cut: the moving state in an end_state
     must be sustainable (walking/trotting/rolling/camera move) — NEVER a
     suspended one-off action ("mid-air", "mid-fall", "mid-impact").
     Complete the jump/fall inside its own shot first (COMPLETE-ACTION
     LAW above).
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
  {"description": "Shot 1: scene 1 — a clear drinking glass teeters on the edge of a wooden kitchen table, warm daylight, eye-level close-up, then tips over the edge", "duration_s": 5,
   "end_state": "the glass is in mid-air just below the table edge, falling fast toward the tile floor"},
  {"description": "Shot 2: scene 1 — the falling glass shatters on the tile floor and shards scatter outward, low camera angle at floor level, shallow depth of field", "duration_s": 4,
   "end_state": "shards lie scattered and at rest on the tile floor around the impact point"},
  {"description": "Shot 3: scene 1 — a young boy kneels down beside the scattered shards, carefully collects them into his hand, then stands up and walks away smiling, medium shot", "duration_s": 8,
   "end_state": "the boy is walking away from camera, shards in hand, floor clear"}
]}

Note how the batons connect: shot 1 ends with the glass FALLING (not resting)
because shot 2 needs the impact; shot 2 may end at rest because shot 3
introduces a NEW agent (the boy) acting on the shards.

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
