---
name: scene_write
agent: window-loop brain LLM via _write_outline (ScreenwriterAgent = deterministic fallback; DirectorAgent expands specs), window loop stage A
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
   COMPLETE-ACTION LAW: every shot contains COMPLETE action
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
   DURATION-DENSITY LAW: the described action
   must FILL the declared seconds. A shot whose whole content is a thin
   transitional beat ("continues trotting one more body length toward
   the bowl") cannot fill even 3 seconds — the model has nothing left to
   do and LOOPS. Never give a "continue X" transition its own shot:
   merge it into the shot it leads into ("crosses the floor, reaches the
   bowl, stops and starts eating" is ONE shot). If a beat deserves its
   own shot, it contains a complete action arc; if it only bridges two
   arcs, it belongs inside one of them.
   Event budget per duration: 4-5s fits ONE complete action; 6-8s fits
   one core action plus at most one secondary beat; 9-10s fits two to
   three CHAINED actions. Match the budget honestly in both directions —
   an overstuffed short shot drops actions, an underfilled long shot
   loops.
4. Time order IS generation order: the window loop walks the list strictly in
   order, and cross-shot continuity (previous shot's last frame / tail as the
   anchor) only holds between ADJACENT shots — put actions that must flow
   continuously into adjacent shots.
5. Entity consistency: use the SAME name for the same character/object across
   all shot descriptions (detection, tracking and asset retrieval all align
   by name).
   CAST MARKER LAW: every time a CAST character
   appears in a shot description, wrap its name in angle brackets — `<the
   boy>`, `<the orange-and-white cat>` — with the name COPIED EXACTLY from
   the cast keys. The markers are MACHINE-PARSED: the executor derives
   "which cast members are on screen in this shot" from them, injects only
   THOSE appearance descriptors into the prompt, and tells the reviewer to
   check only those. Unmarked cast mention = that shot gets the FULL cast
   injected (safe but imprecise); a marker that matches no cast key is
   wasted. Markers never reach the generation model — the executor strips
   the brackets before any prompt. Mark characters only, not props or the
   setting.
6. Asset awareness: when the user provided assets (a location image, character
   photos, source clips), write the shots AROUND what the assets afford — a
   living-room background image is wasted on a beach script. VIDEO catalog
   entries describe the WHOLE clip (subject identity + setting + its main
   motion/camera, from native video understanding) — you can therefore
   write shots that directly CONTINUE or incorporate the user's footage
   ("the character keeps walking forward along the same path as in the
   clip"), not just shots that borrow a still.
7. ASSET MENTION LAW: when the user's task names a provided asset ("the cat
   from the photo appears", "use my living-room image"), the shot
   description(s) where it appears MUST state that requirement explicitly —
   using the asset's IDENTITY WORDS from the catalog (species, coat,
   markings, clothing: "the orange-and-white cat (from the user's photo)").
   NEVER paste the whole catalog caption: its pose/scene words describe
   the PHOTO, not this shot ("sleeping on a windowsill" inside a shot
   where the cat trots on the floor is a contradiction). One natural
   mention per shot is enough. Do NOT write
   @Image/@Video references here — numbering happens downstream (the slot
   manifest). A user-named asset that no shot description mentions is a
   script BUG. (The executor's deterministic warning only fires when the
   WHOLE catalog goes unmentioned — the law applies PER asset; do not rely
   on the warning.)
8. FRAME GEOGRAPHY: state WHERE in the frame subjects are
   and WHICH WAY they face whenever it matters for the cut or the action —
   "on the left side of the frame, facing right", "back to camera, walking
   away". When a shot is a close-up, name the exact body part / region in
   frame ("only her hands and the cup are visible"). NO-INVISIBLE-ELEMENTS:
   never mention an entity in a shot description that is NOT visible in
   that shot — an off-screen voice or an implied presence confuses both
   the generator and the reviewer; if it isn't in frame, it isn't in the
   description.
9. CAMERA-POSITION REUSE: within one scene, prefer
   RETURNING to an already-established camera setup over inventing a new
   angle for every shot — reuse stabilizes background, lighting and
   spatial layout across cuts for free. Say it explicitly: "same
   framing/angle as shot 1". Change the angle only when the story needs
   it (a reveal, a new subject, an axis change).
10. LANGUAGE LAW: cast descriptors, setting, shot
   descriptions, end_state, variation and opening_frame MUST be written in
   ENGLISH no matter what language the user's task is in — these fields
   feed image/video models directly, and a non-English descriptor reaching
   an image model produces a wrong character in a wrong scene. Entity
   NAMES (cast keys, used inside <angle markers>) and `dialogue` lines
   (spoken on screen) may stay in the user's language.
11. GIVEN-CHARACTERS MARKER DISCIPLINE: when the cast canon contains
   characters bound to USER-PROVIDED official images (they arrive in
   `cast_canon`), their names must appear in shot descriptions as
   `<markers>` COPIED CHARACTER-FOR-CHARACTER from the canon keys — the
   marker is the key that auto-attaches that character's reference image
   to the shot's generation call; one wrong character in the name breaks
   the whole identity chain for that shot. Never invent an alias for a
   bound character inside a description.
12. OBSERVABLE ACTION, NOT ABSTRACT EMOTION: the model renders pixels,
   not feelings. Convert every mood or intention into things a camera
   can record — facial expression, gaze direction, posture and gesture,
   interaction with objects, environmental feedback ("wind lifts her
   hair"). Bad: "she feels free and hopeful"; good: "she lifts her face
   to the sky and slowly opens her arms, a faint smile forming". Ban
   vague evaluative words as action carriers ("showcases elegance",
   "full of power").

## Output format (STRICT JSON — output this and nothing else)

{"cast": {"<entity name>": "<CANONICAL appearance descriptor in the form
           'static: <10-20 words of UNCHANGING visual identity —
           species/build, coat/wardrobe with colors, distinctive marks>;
           dynamic: <what varies across shots — expression, pose,
           held items — or "none">'>"},
 "setting": "<one canonical set-dressing + lighting sentence>",
 "shots": [{"description": "Shot 1: <detailed filmable description; every
             cast appearance marked as <name>>",
            "duration_s": <int 4-10>,
            "end_state": "<one sentence: at the cut, who/what is where,
                          moving or still, in which direction>",
            "variation": "large|medium|small",
            "opening_frame": "<first shot & scene cuts ONLY: one purely
             STATIC opening snapshot — layout, subjects' positions, NO
             ongoing action; omit for continuing shots>",
            "dialogue": {"speaker": "<cast key of WHO SPEAKS,
             verbatim>", "line": "<ONE spoken line, at most 6 words>"}
             — ONLY when a cast character visibly speaks at medium
             close-up or closer; omit otherwise,
            "bg": "<background id, e.g. bg_1>"},
           {"description": "Shot 2: <detailed filmable description>",
            "duration_s": <int 4-10>,
            "end_state": "...",
            "variation": "..."}, ...],
 "music_plan": {"scene 1": "<one music description per scene: mood +
                genre + tempo/BPM; omit a scene (or the field) for
                deliberate silence>"}}

- `cast` + `setting` are the CROSS-SHOT CONSISTENCY CONTRACT (video models
  have NO memory across calls): one canonical appearance descriptor per
  recurring character/object and one canonical set-dressing+lighting line.
  Every downstream prompt (every shot, every repair) restates the relevant
  descriptors VERBATIM, and the reviewer judges each shot against them.
  Write them concrete and visual — colors, marks, materials — not moods.
  For user-provided assets, derive the descriptor from the asset's catalog
  identity words.
  CAST CANON LAW: when the task JSON carries a non-empty `cast_canon`,
  those names and descriptors are the film-wide identity canon, already
  settled by the character-extraction stage (portraits are generated from
  them). Adopt every canon entry VERBATIM in your `cast` output — same
  key, same descriptor, word for word. You may only ADD characters the
  canon missed; never rename, rewrite, restyle or "improve" a canon
  entry (the executor overrides same-name entries with the canon anyway
  — a divergent rewrite is wasted and misleading).
  STATIC/DYNAMIC SPLIT: the `static:` half is the identity
  contract — it must stay word-for-word true in EVERY shot; the `dynamic:`
  half names what is ALLOWED to change (expression, pose, held items), so
  downstream prompts vary only those. DISTINCTNESS: when two cast members
  could be confused (two boys, two cats), give each at least one loud
  distinguishing static feature (hair color, clothing color, a marking).
  COMPLETION-DESIGN: describe every character as if designing it for a
  character sheet — complete enough that an artist could draw it from the
  static half alone (species/age/build, hair, full outfit with colors,
  footwear); "a boy" is not a descriptor.

- Each description is 15-40 words: subject + action + setting + camera /
  lighting where useful. One sentence that a video model can shoot.
- `duration_s` is YOUR call per shot (integer seconds, 4-10 — the fixed
  planning range): a quick impact beat wants 4-5s, a slow establishing pan
  or a multi-step action wants 8-10s. Judge from how long the described
  action NEEDS. It is never preset by config; the executor maps your
  seconds onto each generation model's own duration domain. If you omit
  duration_s the executor sends NO duration and the model's own default
  applies — so always state it.
- `end_state` is MACHINE-USED twice beyond the baton: the reviewer treats
  it as the shot-end acceptance criterion, and at generation time a VLM
  reads the previous shot's ACTUAL last frame next to your scripted baton
  — write end_states as concrete, visually checkable freeze-frames, never
  literary moods. It is the HANDOFF BATON (the window loop shoots shots in order
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
  5. CAMERA HANDOFF LAW: the camera is a moving subject
     too. Every end_state ALSO states the camera's own state at the cut
     ("camera: static" / "camera: slowly pushing in" / "camera: tracking
     right at walking pace"). The next shot's opening must CONTINUE that
     camera state or open from a settled camera; NEVER reverse the
     camera's direction across a cut — ending on a push-in and opening
     on a pull-back is the #1 source of jump-cut feel. A junction reader
     watches the previous shot's final seconds and reports the actual
     camera motion; the reviewer checks the handoff.
- `variation` = how much the LAST frame differs from the
  FIRST frame of this shot: `small` (same composition, minor motion — a
  glass rocking), `medium` (subject moved / pose changed inside the same
  framing), `large` (composition or location visibly different — subject
  crossed the frame, camera traveled). It is a STRATEGY HINT downstream
  (small favors continuation-style generation; large favors two-anchor /
  free routes). State it honestly per shot; omit only if truly unsure.
- `opening_frame` — ONLY for the FIRST shot and shots right after a SCENE
  CUT: one purely STATIC snapshot of the opening composition (who is
  where, facing which way, the set and light) with NO ongoing actions —
  it becomes the base for a generated opening still. CONTINUING shots must
  OMIT it (their opening IS the previous end_state; restating it invites
  contradictions).
- `dialogue` (audio line) — {"speaker": "<cast key>", "line": "<ONE
  spoken line of AT MOST 6 words>"}, ONLY when a cast character visibly
  speaks on screen at medium close-up or closer (lip-sync needs face
  resolution; a wide shot gets imprecise lips). The `speaker` MUST be a
  cast key copied verbatim — the executor points the lip-sync clause at
  exactly that character; a wrong or missing speaker puts the line on
  the wrong mouth. Omit the field for shots with no on-screen speech —
  never narrate through it. Do NOT write the line into the description
  itself.
- `bg` (BACKGROUND PREDICTION — drives which background reference image
  the generator receives): give every shot a background id. Keep the
  SAME id (e.g. "bg_1") for every shot that happens in the same
  physical space, even across different camera angles — the executor
  then feeds all those shots ONE shared background reference so the
  master background cannot drift. Switch to a NEW id ("bg_2") ONLY when
  the action moves to a genuinely different space (another room,
  outdoors, a new location). A camera angle change, a close-up, or a
  different corner of the same hall is NOT a background change.
- `music_plan` (film-level) — one music description PER
  SCENE ("scene 1": "warm playful orchestral, light pizzicato, 95bpm").
  All shots in a scene share ONE generated track (one track cannot
  disagree with itself — that is how cross-segment music consistency is
  achieved). State mood + genre + tempo; mark deliberate silence by
  omitting the scene. Music is mixed in AFTER assembly and ducks under
  dialogue automatically — do not mention music inside shot
  descriptions or prompts.
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
{"cast": {"the boy": "static: a young boy of about eight, short black hair, red striped t-shirt, blue shorts, white sneakers; dynamic: expression shifts from focused to smiling, hands empty then holding shards"},
 "setting": "a warm daylit kitchen with a wooden table, beige tile floor and a window over the sink",
 "shots": [
  {"description": "Shot 1: scene 1 — a clear drinking glass teeters at the very edge of the wooden kitchen table, right of frame, rocking further with each wobble, warm daylight, eye-level close-up slowly pushing in", "duration_s": 5,
   "end_state": "the glass rocks at maximum lean on the table edge, still in motion; camera: still pushing in slowly",
   "variation": "small",
   "opening_frame": "a clear drinking glass stands at the very edge of a wooden kitchen table, right of frame, warm window daylight, eye-level close-up"},
  {"description": "Shot 2: scene 1 — taking over from the rocking glass, it tips past the edge, FALLS and SHATTERS on the beige tile floor, shards scattering outward, low floor-level angle, shallow depth of field", "duration_s": 4,
   "end_state": "shards lie scattered and at rest on the tile floor around the impact point; camera: static at floor level",
   "variation": "large"},
  {"description": "Shot 3: scene 1 — <the boy> kneels down beside the scattered shards, facing camera, carefully collects them into his hand, then stands up and walks away smiling, medium shot, same floor-level area as shot 2", "duration_s": 8,
   "end_state": "the boy is walking away from camera, shards in hand, floor clear; camera: static",
   "variation": "large"}
]}

Note how the batons connect and obey the COMPLETE-ACTION LAW: shot 1 hands
off SUSTAINABLE motion (rocking, camera push) — never a suspended fall;
shot 2 completes tip + fall + impact INSIDE one shot; shot 2 may end at
rest because shot 3 introduces a NEW agent (the boy) acting on the shards.
The cast entry splits static identity from what may vary; `<the boy>` is
marked exactly where he appears (shots 1-2 have no cast on screen — no
markers, and the boy is rightly NOT mentioned there: no-invisible-elements);
only shot 1 opens the scene, so only it carries an opening_frame.

## Where the output goes
outline (one line per shot) → DirectorAgent expands each line into a ShotSpec
(duration, camera language, identity/style refs, physics annotation, event
graph) → StoryboardMemory.from_outline builds the ledger (all pending).

## Current implementation status
The window loop's `_write_outline` drives REAL LLM playwriting with this
skill as the prompt (strict-JSON output above; validated per shot object:
description ≥12 chars, case-insensitive exact duplicates dropped, list
capped at max_shots; duration_s coerced to int and clamped into 4-10
(invalid/missing → no duration sent to the API); end_state and
cast/setting passed through as-is — missing → empty, never fabricated;
variation validated against {large, medium, small} else empty;
opening_frame passed through; `<name>` markers parsed for per-shot cast
then STRIPPED from every prompt-bound string — they never reach a
generation model). The deterministic splitter
(ScreenwriterAgent: semicolon clauses cycled over n_shots) is the FALLBACK
only — it CANNOT be the primary because clause-cycling fabricates duplicate
shots whenever clauses < n_shots.
