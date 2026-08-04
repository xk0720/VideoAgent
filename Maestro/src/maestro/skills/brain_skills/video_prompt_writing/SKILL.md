---
name: video_prompt_writing
agent: every prompt-writing role (scene_write, window brain, prompt enhancer, repair hint writer)
description: Central craft rules for writing video-generation prompts — subject/scene/action-with-timing structure, the three motion channels, camera grammar, identity anchors, mode-specific rules for t2v/i2v/flf2v, reference-slot binding, and a pre-flight quality checklist.
---

# Video Prompt Writing — the central craft skill

A video prompt is not an image prompt with adjectives. An image prompt
describes a state; a video prompt describes a **state change over time**.
Every prompt you write must answer, explicitly and without conflict:

1. What is in the frame (subject, scene).
2. What moves, and how the motion unfolds second by second.
3. How the camera observes and travels.
4. What must stay exactly the same from first frame to last.

If a prompt only answers (1), the model will produce a near-still frame,
random drift, or invented motion. The three dimensions that matter most,
in order: **what the subject does → how the action progresses through
time → how the camera moves**. Never ship a prompt missing any of the
three.

## 1. The core formula

Every video prompt is assembled from these blocks:

```text
Subject + Scene + Action-with-timing + Environmental motion
        + Camera + Style + Consistency constraints
```

Not every block is verbose in every mode (i2v collapses most of it — see
Section 12), but every block is a *decision you consciously made*, never
an accident of omission.

Recommended paragraph order for the final prompt:

```text
[subject & scene] [action & timing] [environmental motion]
[camera] [style] [constraints]
```

Length discipline: clarity and absence of conflict beat length. A short
prompt with one clean action and one clean camera move outperforms a long
prompt with three competing ideas. Delete repeated adjectives, actions
that exceed the clip duration, contradictory camera instructions, and
background detail that serves no purpose. As rough envelopes: a
motion-only i2v prompt is a short paragraph; a single t2v shot is one to
three paragraphs; anything needing more should be split into multiple
shots upstream, not crammed into one prompt.

## 2. Name no names — visual anchors only

Video generation models do not know who "Alice" is. A character name is
an empty token that binds to nothing on screen. **Never use character
names in motion or appearance description.** Refer to every subject by
its distinguishing visual features:

```text
Bad : Alice is walking across the plaza.
Good: The short-haired woman in a green dress is walking across the plaza.
```

Rules:

- Pick the same descriptor phrase for a character and repeat it verbatim
  every time that character acts ("the young baker", "the woman in a
  dark green raincoat"). Do not vary the phrasing mid-prompt — a new
  phrase risks spawning a new person.
- With two or more subjects, descriptors must be mutually exclusive
  ("the taller man in the gray coat" vs "the boy in the yellow cap"),
  so every action verb has exactly one possible owner.
- Never reference real celebrities, real brands, or recognizable IP in
  any prompt or example you produce; keep all characters generic.

## 3. Observable action, not abstract emotion

The model renders pixels, not feelings. Convert every abstract state
into things a camera could record:

- facial expression (brows tighten, jaw sets, a slight smile forms)
- gaze (looks down, lifts eyes toward the doorway)
- posture and gesture (shoulders drop, arms open slowly)
- interaction with objects (grips the railing, sets the cup down)
- environmental feedback (wind lifts her hair, steam curls past his face)
- camera treatment (slow push-in on the eyes)

```text
Bad : She feels free and hopeful.
Good: She lifts her face to the sky and slowly opens her arms; wind
      streams through her hair and coat, and a faint smile forms.
```

Ban vague evaluative words as action carriers: "showcases elegance",
"full of power", "feels premium", "cinematic vibes". If a mood word must
survive, it belongs in the style block, backed by concrete light, color
and motion choices that actually produce that mood.

## 4. Technical precision beats intensity adverbs

"Very fast" is not a speed; it is a shrug. Describe motion with concrete
nouns, trajectories, a sense of speed, and the physical consequences of
force:

```text
Bad : The car drives away very fast.
Good: The car accelerates from a standstill, rear tires briefly slipping,
      motion blur streaking the background as it passes the camera.
```

Concretely:

- **Trajectory**: direction, path, start point, end point ("walks from
  the doorway to the window, stopping at arm's length from the glass").
- **Speed sense**: tie speed to visible evidence (motion blur, stride
  frequency, water spray, cloth lag) rather than bare adverbs.
- **Force results**: impacts displace things — dust kicks up, the table
  jolts, liquid sloshes. Stating the consequence makes the model commit
  to the cause.
- **Amplitude**: "turns her head slightly to the right" is executable;
  "turns" alone invites a full-body spin.

## 5. The three motion channels — always written apart

Every prompt separates motion into three channels, each in its own
sentence(s), never fused:

```text
Subject motion    : The young baker kneads the dough, leaning into each push.
Environment motion: Flour dust drifts in the warm light; a curtain sways.
Camera motion     : The camera holds a static medium shot.
```

Rules:

- **Environmental motion must be weaker than subject motion.** It exists
  to make the world alive, never to compete. Background figures stay
  soft, slow and few; never schedule multiple fast background events.
- If a channel is intentionally still, say so ("the camera is static",
  "the background remains motionless") — silence is ambiguity.
- Never phrase camera motion in a way that could be read as subject or
  background motion, or vice versa. "The scene moves closer" is poison;
  write "the camera slowly dollies in".

## 6. Temporal progression — the three-beat spine

Default structure for any clip:

```text
At first  — establish the initial state.
Then      — execute the core action.
Finally   — settle into a stable end state.
```

```text
At first, the woman in a dark green raincoat walks slowly with her head
down. Then she stops and lifts her gaze toward the end of the street.
Finally, she stands still, and the shot rests on her calm profile.
```

Time connectives to use: *At first / Then / As the shot continues /
Gradually / Finally*. Connectives sequence actions; they never replace
them — each beat still needs a concrete, observable verb.

**Event density must fit the duration:**

| Clip length | Event budget                              | Structure                    |
|-------------|-------------------------------------------|------------------------------|
| 2–4 s       | one simple action                         | initial state → action       |
| 5–8 s       | one core action + at most one secondary   | initial → action → settle    |
| 8–12 s      | two to three chained actions              | three or four beats          |
| > 12 s      | do not write one prompt — split into shots| multi-shot planning upstream |

Every clip needs an explicit end state ("she holds the smile", "the cup
rests on the table"). Without one, the model improvises an ending —
usually badly.

## 7. Identity anchors

For any human or creature that must stay recognizable, fix **3–5 stable
anchors** and repeat them wherever the character appears (and across
shots of the same character):

1. face / species,
2. hairstyle,
3. dominant clothing color or garment,
4. age range and build,
5. one signature item (earrings, apron, scarf, satchel).

```text
The same short-haired woman in a dark green raincoat, silver earrings,
slight build, remains identical throughout the video.
```

Too few anchors → drift; a wall of micro-details → noise the model
ignores. Identity anchors outrank decorative detail: when trimming for
length, decoration goes first, anchors never.

## 8. Scene and space

A scene is a set of spatial relationships, not a place name. When the
mode requires scene description (t2v; never re-described in i2v), state:

- place, time of day, weather;
- foreground / middle ground / background, and which layer the subject
  occupies;
- the subject's position within the frame;
- which environmental elements are allowed to move — and, implicitly,
  that everything else holds still.

Recommended shape:

```text
place + time/weather + subject position + foreground elements
      + background elements + spatial mood
```

```text
A city street at night just after rain; the subject stands in the
middle ground; the foreground shows wet asphalt and shallow puddles;
softly blurred cars, pedestrians and lit signs fill the background.
```

Rules:

- Pin the relative positions of objects that matter to the action or
  must stay put.
- Keep background entities few; every extra entity is a flicker risk.
- If the background is complex, reduce the subject's action complexity
  to compensate — never max out both.

## 9. Camera grammar

A camera block states at minimum: **shot size, angle, movement, focus**.

Shot sizes: `extreme close-up`, `close-up`, `medium close-up`,
`medium shot`, `full shot`, `wide shot`, `establishing shot`.

Angles: `eye-level`, `low-angle`, `high-angle`, `top-down`,
`side profile`, `over-the-shoulder`, `POV`, `rear view`.

Movements: `static shot`, `slow dolly in`, `slow dolly out`,
`tracking shot`, `lateral tracking`, `pan`, `tilt`, `orbit shot`,
`crane shot`, `handheld`, `zoom`.

Selection rules:

- **One primary camera movement per shot.** Composite requests breed
  chaos. If a transition between moves is essential, sequence it
  explicitly ("starts as a static medium shot, then slowly orbits about
  30 degrees to the right") — never stack simultaneous moves.
- **Camera speed matches action speed.** Calm scene → static or slow
  move; action scene → tracking or restrained handheld; product shot →
  slow push-in, orbit, or macro glide.
- Give the movement a direction and an implied distance or angle, not
  just a verb ("slow dolly in toward her face" beats "dolly in").
- Focus changes at most once per shot ("focus shifts from the raindrops
  on the glass to her face"); no repeated rack-focus.
- Never write physically contradictory instructions ("static shot while
  rapidly orbiting the subject and pulling back").

Composition, when it matters, is one sentence that pins the subject's
frame position ("she stays on the right third of the frame; the camera
preserves this framing throughout") — this measurably reduces subject
drift.

## 10. Style, light, and constraints

**Style** is concrete: medium + photographic/animation idiom + lens
texture + material detail. "Cinematic live-action, 35mm photography,
realistic skin texture, subtle film grain" works; "premium epic vibes"
does not. Never stack incompatible styles (documentary realism + heavy
cartoon shading). Style must never outweigh the action description.

**Light and color** follow one compact recipe: key light direction +
hard/soft + color temperature/palette + mood, expressed visually
("soft backlight rims her silhouette; blue-violet neon reflects off the
wet asphalt; desaturated cool palette; the mood is quiet and wistful").
Light direction and time of day must agree across a multi-shot sequence.

**Constraints — positive first, negatives targeted:**

1. State the desired stable behavior first:
   `The camera stays steady; the subject only blinks, breathes softly,
   and turns her head slowly.`
2. Then add negatives **only for risks actually present in this shot**:
   `No facial distortion, no identity change, no background
   reconstruction.`
3. Never dump a generic negative wall. Complex hand interaction → add
   hand/finger stability. Product shot → shape, logo, text, material
   stability. Portrait → face, hair, clothing, body-proportion
   stability. Nothing else.

## 11. Defaults when input is underspecified

Fill in the minimum needed to make the shot executable; never invent
plot, characters, or scene changes the request did not ask for.

- No duration given → design for a single shot of about 5 seconds.
- No camera given → one steady, slow, single movement — or static.
- No action speed given → natural, smooth, unhurried.
- No style given → high-quality realism; for i2v, inherit the input
  image's style.
- No end state given → design a natural settle or hold for the action.
- No shot count given → single shot.
- No aspect ratio given → leave it out of the prompt; it is a parameter,
  not prose.
- i2v with no appearance notes → keep the input frame's appearance,
  composition and background structure unchanged.

If a missing fact would genuinely change the output — subject count, the
core action, whether the camera moves — infer it from context first;
only when it is truly undecidable, surface one high-value question
upstream instead of guessing.

## 12. Mode-specific rules

### 12.1 Text-to-video (t2v)

The prompt is the only source of truth, so every block of the core
formula is written out. Workflow: one-sentence shot goal (who, where,
what core action) → identity anchors → space (place, time, weather,
foreground/background, subject position) → three-beat timeline → three
motion channels → one camera move → style and light → targeted
constraints → compress and de-conflict.

### 12.2 Image-to-video (i2v) — first frame already fixed

The image already defines appearance, composition, initial pose, style,
and lighting. **Do not re-describe them.** Re-narrating static
appearance dilutes the motion instruction — the one thing the model
needs from you. The whole prompt is:

```text
what moves + how the camera moves + what must not change
```

Template:

```text
Keep the subject's appearance, outfit, composition, background layout
and visual style from the input frame unchanged.
<subject motion, with speed and amplitude>.
<one gentle environmental motion>.
<one camera behavior, preserving the original framing>.
No identity change, no facial distortion, no background reconstruction.
```

Extra i2v rules:

- Requested motion must be reachable from the visible pose — no
  large turns or relocations the first frame cannot plausibly begin.
- Change one thing at a time; never simultaneously alter subject,
  background, lighting and camera.
- Keep camera amplitude small; a big move forces the model to invent
  off-frame geometry and reconstructs the background.
- For a continuation shot (extending an existing clip), the same law
  applies with zero appearance description: motion, camera, and
  invariants only.

### 12.3 First-and-last-frame (flf2v)

Both endpoints are fixed; you are writing the road between them.

- Compare the two frames: subject position, pose, gaze, object states,
  lighting.
- Describe the **shortest natural motion path** that connects them, at
  a steady rhythm, with physically plausible intermediate poses.
- **Add no event that is absent from both frames.** No new props,
  characters, weather, or detours — the model must travel, not sightsee.
- State whether the change between frames is subject motion, camera
  motion, or both, and keep identity, wardrobe, scene structure and
  lighting continuous throughout.

## 12b. The reference rule (absolute, overrides all other sections)

When reference slots exist, the slot manifest maps every token to its
character. In the prompt: a character is referred to by TOKEN ONLY at
the word position where they act ("<<<image_2>>> walks toward
<<<image_4>>>"); character NAMES are FORBIDDEN (a name means nothing to
the video model and reads as a phantom extra person); APPEARANCE
DESCRIPTION is FORBIDDEN (no face, hair, wardrobe or color words — the
reference images carry the look; text beside pixels is a second,
competing description and the #1 identity killer). Write only: who
(token) + action + camera. Dialogue uses the speaker's token:
'<<<image_4>>> says: "…"'. Full appearance prose belongs solely to
characters that have NO reference token.

## 13. Reference-image binding contract

When the executor attaches reference images, it hands you a **slot
manifest** listing each slot's ID and role. This manifest is law:

- Use **only** the reference tokens listed in the manifest, copied
  **verbatim** — the exact dialect the executor declared. Kling dialect
  looks like `<<<image_1>>>`; seedance dialect looks like `@Image1`. Do
  not translate between dialects, do not change case, brackets, or
  numbering, and do not invent slots that are not in the manifest.
- Bind each token to its subject descriptor in prose:
  `the young baker (<<<image_1>>>) lifts the tray` or
  `the young baker (@Image1) lifts the tray`.
- **A portrait reference binds identity only.** It tells the model whose
  face this is — never replicate the reference photo's pose, framing,
  or background in the shot. The shot's pose and composition come from
  your prompt, not from the portrait.
- Object/product references bind shape, logo, color, material — again,
  not the reference photo's background or angle.

## 14. Pre-flight quality checklist

Run this before emitting any prompt; fix, simplify, or rewrite instead
of shipping a failure:

- [ ] Subject and scene are unambiguous; every actor has a unique visual
      descriptor (no character names).
- [ ] The core action is observable, has direction/speed/amplitude, and
      an explicit end state.
- [ ] Event count fits the duration budget (Section 6 table).
- [ ] Actions are sequenced with time connectives; nothing important
      happens "simultaneously".
- [ ] Subject, environment, and camera motion are in separate sentences;
      environment is weaker than subject.
- [ ] Exactly one primary camera movement, with shot size, angle, and a
      speed that matches the action.
- [ ] Identity anchors present (3–5) and repeated; consistency
      constraints match this shot's actual risks.
- [ ] Style, light, and palette are mutually compatible and do not
      outweigh the action text.
- [ ] Mode rules honored: i2v/continuation is motion-only; flf2v adds no
      event absent from both frames.
- [ ] Reference tokens copied verbatim from the slot manifest; portrait
      references used for identity only.

## 15. Failure modes → fixes

| Failure | Root cause | Fix |
|---|---|---|
| Frame barely moves | prompt is subject+style only; action abstract, no direction/amplitude | write start state → direction → duration → end state for the core action |
| Chaotic or dropped actions | too many events for the duration; no time connectives; conflicting verbs | cut to one core action, sequence with At first/Then/Finally, split into shots if needed |
| Identity drift | too few (or too many) anchors; big turns/occlusion; heavy camera move; everything changing at once | fix 3–5 anchors, shrink action and camera amplitude, state "the same … remains identical" |
| Background flicker / reconstruction | crowded background; large camera travel; prompt demands background change | fewer background entities, slow small camera move, state "background layout stays unchanged" |
| Camera runaway | multiple simultaneous moves; no direction or speed given | one movement, explicit direction + steady speed, e.g. "slow steady dolly in toward her face, no shake, no rotation" |

## 16. Worked examples

### Example A — t2v, single shot (8 s)

```text
A woman in a dark green raincoat with short black hair and a slight
build walks alone down a city street at night, just after rain. She is
in the middle ground of the frame; the foreground shows wet asphalt and
shallow puddles; blurred storefront signs and a few slow pedestrians
fill the background.

At first, she walks slowly forward with her head down, her boots
stepping through a shallow puddle. Then she stops and lifts her gaze
toward the far end of the street. Finally, she stands still, and the
shot rests on her calm profile.

A light wind stirs her hair and the hem of the raincoat; water drips
slowly from a shop awning; the wet pavement reflects the signs above.

Medium shot, eye-level, tracking her from behind and to the side at a
steady walking pace, then easing to a stop as she stops. Shallow depth
of field: she stays sharp, the background softly blurred. She holds the
right third of the frame throughout.

Cinematic live-action, 35mm photography, cool desaturated blue-gray
palette, low-key lighting with soft neon reflections; the mood is quiet
and wistful.

The same woman — short black hair, dark green raincoat, slight build —
remains identical throughout. Motion is natural and continuous, the
camera steady. No identity change, no extra people appearing, no
background reconstruction, no flicker.
```

### Example B — i2v continuation shot, motion-only (5 s)

The first frame already shows the young baker standing at a wooden
counter, hands resting on a ball of dough, morning light from a side
window. The prompt adds motion only:

```text
Keep the baker's appearance, outfit, pose baseline, the counter, the
background layout and the lighting from the input frame unchanged.

He presses both palms into the ball of dough and kneads it with slow,
steady pushes, leaning slightly into each press; flour dust lifts
around his hands. Then he pauses, brushes his palms together once, and
looks down at the dough with a small satisfied nod.

Fine flour particles drift through the shaft of window light; nothing
else in the background moves.

The camera performs a very slow dolly in toward his hands, keeping the
original framing and his position in frame.

No identity change, no facial distortion, no new objects, no background
reconstruction.
```

### Example C — flf2v, first-to-last-frame transition (6 s)

First frame: the young baker stands at the open oven, holding a tray of
bread with both hands, body turned toward the oven. Last frame: the
tray rests on the wooden counter, and the baker leans on the counter
edge, looking down at the loaves. The prompt describes only the
shortest natural path between the two:

```text
Starting from the first frame, the young baker turns from the oven
toward the counter in one smooth motion, carrying the tray level with
both hands. He takes two unhurried steps, lowers the tray, and sets it
down gently on the wooden counter. Finally, he rests both hands on the
counter edge and leans forward slightly, looking down at the loaves,
matching the last frame.

Faint steam rises from the bread; nothing else in the room moves.

The camera holds a static medium shot at eye-level for the entire
transition; the framing of the last frame is reached by his movement,
not by camera travel.

His identity, apron, hairstyle, the kitchen layout and the warm side
lighting stay continuous throughout. The tray and loaves keep the same
shape and count. No new objects or characters appear, no jump in pose
or position, no camera shake.
```

## 17. One-line memory rule

> First say who and where; then write the beginning, the middle, and the
> end; keep subject, environment, and camera motion in separate
> sentences; finish with style, light, and only the stability
> constraints this shot actually needs.
