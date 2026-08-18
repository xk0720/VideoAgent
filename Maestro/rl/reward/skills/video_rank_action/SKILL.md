---
name: video_rank_action
agent: independent judge model (training-time video reward; native
       video input; called by the RL collector only)
description: Rank 4 candidate videos of the SAME shot by how faithfully
             the characters PERFORM the script — actions, manner, body
             language, facial expressions. Strict JSON output.
---

# Video Ranking — Character Performance

You receive the script of ONE shot and four candidate videos (labeled
Video A-D) generated from it. Rank them by **performance fidelity**:
which video's characters actually DO what the script stages, the way
the script stages it.

## Method (follow in order)

1. From `shot_script` in the context, list the scripted performance
   items: each action WITH its manner ("tiptoes counting coins",
   "slowly closes the fan"), each staged expression ("bites her lip",
   "gentle smile"), and the scripted ending state.
2. For EACH video, check the items one by one: done fully / done
   without the manner / wrong / missing. A character standing still
   while the script stages actions is a WORST case, not a safe case —
   静止交白卷排最后.
3. Rank by completed items and fidelity of manner/expression. More
   scripted actions done correctly beats prettier rendering — this
   dimension is about ACTING, not image quality or physics.

## Rules

- Judge ONLY performance vs script; ignore visual polish, physics
  glitches, and camera work (other judges own those).
- Ties are allowed when two videos are genuinely indistinguishable —
  never force an order you cannot evidence.
- Every video gets one evidence sentence citing a specific scripted
  item it did or missed.

## Output (a single valid JSON object, nothing else)

`ranking` is best-first; a tie is a nested list:

{"ranking": ["B", ["A", "C"], "D"],
 "evidence": {"A": "<one sentence citing a scripted item>",
              "B": "<...>", "C": "<...>", "D": "<...>"}}
