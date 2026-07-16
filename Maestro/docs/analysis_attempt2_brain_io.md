# attempt2 Brain I/O 分析(2026-07-16)

> 逐次 I/O 全文在下半部;这里是问题清单与修复对照(本轮已全部落码)。

## 这次跑出的问题

1. **剧本切在半空(致命)**:shot1 end_state = "cat is in mid-air just
   below the windowsill"。生成模型在片尾把猫冻在半空(镜尾刹车 + 一次性
   动作被切半),shot2 无从接起。
   → 修复:scene_write 加 COMPLETE-ACTION LAW(跳/坠必须在本镜内落地,
   切点只许可持续运动)+ end_state 禁止 mid-air 类悬停。
2. **shot2 没接上 shot1(机制性)**:条件本身没错(reference_video =
   shot1 尾段,junction 也正确注入),但 tiv2v_window 走的 seedance t2v
   `reference_videos` 是【参考】通道 —— 参考运动风格,不从末帧续画。
   prompt 写 "Continuing directly from @Video1" 无济于事。
   → 修复:新策略 extend_prev(bytedance/seedance-2.0/video-extend,
   官方语义 = 从输入视频末帧继续,last_image 可作目标尾帧;输出为
   原片+续段拼接,执行器裁头)顶替 tiv2v_window。
3. **描述单薄 + caption 整段粘贴**:三镜 description 都内嵌了完整素材
   caption(连 "sleeping on a windowsill" 都进了地板小跑镜头)。
   → 修复:ASSET MENTION LAW 改为只取身份词(橘白猫/白胸),照片的
   姿态/场景词禁止照搬;新增 LINKING NARRATION 要求(开头写如何接棒、
   结尾写如何交棒)。
4. **junction 机制本身工作正常**(值得记录的好消息):VLM 实况正确指出
   真实尾帧与剧本 end_state 不符,brain 的 reason 也如实引用 —— 败在
   续接原语,不在感知。
5. **修复层**:keyframe_edit_propagate 被 verifier 接受但人眼不合格
   (中间帧编辑引发前后失调)→ 已禁用;12 笔 duration=4 的级联 i2v →
   regenerate_segment 重做为 flf2v 双锚免级联(每次修复 1 笔调用)。
6. **"每次都修"**:VLM 惯出 severity 0.5 小项,任何失败项都触发修复。
   → 新旋钮 --repair-severity(低于阈值不修,荐 0.6);converged/
   quality_bar 原有机制不变(本次 shot3 即零修复收敛)。

---

# Brain I/O 报告 — `outputs/attempt2/brain_calls.jsonl`

共 15 次 brain 调用。

## #0 `window/baseline_anchor`  (—)
- usable: **True** · via: llm

<details><summary>LLM 原始输出</summary>

```
{"video_prompt":"Use @Image1 as the character reference for the orange-and-white cat. Single continuous shot in a cozy sunlit living room: the cat wakes on a warm wooden windowsill, blinks, stretches lazily, then leaps down to the floor. With tail raised, it trots across the room toward a small food bowl, arrives, stops on its own, lowers its head, and begins eating happily. Camera follows smoothly from windowsill to bowl at cat-eye level."}
```
</details>

**解析结果**

```json
{
  "video_prompt": "Use @Image1 as the character reference for the orange-and-white cat. Single continuous shot in a cozy sunlit living room: the cat wakes on a warm wooden windowsill, blinks, stretches lazily, then leaps down to the floor. With tail raised, it trots across the room toward a small food bowl, arrives, stops on its own, lowers its head, and begins eating happily. Camera follows smoothly from windowsill to bowl at cat-eye level."
}
```

## #1 `window/scene_write`  (—)
- usable: **True**
- skill: `scene_write` (7241 chars, loaded=True)

<details><summary>输入 context</summary>

```json
{
  "user_prompt": "A short film about the cat from the provided photo: the cat wakes up on a sunlit windowsill and stretches, then leaps down; it trots across the living-room floor toward its food bowl, tail up, still moving as we cut; it arrives at the bowl, stops by itself, and starts eating happily",
  "asset_catalog": [
    {
      "kind": "identity",
      "name": "example1",
      "label": "identity: example1 character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight.",
      "desc": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight.",
      "path": "/mnt/workspace1/xiangyukong/projects/VideoAgent/Maestro/resource/image/example1.png"
    }
  ],
  "episode_guidance": {
    "replay_hints": [],
    "avoid": [],
    "past_task_shapes": [],
    "n_episodes_matched": 0
  },
  "max_shots": 6
}
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"shots":[{"description":"Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up","duration_s":7,"end_state":"At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor."},{"description":"Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot","duration_s":6,"end_state":"At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward."},{"description":"Shot 3: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight continues trotting toward the food bowl, arrives, stops by itself beside it, and eats happily, low medium shot","duration_s":7,"end_state":"At the cut, the cat is stopped beside the food bowl with its head lowered into the bowl, happily eating in place."}]}
```
</details>

**解析结果**

```json
{
  "shots": [
    {
      "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
      "duration_s": 7,
      "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor."
    },
    {
      "description": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
      "duration_s": 6,
      "end_state": "At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward."
    },
    {
      "description": "Shot 3: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight continues trotting toward the food bowl, arrives, stops by itself beside it, and eats happily, low medium shot",
      "duration_s": 7,
      "end_state": "At the cut, the cat is stopped beside the food bowl with its head lowered into the bowl, happily eating in place."
    }
  ]
}
```

## #2 `window/image-plan`  (scene 1 shot 1)
- usable: **True**
- skill: `image_plan` (6161 chars, loaded=True)
- menu: none, pair_first_last, pair_reference, single_first_frame, single_reference

<details><summary>输入 context</summary>

```json
{
  "shot": {
    "label": "scene 1 shot 1",
    "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
    "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor.",
    "status": "pending",
    "image_plan": "",
    "images": [],
    "keyframe": "",
    "keyframe_source": "",
    "video": "",
    "condition_strategy": "",
    "last_score": null,
    "open_defects": null
  },
  "storyboard": [
    {
      "label": "scene 1 shot 1",
      "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
      "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor.",
      "status": "pending",
      "image_plan": "",
      "images": [],
      "keyframe": "",
      "keyframe_source": "",
      "video": "",
      "condition_strategy": "",
      "last_score": null,
      "open_defects": null
    },
    {
      "label": "scene 1 shot 2",
      "description": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
      "end_state": "At the cut, the cat is on the living-room floor, still trotting 
… <3106 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"strategy":"single_first_frame","images":[{"source":"asset_image","description":"orange and white cat sleeping on a wooden windowsill in warm sunlight"}],"reason":"the opening shot should start on the provided cat-on-windowsill image before it wakes and jumps down"}
```
</details>

**解析结果**

```json
{
  "strategy": "single_first_frame",
  "reason": "the opening shot should start on the provided cat-on-windowsill image before it wakes and jumps down",
  "images": [
    {
      "source": "asset_image",
      "description": "orange and white cat sleeping on a wooden windowsill in warm sunlight"
    }
  ]
}
```

## #3 `window/image-plan`  (scene 1 shot 2)
- usable: **True**
- skill: `image_plan` (6161 chars, loaded=True)
- menu: none, pair_first_last, pair_reference, single_first_frame, single_reference

<details><summary>输入 context</summary>

```json
{
  "shot": {
    "label": "scene 1 shot 2",
    "description": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
    "end_state": "At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward.",
    "status": "pending",
    "image_plan": "",
    "images": [],
    "keyframe": "",
    "keyframe_source": "",
    "video": "",
    "condition_strategy": "",
    "last_score": null,
    "open_defects": null
  },
  "storyboard": [
    {
      "label": "scene 1 shot 1",
      "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
      "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor.",
      "status": "keyframed",
      "image_plan": "single_first_frame",
      "images": [
        {
          "role": "first_frame",
          "source": "asset_image",
          "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
        }
      ],
      "keyframe": "/mnt/workspace1/xiangyukong/projects/VideoAgent/Maestro/resource/image/example1.png",
      "keyframe_source": "asset_image",
      "video": "",
      "condition_strategy": "",
      "last_score": null,
      "open_defects": null
    },
    {
   
… <3453 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"strategy":"single_reference","images":[{"source":"asset_image","description":"character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight"}],"reason":"the shot should continue the action freely while keeping the orange and white cat recognizable from the provided character image"}
```
</details>

**解析结果**

```json
{
  "strategy": "single_reference",
  "reason": "the shot should continue the action freely while keeping the orange and white cat recognizable from the provided character image",
  "images": [
    {
      "source": "asset_image",
      "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight"
    }
  ]
}
```

## #4 `window/image-plan`  (scene 1 shot 3)
- usable: **True**
- skill: `image_plan` (6161 chars, loaded=True)
- menu: none, pair_first_last, pair_reference, single_first_frame, single_reference

<details><summary>输入 context</summary>

```json
{
  "shot": {
    "label": "scene 1 shot 3",
    "description": "Shot 3: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight continues trotting toward the food bowl, arrives, stops by itself beside it, and eats happily, low medium shot",
    "end_state": "At the cut, the cat is stopped beside the food bowl with its head lowered into the bowl, happily eating in place.",
    "status": "pending",
    "image_plan": "",
    "images": [],
    "keyframe": "",
    "keyframe_source": "",
    "video": "",
    "condition_strategy": "",
    "last_score": null,
    "open_defects": null
  },
  "storyboard": [
    {
      "label": "scene 1 shot 1",
      "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
      "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor.",
      "status": "keyframed",
      "image_plan": "single_first_frame",
      "images": [
        {
          "role": "first_frame",
          "source": "asset_image",
          "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
        }
      ],
      "keyframe": "/mnt/workspace1/xiangyukong/projects/VideoAgent/Maestro/resource/image/example1.png",
      "keyframe_source": "asset_image",
      "video": "",
      "condition_strategy": "",
      "last_score": null,
      "open_defects":
… <3702 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"strategy":"single_reference","images":[{"source":"asset_image","description":"character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."}],"reason":"the cat remains the main subject and needs identity consistency while the shot can compose the food-bowl action freely"}
```
</details>

**解析结果**

```json
{
  "strategy": "single_reference",
  "reason": "the cat remains the main subject and needs identity consistency while the shot can compose the food-bowl action freely",
  "images": [
    {
      "source": "asset_image",
      "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
    }
  ]
}
```

## #5 `window/generation-condition`  (scene 1 shot 1)
- usable: **True**
- skill: `window_generation` (11309 chars, loaded=True)
- menu: i2v_keyframe, t2v

<details><summary>输入 context</summary>

```json
{
  "shot": {
    "label": "scene 1 shot 1",
    "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
    "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor.",
    "status": "keyframed",
    "image_plan": "single_first_frame",
    "images": [
      {
        "role": "first_frame",
        "source": "asset_image",
        "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
      }
    ],
    "keyframe": "/mnt/workspace1/xiangyukong/projects/VideoAgent/Maestro/resource/image/example1.png",
    "keyframe_source": "asset_image",
    "video": "",
    "condition_strategy": "",
    "last_score": null,
    "open_defects": null
  },
  "prev_shot": null,
  "junction": {
    "prev_last_frame_actual": null,
    "prev_end_state_script": null,
    "required_end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor."
  },
  "slots_by_strategy": {
    "t2v": [],
    "i2v_keyframe": [
      {
        "slot": "FIRST_FRAME",
        "referenceable": false,
        "content": "user asset: character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
      }
    ]
  },
  "storyboard": [
    {
      "label": "scene 1 shot 1",
      "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleep
… <4305 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"strategy":"i2v_keyframe","reason":"the first shot has a user-asset first frame and no previous shot, so it must open on the cat image","video_prompt":"Opening exactly on the orange and white cat asleep on the wooden windowsill in warm sunlight, in a cozy living-room close-up. The cat slowly wakes, blinks, gives a big natural yawn, then stretches its front legs and arches its back on the sill. After the stretch, it gathers itself and leaps down from the wooden windowsill toward the living-room floor. End the shot with the cat still in mid-air just below the sill, moving downward; it does not land, slow down, or settle before the cut.","use_prev_tail_video":false}
```
</details>

**解析结果**

```json
{
  "strategy": "i2v_keyframe",
  "reason": "the first shot has a user-asset first frame and no previous shot, so it must open on the cat image",
  "video_prompt": "Opening exactly on the orange and white cat asleep on the wooden windowsill in warm sunlight, in a cozy living-room close-up. The cat slowly wakes, blinks, gives a big natural yawn, then stretches its front legs and arches its back on the sill. After the stretch, it gathers itself and leaps down from the wooden windowsill toward the living-room floor. End the shot with the cat still in mid-air just below the sill, moving downward; it does not land, slow down, or settle before the cut.",
  "use_prev_tail_video": false
}
```

## #6 `window/prompt_enhance`  (scene 1 shot 1)
- usable: **True**
- skill: `prompt_enhancer` (6416 chars, loaded=True)
- ref_audit: `{"ok": true, "unknown": [], "appended": []}`

<details><summary>输入 context</summary>

```json
{
  "shot_description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
  "strategy": "i2v_keyframe",
  "model_family": "seedance_i2v",
  "conditions": [
    {
      "kind": "image",
      "slot": "FIRST_FRAME",
      "referenceable": false,
      "description": "user asset: character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
    },
    {
      "kind": "state",
      "role": "required_end_state",
      "description": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor."
    }
  ],
  "current_prompt": "Opening exactly on the orange and white cat asleep on the wooden windowsill in warm sunlight, in a cozy living-room close-up. The cat slowly wakes, blinks, gives a big natural yawn, then stretches its front legs and arches its back on the sill. After the stretch, it gathers itself and leaps down from the wooden windowsill toward the living-room floor. End the shot with the cat still in mid-air just below the sill, moving downward; it does not land, slow down, or settle before the cut."
}
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"video_prompt":"From this exact close-up living-room frame, the sleeping orange-and-white cat wakes in the warm sunlight, blinks, opens into a wide natural yawn, then stretches its front legs and arches its back on the wooden windowsill. It gathers its paws under itself and leaps down toward the living-room floor. End with the cat still in mid-air just below the sill, moving downward — it does not land, slow down, or settle."}
```
</details>

**解析结果**

```json
{
  "video_prompt": "From this exact close-up living-room frame, the sleeping orange-and-white cat wakes in the warm sunlight, blinks, opens into a wide natural yawn, then stretches its front legs and arches its back on the wooden windowsill. It gathers its paws under itself and leaps down toward the living-room floor. End with the cat still in mid-air just below the sill, moving downward — it does not land, slow down, or settle."
}
```

## #7 `repair/decide`  (shot 0)
- usable: **True**
- skill: `orchestrator` (14353 chars, loaded=None)
- menu: accept, depth_edit, edit_clip, extend_clip, frame_to_frame, keyframe_edit, keyframe_edit_propagate, regenerate, regenerate_segment, style_edit

<details><summary>输入 context</summary>

```json
{
  "shot_prompt": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
  "review_brief": {
    "headline": "Worst: Does the cat leap down toward the floor? on 'the clip' frames 55-56 (severity 0.50, multi)",
    "issues": [
      {
        "title": "Does the cat leap down toward the floor?",
        "entity": "",
        "frame_range": [
          55,
          56
        ],
        "severity": 0.5,
        "fix_classes": [
          "keyframe_edit",
          "localized_regen"
        ],
        "fix_modality": "content",
        "evidence": [
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Does the cat leap down toward the floor?"
          },
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Does the video end with the cat in mid-air below the sill?"
          }
        ],
        "agreement": "multi",
        "confidence": 0.6,
        "key": "semantic|clip|content|q3",
        "status": "initial",
        "id": "I-1"
      }
    ],
    "conflicts": [],
    "progress": {
      "fixed": [],
      "new": [],
      "regressed": [],
      "unchanged": []
    },
    "do_not_repeat": [],
    "brief_nl": "The main issue is whether the cat leaps down toward the floor in frame_range [55, 56]. The same semantic critic also questions whether the video ends with the cat in mid-air below the sill. This is an opinion-only conte
… <3313 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"tool":"regenerate_segment","args":{"frame_start":55,"frame_end":56,"hint":"The orange and white cat on the wooden windowsill was not clearly leaping down; as it finishes stretching, it launches off the sill and ends fully airborne, moving downward toward the living-room floor, same warm sunlight, cozy close-up camera, wooden sill, and cat markings."},"reason":"Tail content defect: cut before the unclear ending and regrow the final moment so the cat is visibly airborne and descending."}
```
</details>

**解析结果**

```json
{
  "tool": "regenerate_segment",
  "args": {
    "frame_start": 55,
    "frame_end": 56,
    "hint": "The orange and white cat on the wooden windowsill was not clearly leaping down; as it finishes stretching, it launches off the sill and ends fully airborne, moving downward toward the living-room floor, same warm sunlight, cozy close-up camera, wooden sill, and cat markings."
  },
  "reason": "Tail content defect: cut before the unclear ending and regrow the final moment so the cat is visibly airborne and descending.",
  "via": "llm"
}
```

## #8 `repair/decide`  (shot 0)
- usable: **True**
- skill: `orchestrator` (14353 chars, loaded=None)
- menu: accept, depth_edit, edit_clip, extend_clip, frame_to_frame, keyframe_edit, keyframe_edit_propagate, regenerate, regenerate_segment, style_edit

<details><summary>输入 context</summary>

```json
{
  "shot_prompt": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
  "review_brief": {
    "headline": "Worst: Does the cat leap down toward the floor? on 'the clip' frames 55-56 (severity 0.50, multi)",
    "issues": [
      {
        "title": "Does the cat leap down toward the floor?",
        "entity": "",
        "frame_range": [
          55,
          56
        ],
        "severity": 0.5,
        "fix_classes": [
          "keyframe_edit",
          "localized_regen"
        ],
        "fix_modality": "content",
        "evidence": [
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Does the cat leap down toward the floor?"
          },
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Does the video end with the cat in mid-air below the sill?"
          }
        ],
        "agreement": "multi",
        "confidence": 0.6,
        "key": "semantic|clip|content|q3",
        "status": "unchanged",
        "id": "I-1"
      }
    ],
    "conflicts": [],
    "progress": {
      "fixed": [],
      "new": [],
      "regressed": [],
      "unchanged": [
        "semantic|clip|content|q3"
      ]
    },
    "do_not_repeat": [
      {
        "tool": "regenerate_segment",
        "args": {
          "frame_start": 55,
          "frame_end": 56,
          "hint": "The orange and white cat on the wooden windowsill was n
… <4617 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"tool":"keyframe_edit_propagate","args":{"frame_idx":56,"edit_instruction":"In the cozy living-room close-up, the orange and white cat at the wooden windowsill was not clearly leaping down. Change this moment so the cat is fully airborne below the sill, body compact and moving downward toward the floor after launching. Keep the same warm sunlight, camera angle, wooden sill, and cat markings."},"reason":"The rejected tail segment regeneration cannot be repeated, so edit the final content frame to show the required airborne leap."}
```
</details>

**解析结果**

```json
{
  "tool": "keyframe_edit_propagate",
  "args": {
    "frame_idx": 56,
    "edit_instruction": "In the cozy living-room close-up, the orange and white cat at the wooden windowsill was not clearly leaping down. Change this moment so the cat is fully airborne below the sill, body compact and moving downward toward the floor after launching. Keep the same warm sunlight, camera angle, wooden sill, and cat markings."
  },
  "reason": "The rejected tail segment regeneration cannot be repeated, so edit the final content frame to show the required airborne leap.",
  "via": "llm"
}
```

## #9 `window/generation-condition`  (scene 1 shot 2)
- usable: **True**
- skill: `window_generation` (11309 chars, loaded=True)
- menu: multi_image_fusion, t2v, t2v_own_refs, ti2v_prev_last, ti2v_prev_plus_keyframe, tiv2v_window

<details><summary>输入 context</summary>

```json
{
  "shot": {
    "label": "scene 1 shot 2",
    "description": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
    "end_state": "At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward.",
    "status": "keyframed",
    "image_plan": "single_reference",
    "images": [
      {
        "role": "reference",
        "source": "asset_image",
        "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
      }
    ],
    "keyframe": "",
    "keyframe_source": "",
    "video": "",
    "condition_strategy": "",
    "last_score": null,
    "open_defects": null
  },
  "prev_shot": {
    "label": "scene 1 shot 1",
    "description": "Shot 1: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight wakes, yawns, stretches, and leaps down, cozy living-room close-up",
    "end_state": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor.",
    "status": "generated_with_defects",
    "image_plan": "single_first_frame",
    "images": [
      {
        "role": "first_frame",
        "source": "asset_image",
        "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
      }
    ],
    "keyframe": "/mnt/workspace1/xiangyukong/projects/VideoAgent/Mae
… <7246 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"strategy":"tiv2v_window","reason":"the scene continues, but the actual last frame shows the cat moving left rather than mid-air, so continue that motion while preserving the user cat asset","video_prompt":"Continue @Video1's motion seamlessly from the actual last frame: the orange tabby cat is on the right side of the living room moving forward-left toward a wooden structure. Use @Image1, the user's orange-and-white cat sleeping on a wooden windowsill in warm sunlight, as the identity and coat-pattern reference for the same cat. The cat plants its paws on the living-room floor, pivots naturally around the wooden structure, raises its tail, and trots toward its food bowl with a smooth tracking camera following at low height. By the end of the shot the cat is on the living-room floor still trotting rightward toward the food bowl with tail up; it remains in motion as the shot ends and does not slow down or settle.","use_prev_tail_video":false}
```
</details>

**解析结果**

```json
{
  "strategy": "tiv2v_window",
  "reason": "the scene continues, but the actual last frame shows the cat moving left rather than mid-air, so continue that motion while preserving the user cat asset",
  "video_prompt": "Continue @Video1's motion seamlessly from the actual last frame: the orange tabby cat is on the right side of the living room moving forward-left toward a wooden structure. Use @Image1, the user's orange-and-white cat sleeping on a wooden windowsill in warm sunlight, as the identity and coat-pattern reference for the same cat. The cat plants its paws on the living-room floor, pivots naturally around the wooden structure, raises its tail, and trots toward its food bowl with a smooth tracking camera following at low height. By the end of the shot the cat is on the living-room floor still trotting rightward toward the food bowl with tail up; it remains in motion as the shot ends and does not slow down or settle.",
  "use_prev_tail_video": false
}
```

## #10 `window/prompt_enhance`  (scene 1 shot 2)
- usable: **True**
- skill: `prompt_enhancer` (6416 chars, loaded=True)
- ref_audit: `{"ok": true, "unknown": [], "appended": []}`

<details><summary>输入 context</summary>

```json
{
  "shot_description": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
  "strategy": "tiv2v_window",
  "model_family": "seedance_t2v",
  "conditions": [
    {
      "kind": "video",
      "slot": "@Video1",
      "referenceable": true,
      "description": "the previous shot's tail segment — the ongoing motion this shot continues"
    },
    {
      "kind": "image",
      "slot": "@Image1",
      "referenceable": true,
      "description": "user asset: character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
    },
    {
      "kind": "state",
      "role": "opening_state_actual",
      "description": "An orange tabby cat is positioned on the right side of the frame, moving forward and to the left towards a wooden structure on the left."
    },
    {
      "kind": "state",
      "role": "previous_end_state_script",
      "description": "At the cut, the cat is in mid-air just below the wooden windowsill, moving downward toward the living-room floor."
    },
    {
      "kind": "state",
      "role": "required_end_state",
      "description": "At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward."
    }
  ],
  "current_prompt": "Continue @Video1's motion seamlessly from the actual last frame: the orange tabby cat is on the right side of the living room moving forward-left toward a wooden structure. Use @Image1, the user
… <2127 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"video_prompt":"Continuing directly from @Video1, open on the actual state: an orange tabby cat on the right side moves forward-left toward a wooden structure in the living room. Use @Image1 for the same orange-and-white cat’s coat and identity from the warm windowsill photo. The cat completes its downward landing onto the wooden floor, pivots around the structure, lifts its tail, and trots rightward toward its food bowl as a low tracking camera follows in warm sunlight, still trotting as the shot ends — it does not slow down or settle."}
```
</details>

**解析结果**

```json
{
  "video_prompt": "Continuing directly from @Video1, open on the actual state: an orange tabby cat on the right side moves forward-left toward a wooden structure in the living room. Use @Image1 for the same orange-and-white cat’s coat and identity from the warm windowsill photo. The cat completes its downward landing onto the wooden floor, pivots around the structure, lifts its tail, and trots rightward toward its food bowl as a low tracking camera follows in warm sunlight, still trotting as the shot ends — it does not slow down or settle."
}
```

## #11 `repair/decide`  (shot 1)
- usable: **True**
- skill: `orchestrator` (14353 chars, loaded=None)
- menu: accept, depth_edit, edit_clip, extend_clip, frame_to_frame, keyframe_edit, keyframe_edit_propagate, regenerate, regenerate_segment, style_edit

<details><summary>输入 context</summary>

```json
{
  "shot_prompt": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
  "review_brief": {
    "headline": "Worst: Does the cat trot toward its food bowl? on 'the clip' frames 0-36 (severity 0.50, single)",
    "issues": [
      {
        "title": "Does the cat trot toward its food bowl?",
        "entity": "",
        "frame_range": [
          0,
          36
        ],
        "severity": 0.5,
        "fix_classes": [
          "keyframe_edit",
          "localized_regen"
        ],
        "fix_modality": "content",
        "evidence": [
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Does the cat trot toward its food bowl?"
          }
        ],
        "agreement": "single",
        "confidence": 0.6,
        "key": "semantic|clip|content|q0",
        "status": "initial",
        "id": "I-1"
      },
      {
        "title": "Is the motion temporally consistent without sudden flips or jumps?",
        "entity": "",
        "frame_range": [
          47,
          48
        ],
        "severity": 0.5,
        "fix_classes": [
          "keyframe_edit",
          "localized_regen"
        ],
        "fix_modality": "content",
        "evidence": [
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Is the motion temporally consistent without sudden flips or jumps?"
          }
        ],
       
… <3865 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"tool":"regenerate_segment","args":{"frame_start":0,"frame_end":36,"hint":"The orange and white cat lands from the warm wooden windowsill onto the living-room floor and clearly trots rightward toward a visible food bowl placed along its path. The bowl must be the destination, not missing or off-direction; keep the same sunlight, tracking camera, and cozy living room."},"reason":"Worst defect is a span-localized content issue, so cut and regrow the affected opening segment with the food bowl placed in the cat’s path."}
```
</details>

**解析结果**

```json
{
  "tool": "regenerate_segment",
  "args": {
    "frame_start": 0,
    "frame_end": 36,
    "hint": "The orange and white cat lands from the warm wooden windowsill onto the living-room floor and clearly trots rightward toward a visible food bowl placed along its path. The bowl must be the destination, not missing or off-direction; keep the same sunlight, tracking camera, and cozy living room."
  },
  "reason": "Worst defect is a span-localized content issue, so cut and regrow the affected opening segment with the food bowl placed in the cat’s path.",
  "via": "llm"
}
```

## #12 `repair/decide`  (shot 1)
- usable: **True**
- skill: `orchestrator` (14353 chars, loaded=None)
- menu: accept, depth_edit, edit_clip, extend_clip, frame_to_frame, keyframe_edit, keyframe_edit_propagate, regenerate, regenerate_segment, style_edit

<details><summary>输入 context</summary>

```json
{
  "shot_prompt": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
  "review_brief": {
    "headline": "Worst: Does the cat trot toward its food bowl? on 'the clip' frames 0-36 (severity 0.50, single)",
    "issues": [
      {
        "title": "Does the cat trot toward its food bowl?",
        "entity": "",
        "frame_range": [
          0,
          36
        ],
        "severity": 0.5,
        "fix_classes": [
          "keyframe_edit",
          "localized_regen"
        ],
        "fix_modality": "content",
        "evidence": [
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Does the cat trot toward its food bowl?"
          }
        ],
        "agreement": "single",
        "confidence": 0.6,
        "key": "semantic|clip|content|q0",
        "status": "unchanged",
        "id": "I-1"
      },
      {
        "title": "Is the motion temporally consistent without sudden flips or jumps?",
        "entity": "",
        "frame_range": [
          47,
          48
        ],
        "severity": 0.5,
        "fix_classes": [
          "keyframe_edit",
          "localized_regen"
        ],
        "fix_modality": "content",
        "evidence": [
          {
            "type": "opinion",
            "reviewer": "semantic_critic",
            "detail": "Is the motion temporally consistent without sudden flips or jumps?"
          }
        ],
     
… <5076 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"tool":"keyframe_edit_propagate","args":{"frame_idx":36,"edit_instruction":"In the warm sunlit living room, add a clearly visible cat food bowl directly ahead along the orange and white cat’s rightward trotting path. The problem is the destination is missing or off-direction; make the bowl the obvious target while keeping the same wooden windowsill, lighting, tracking camera, and cozy room."},"reason":"The worst remaining defect is content, and the rejected segment regeneration cannot be repeated, so edit the destination keyframe and propagate forward."}
```
</details>

**解析结果**

```json
{
  "tool": "keyframe_edit_propagate",
  "args": {
    "frame_idx": 36,
    "edit_instruction": "In the warm sunlit living room, add a clearly visible cat food bowl directly ahead along the orange and white cat’s rightward trotting path. The problem is the destination is missing or off-direction; make the bowl the obvious target while keeping the same wooden windowsill, lighting, tracking camera, and cozy room."
  },
  "reason": "The worst remaining defect is content, and the rejected segment regeneration cannot be repeated, so edit the destination keyframe and propagate forward.",
  "via": "llm"
}
```

## #13 `window/generation-condition`  (scene 1 shot 3)
- usable: **True**
- skill: `window_generation` (11309 chars, loaded=True)
- menu: multi_image_fusion, t2v, t2v_own_refs, ti2v_prev_last, ti2v_prev_plus_keyframe, tiv2v_window

<details><summary>输入 context</summary>

```json
{
  "shot": {
    "label": "scene 1 shot 3",
    "description": "Shot 3: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight continues trotting toward the food bowl, arrives, stops by itself beside it, and eats happily, low medium shot",
    "end_state": "At the cut, the cat is stopped beside the food bowl with its head lowered into the bowl, happily eating in place.",
    "status": "keyframed",
    "image_plan": "single_reference",
    "images": [
      {
        "role": "reference",
        "source": "asset_image",
        "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
      }
    ],
    "keyframe": "",
    "keyframe_source": "",
    "video": "",
    "condition_strategy": "",
    "last_score": null,
    "open_defects": null
  },
  "prev_shot": {
    "label": "scene 1 shot 2",
    "description": "Shot 2: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight lands on the living-room floor and trots toward its food bowl, tail up, tracking shot",
    "end_state": "At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward.",
    "status": "generated_with_defects",
    "image_plan": "single_reference",
    "images": [
      {
        "role": "reference",
        "source": "asset_image",
        "description": "character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
      }
    ],
    "keyframe": "",
    "key
… <7438 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"strategy":"ti2v_prev_plus_keyframe","reason":"the actual previous frame already shows the cat stopped and eating, so continue that state while preserving the user asset identity","video_prompt":"Open from @Image1, the previous shot's final frame: the orange and white cat is already standing at rest beside the grey food bowl on the wooden floor with its head lowered into the food. Do not resume trotting; it has already arrived. Use @Image2, the user asset showing the orange and white cat sleeping on a wooden windowsill in warm sunlight, to preserve the cat's orange-and-white coat, face, and cozy character identity. Low medium shot in warm living-room light: the cat stays beside the bowl, happily eating in place with gentle head bobs, small chewing motions, relaxed ears, and a calm tail. End with the cat still stopped beside the food bowl, head lowered into the bowl, happily eating in place.","use_prev_tail_video":false}
```
</details>

**解析结果**

```json
{
  "strategy": "ti2v_prev_plus_keyframe",
  "reason": "the actual previous frame already shows the cat stopped and eating, so continue that state while preserving the user asset identity",
  "video_prompt": "Open from @Image1, the previous shot's final frame: the orange and white cat is already standing at rest beside the grey food bowl on the wooden floor with its head lowered into the food. Do not resume trotting; it has already arrived. Use @Image2, the user asset showing the orange and white cat sleeping on a wooden windowsill in warm sunlight, to preserve the cat's orange-and-white coat, face, and cozy character identity. Low medium shot in warm living-room light: the cat stays beside the bowl, happily eating in place with gentle head bobs, small chewing motions, relaxed ears, and a calm tail. End with the cat still stopped beside the food bowl, head lowered into the bowl, happily eating in place.",
  "use_prev_tail_video": false
}
```

## #14 `window/prompt_enhance`  (scene 1 shot 3)
- usable: **True**
- skill: `prompt_enhancer` (6416 chars, loaded=True)
- ref_audit: `{"ok": true, "unknown": [], "appended": []}`

<details><summary>输入 context</summary>

```json
{
  "shot_description": "Shot 3: scene 1 — the cat matching character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight continues trotting toward the food bowl, arrives, stops by itself beside it, and eats happily, low medium shot",
  "strategy": "ti2v_prev_plus_keyframe",
  "model_family": "seedance_t2v",
  "conditions": [
    {
      "kind": "image",
      "slot": "@Image1",
      "referenceable": true,
      "description": "the previous shot's final frame (the exact moment to continue from)"
    },
    {
      "kind": "image",
      "slot": "@Image2",
      "referenceable": true,
      "description": "user asset: character: an orange and white cat sleeping on a wooden windowsill in the warm sunlight."
    },
    {
      "kind": "state",
      "role": "opening_state_actual",
      "description": "An orange and white cat wearing a blue collar is positioned in the center of the frame, standing at rest as it leans its head down into a grey bowl of food on a wooden floor."
    },
    {
      "kind": "state",
      "role": "previous_end_state_script",
      "description": "At the cut, the cat is on the living-room floor, still trotting with tail up toward the food bowl, moving rightward."
    },
    {
      "kind": "state",
      "role": "required_end_state",
      "description": "At the cut, the cat is stopped beside the food bowl with its head lowered into the bowl, happily eating in place."
    }
  ],
  "current_prompt": "Open from @Image1, the previous shot's final frame: the orange and white cat is already standing at rest beside the grey food 
… <2184 chars total>
```
</details>

<details><summary>LLM 原始输出</summary>

```
{"video_prompt":"Continue directly from @Image1, the previous final frame: the orange-and-white cat with a blue collar is centered on the wooden floor, stopped beside the grey food bowl with its head lowered into the food. Reference @Image2 for the cat’s orange-and-white coat, face, and warm cozy identity. The earlier tail-up rightward trot has completed; in a low medium shot, warm living-room sunlight catches gentle head bobs and chewing as the cat happily eats in place until the cut."}
```
</details>

**解析结果**

```json
{
  "video_prompt": "Continue directly from @Image1, the previous final frame: the orange-and-white cat with a blue collar is centered on the wooden floor, stopped beside the grey food bowl with its head lowered into the food. Reference @Image2 for the cat’s orange-and-white coat, face, and warm cozy identity. The earlier tail-up rightward trot has completed; in a low medium shot, warm living-room sunlight catches gentle head bobs and chewing as the cat happily eats in place until the cut."
}
```
