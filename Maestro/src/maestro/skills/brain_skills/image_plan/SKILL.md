---
name: image_plan
agent: window brain(pipeline/window_loop.py 的 Image Plan 决策)
description: 逐 shot 决定【要几张图、每张什么角色、每张什么来源】,并为后续视频调用写出角色化 prompt——角色锁死模型族,输出严格 JSON。
---

# Image Plan — 图片计划技能(数量 + 角色 + 来源)

## 角色
你在为【一个 shot】的视频生成做图片准备。你的决定有三层,一次给全:
1. **要几张图**:0 / 1 / 2(暂定最多两张);
2. **每张图的角色**(提前设定,角色**锁死**后续能用的视频模型族):
3. **每张图的来源**:t2i 文生图 / asset_image 用户素材检索 / video_extract
   用户视频抽帧 —— **来源可以混搭**(例:图1 用用户人物照,图2 文生图场景)。

## 角色 → 视频模型族(锁死映射,不允许错配)

| plan | 图片角色 | 后续视频调用 | payload 图片字段 |
|---|---|---|---|
| single_first_frame | 首帧锚 | seedance-2.0 i2v(ti2v) | `image` |
| single_reference | 参考(人物/物体/场景一致性) | seedance-2.0 t2v+refs 或 kling-video-o1 | `reference_images` / `images` |
| pair_first_last | 首帧+尾帧 | seedance-2.0 i2v(首尾双锚) | `image`+`last_image` |
| pair_reference | 双参考(如两个角色;角色+场景) | kling-video-o1(可再带上镜尾段 `video`) | `images` |
| none | 无图 | t2v / 上镜锚定路线 | — |

## 决策思路(不是死规则——按剧情和素材推理)

- **shot 必须从一个精确画面开场**(顺时续接、开场定格)→ first_frame。
- **shot 里有需要长相/外观一致的主体**(角色、特定物体、特定场景),但
  画面构图应该让模型自由发挥 → reference。
- **shot 的开场和收场都明确**(一个动作从 A 到 B;转场镜头)→ pair_first_last:
  图1 = 开场帧,图2 = 收场帧,两张图的描述要写成同一场景的两个时刻。
- **shot 要把多个独立元素融进一个画面**(两个角色同框;把用户的角色放进
  用户的场景)→ pair_reference。
- **素材场景举例(推理示范,不要背成规则)**:
  · 用户给了一张【背景/场景图】:该场景的第一镜可以 single_first_frame
    直接用它开场(source=asset_image);同场景后续镜头用它当
    single_reference(场景一致性)——别每镜都拿它当首帧,那会让每一镜都
    从同一个静止画面开始。
  · 用户给了一张【人物照】:人物出场的每一镜都带上它当 reference
    (single_reference 或 pair_reference 的一张);除非剧本要求"从人物
    特写定格开场"才当 first_frame。
  · 用户给了【两张人物照】(如男女主):两人同框的镜头用 pair_reference。
  · 用户给了【源视频】:video_extract 抽帧,当首帧(续用户的画面)或参考。
  · 什么都没给、纯生成:t2i;开场镜头 single_first_frame 定基调,后续镜头
    多数 none(靠上镜尾帧/尾段续接,见 window_generation 技能)——
    **不是每一镜都需要自己的图**,滥造 keyframe 反而打断连续性。
- asset_catalog 里每个素材带 kind + 描述,选 asset_image 时把检索词写进
  该图的 description(检索按关键词重叠打分)。

## 输出格式(严格 JSON,只输出这个)

{"strategy": "<菜单里的 plan 名>",
 "images": [{"source": "t2i"|"asset_image"|"video_extract",
             "description": "<t2i 的完整生图 prompt,或检索词>"}, ...],
 "reason": "<一句话>"}

- images 数量必须等于 plan 要求(single_*=1,pair_*=2,none=0/省略)。
- pair_first_last 的两条 description 必须是【同一场景的开场时刻和收场时刻】。
- t2i 的 description 是完整生图 prompt(主体+场景+光线+风格),不是一个词。

### 例 1 —— 纯生成,开场镜头
{"strategy": "single_first_frame", "images": [{"source": "t2i", "description": "a glass of water standing near the edge of a wooden kitchen table, warm morning light, photorealistic, eye-level close-up"}], "reason": "opening shot sets the look; the shot must start exactly on this framing"}

### 例 2 —— 用户给了男女主两张照片,本镜两人同框
{"strategy": "pair_reference", "images": [{"source": "asset_image", "description": "female character portrait"}, {"source": "asset_image", "description": "male character portrait"}], "reason": "both characters appear together; their faces must stay recognizable — reference pair via kling"}

### 例 3 —— 动作从 A 到 B 的转场镜头(混搭来源)
{"strategy": "pair_first_last", "images": [{"source": "video_extract", "description": "the corridor from the user's source clip"}, {"source": "t2i", "description": "the same corridor, door at the end now open, camera slightly closer, same lighting"}], "reason": "shot opens on the user's real corridor and must end on the opened door"}

### 例 4 —— 同场景第三镜,不需要自己的图
{"strategy": "none", "images": [], "reason": "mid-scene continuation — anchor on the previous shot's last frame instead of a fresh image"}
