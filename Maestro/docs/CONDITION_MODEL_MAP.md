# 条件策略 → 模型调用映射(权威表,2026-07-14)

> 用户令(任务 0):"所有可能的情况与调用模型的名称的对应关系肯定都是确定的,
> 先调研、写进文档、再按文档改代码。"
> 本表是**规范**:代码(`window_loop._generate_with_condition` /
> `orchestrator.execute` / `WaveSpeedClient`)必须与本表一致,改动先改表。
> 每行的字段名都来自官方 schema 核验
> (docs/research/wavespeed_api_reference_2026_07.md、wavespeed_multi_image_2026_07.md)。

## 0. 一眼看懂:为什么你的 run 里"大多数调用是 image-to-video"

你那次 run 的全部 to-video 调用(共 ~10 次):

| 调用 | 期望模型 | 是否正确 |
|---|---|---|
| shot0 初始(i2v_keyframe) | seedance-2.0/**image-to-video** | ✓ 正确(t2i 关键帧当首帧,就该 i2v) |
| shot1/shot2 初始(tiv2v_window,无关键帧) | seedance-2.0/**text-to-video** + `reference_videos` | ✓ 正确(t2v 端点;dashboard 里应显示 text-to-video) |
| shot0 修复 edit_clip | seedance-2.0/**video-edit** | ✓ 正确 |
| shot0 修复 keyframe_edit | seedream-v4/edit → **image-to-video** | ✓ 正确(改完关键帧必然 i2v 重生成) |
| shot1/shot2 修复 regenerate_segment ×3 | **image-to-video**(段首帧锚定)(+级联段 i2v) | ✓ 设计如此:剪刀式修复靠帧锚定,必然 i2v |
| shot1 修复 keyframe_edit_propagate | seedream-v4/edit → **image-to-video** | ✓ 同上 |

所以 "大多数是 image-to-video" 的直接原因:**修复调用占大头,而所有帧锚定
修复(剪刀类 + 关键帧类)按设计就是 i2v**。初始生成里只有 shot0 是 i2v。
这不是 bug;真正的 bug 是下面 §2 的一条未触发路径(本次 run 恰好没踩到,
因为 shot1/shot2 没有关键帧)。现在每次调用的 model id + 参数都会写进
`<out_dir>/wavespeed_calls.jsonl` + 终端 INFO 日志,可直接核对。

## 1. 窗口初始生成:9 个条件策略 → 模型(确定性映射)

所有 seedance 路线的机械字段(执行器补齐,brain 不碰):
`resolution`(config,默认 480p)、`generate_audio`(默认 false)、
`duration`(brain 规划 3-8s → 按模型时长域 snap,见 §3)、
图片/视频一律先 `POST /media/upload/binary` 换 URL。

| # | strategy | 条件输入 | 模型 id(唯一) | payload 关键字段 | 锚定强度 |
|---|---|---|---|---|---|
| 1 | `t2v` | 纯文本 | `bytedance/seedance-2.0/text-to-video` | `prompt, duration, resolution, generate_audio, aspect_ratio` | 无 |
| 2 | `i2v_keyframe` | 首帧角色图 | `bytedance/seedance-2.0/image-to-video` | `prompt, image(URL), duration, resolution, generate_audio` | 硬锁开场帧 |
| 3 | `flf2v_own_pair` | 本镜首+尾双图 | `bytedance/seedance-2.0/image-to-video` | `prompt, image, last_image, duration, …` | 硬锁两端 |
| 4 | `t2v_own_refs` | 参考角色图 ≤9 | `bytedance/seedance-2.0/text-to-video` | `prompt(@ImageN 提及), reference_images[], …` | 软(身份/构图) |
| 5 | `ti2v_prev_last` | 上镜尾帧 | `bytedance/seedance-2.0/image-to-video` | `image = 上镜尾帧` | 硬锁开场帧 |
| 6 | `flf2v_bridge` | 上镜尾帧 → 本镜图 | `bytedance/seedance-2.0/image-to-video` | `image = 上镜尾帧, last_image = 本镜图` | 硬锁两端 |
| 7 | `ti2v_prev_plus_keyframe` | 上镜尾帧 + 本镜图(软) | `bytedance/seedance-2.0/text-to-video` | `reference_images = [尾帧, 本镜图…](@Image1 续接点, @Image2… 目标构图)` | 软 |
| 8 | `extend_prev`(2026-07-16 顶替 tiv2v_window) | 上镜尾段**视频**(+ 可选 'last' 角色图) | `bytedance/seedance-2.0/video-extend` | `prompt(只写接下来 + 维持清单), video = 上镜尾段(从其末帧续画), last_image = 'last' 角色图(可选目标尾帧), duration, resolution, generate_audio=false` | **真续接**(末帧起画;输出=原片+续段拼接,执行器裁头) |
| 9 | `multi_image_fusion` | 2..7 张图(+ 可选尾段视频) | `kwaivgi/kling-video-o1/reference-to-video` | `prompt, images[](≤7;带 video ≤4), video?, duration{5,10}, aspect_ratio, keep_original_sound` | 软(多图融合) |

规则(硬编码在后端,违反即 RuntimeError,不允许静默):

- `reference_images` / `reference_videos` **只存在于 seedance-2.0 的
  text-to-video 端点**(官方 schema:≤9 图 + ≤3 视频(每段 ≤15s) + ≤3 音频,
  合计 ≤12 文件;@ImageN/@VideoN 提及)。image-to-video 端点 schema 只有
  `image` + `last_image` —— **i2v × refs/ref_videos 是未验证组合,后端直接
  拒绝**(refs:丢弃+告警;ref_videos:抛错,因为丢了窗口条件等于换策略)。
- #8 换代记录(2026-07-16,attempt2 实证):旧 tiv2v_window 走 t2v 的
  `reference_videos` —— 那是【参考】通道,参考运动风格,**不从末帧续画**,
  prompt 写 "Continuing from @Video1" 也无济于事。真续接原语 =
  video-extend(官方语义:generation continues from last frame;
  `last_image` 可选目标尾帧;**输出为输入+续段拼接**,计费只算新段,
  执行器必须裁掉头部输入时长)。tiv2v_window 从菜单退役,执行分支保留
  兼容旧 episode。
- #9 是唯一的 kling 路线(多图融合本来就是它的能力);带 video 时图上限
  自动 7→4(官方规则)。

## 2. 修复工具 → 模型(orchestrator.execute / propagate_repair)

| 工具 | 调用链 | 模型 id | 关键参数 |
|---|---|---|---|
| `regenerate` | generator.run(无锚) | `bytedance/seedance-2.0/text-to-video` | `prompt(+fix hint), duration=spec.duration` |
| `keyframe_edit` | image_edit → generator.run(first_frame=改后图) | `bytedance/seedream-v4/edit` → `…/image-to-video` | `image=改后关键帧` |
| `regenerate_segment`(2026-07-16 重做) | propagate_repair:interior 跨度 → **flf2v 双锚**(左邻【原始】尾帧 + 右邻【原始】首帧)一笔重生,**免级联**(尾锚=下游原开头);tail 跨度 → i2v 从左锚;head 跨度 → 左锚=条件首帧图,没有则诚实 None(brain 改选整镜工具) | `bytedance/seedance-2.0/image-to-video` | `image, last_image, duration=段长(snap 后 _fit_to_seconds 回贴)` |
| ~~`keyframe_edit_propagate`~~ | **已禁用**(2026-07-16 裁决:改中间帧导致前后失调;菜单摘除,代码保留) | — | — |
| ~~`frame_to_frame`~~ | **已摘除**(与新 regenerate_segment 语义重复) | — | — |
| `edit_clip` backend=seedance | edit_video | `bytedance/seedance-2.0/video-edit` | `prompt, video(URL), resolution`(≤15s 输入) |
| `edit_clip` backend=runway | edit_video | `runwayml/gen4-aleph` | `prompt, video, aspect_ratio`(URL-only,base64 会 400) |
| `depth_edit` | edit_video(vace, depth) | `wavespeed-ai/wan-2.1-14b-vace` | `task=depth, video, prompt, size` |
| `style_edit` | edit_video(runway, style 框架化 prompt) | `runwayml/gen4-aleph` | 同 runway |
| `extend_clip` | extend | `bytedance/seedance-2.0/video-extend` | `prompt, video, duration, resolution` |
| `simulate_reference` | Genesis 仿真 → generate(reference_video=仿真片) | `bytedance/seedance-2.0/text-to-video` | `reference_videos=[仿真参考]` |
| `adjust_prompt` / `retrieve_replace` / `accept` | 无生成 API 调用 | — | — |
| (窗口关键帧阶段 t2i) | text_to_image | `wavespeed-ai/flux-kontext-pro/text-to-image` | `prompt, num_images=1, aspect_ratio` |

旧账清偿(2026-07-16):头部段修复不再锚定缺陷片自身第 0 帧 ——
左锚改用该镜条件里的首帧角色图(clip.conditioning,本镜本该开在这张图上);
没有可用首帧图 → propagate_repair 诚实返回 None,brain 改选整镜工具。
("landed on the counter" 一类头部修复丢续接条件的问题就此关闭。)

## 3. duration 规则(任务 1 + 2026-07-14 追加裁决:4-10s,不输出就不传)

1. **brain 规划**:scene_write 逐镜输出 `duration_s`(**整数 4-10s,写死**),
   按"这个动作需要多久"判断。规划值存进 `spec.duration`,**每个生成调用
   都原样传入**(窗口 9 策略、generator.run、extend、propagate 段全部已传)。
2. **brain 没输出 / 输出非法 → `None` = 兼容模式**:payload **不含
   `duration` 字段**,API 用模型自己的自然默认(seedance 默认 5s)。绝不
   feed 任何我们编的数。兜底剧本层(无 LLM)同规则,全 None。
   (内部估算,如 n_frames≈duration×fps,在 None 时按 5s 估,只用于本地
   计算,绝不回传 API。)
3. **API snap(执行器,brain 不碰)**:seedance-2.0 家族时长域 = 整数
   [4,15],规划域 [4,10] 是其子集 → **原样直传,不改动**;kling-video-o1 =
   枚举 {5,10} → 向上 snap(6-9s 提交 10s,call log 里可见)。修复段
   (propagate_repair)生成后仍用 `_fit_to_seconds` 回贴段长(拼接时间轴
   不能变形);窗口整镜不做回贴——宁可镜头略长,不做 setpts 变速伤运动
   自然度。

## 3b. 基线锚点(2026-07-15 需求 1,`--baseline-anchor` 开关)

任务开始时按用户指令【一次调用】直出锚点视频。用户裁决(同日):
**只生成,不做机器对比/verifier 裁决,也不接 prompt enhancer —— 用户
自己看片对比**。路线确定性(用户设定):

| 用户素材 | 模型 id | payload |
|---|---|---|
| 无 | `bytedance/seedance-2.0/text-to-video` | `prompt` |
| 仅图片 | `bytedance/seedance-2.0/image-to-video` | `image = 第一张图`(多图时其余不用,日志留痕) |
| 有视频(可带图) | `bytedance/seedance-2.0/text-to-video` | `reference_videos = [≤3 条,每条裁到 ≤15s]` + `reference_images = [≤9 图]` |

锚点 prompt:brain LLM 把整个故事浓缩成一条 30-100 词的单镜 prompt;
LLM 不可用 → 用户指令原文(via=fallback 留痕)。锚点是附加物:任何失败
只记日志,绝不影响正流程。产物 `<out_dir>/baseline_anchor.mp4`,元信息
(route/prompt/via)在 MovieResult.baseline_anchor。

## 3c. Prompt Enhancer(2026-07-15 需求 2,`--prompt-enhancer` 开关)

可选润色 agent(`agents/prompt_enhancer.py` +
`skills/brain_skills/prompt_enhancer/SKILL.md`):输入 = shot 文字描述 +
执行器收集的【条件事实清单】(每张图的角色/描述、参考视频)+ 策略推导的
模型家族;先想"这镜怎么用这些条件",再按官方 prompt 技巧(seedance
@ImageN/@VideoN、kling "reference image N"、i2v/flf2v 只写运动等,技能
文件全量)重写 video prompt。STRICT JSON 校验失败 → 保留原 prompt
(增强层永不破坏正流程);每次调用原文进 brain_calls.jsonl
(stage=window/prompt_enhance)。

## 4. 调用日志(任务 0:每次调用可核对)

`WaveSpeedClient._run_task` 是所有 WaveSpeed 调用的唯一出口,现在每次:

- 终端:`INFO wavespeed call → <model_id> payload={…}`(base64 缩写,URL 保留);
- 文件:config `call_log` 指定路径,JSONL 逐行
  `{"ts", "event": "submit"|"completed"|"failed", "model", "payload", "out",
  "task_id", "elapsed_s", "error?"}`。
  `scripts/test_window_movie.py` 自动设为 `<out_dir>/wavespeed_calls.jsonl`。

另外 brain 每次决策的**原始输出**也落盘(debug 追加令):
`<out_dir>/brain_calls.jsonl`,逐行
`{"ts", "stage": "window/scene_write"|"window/image-plan"|
"window/generation-condition"|"repair/decide", "label"/"shot_idx",
"menu", "raw"(LLM 原文全量), "parsed"(校验后决策), "via", "usable"}`。
未跑脚本时也可用环境变量 `MAESTRO_BRAIN_LOG=<path>` 开启。

核对方法:三份文件对着看 ——
1. `brain_calls.jsonl`:brain 在这镜选了什么策略、原话是什么;
2. 本表 §1:该策略**应该**调哪个模型、payload 长什么样;
3. `wavespeed_calls.jsonl`:实际调了哪个模型、参数是什么。
三者对不上即 bug,拿着三行日志来对质。
