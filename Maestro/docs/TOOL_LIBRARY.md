# Maestro Tool Library — 完整盘点(2026-07-13)

> 用户令:"写框架之前先把所有 tool library 准确调研、建立完整;缺什么补什么。"
> 本文是权威清单:每个工具 = 谁调用 / 干什么 / 真实后端与端点 / 门控条件 /
> 诚实降级行为。分五层:①修复 brain 工具盘 ②窗口 brain 策略盘 ③生成后端
> 能力 ④确定性工具箱 ⑤物理测量链。真实端点全部经官方文档核验
> (docs/research/wavespeed_api_reference_2026_07.md 及后续多图调研)。

---

## ① 修复 brain 的工具盘(OrchestratorAgent.available_actions)

每回合从【能力+素材双重门控】后的菜单选一个;技能文件
`skills/brain_skills/orchestrator/SKILL.md` 全量解释。执行皆过 Verifier 闸门。

| 工具 | 干什么 | 真实后端 | 门控 |
|---|---|---|---|
| regenerate | 整镜带 hint 重掷 | seedance-2.0 t2v/i2v | 恒在 |
| keyframe_edit | 改一个关键帧→锚定重生成 | **seedream-v4/edit**(image_edit)+ i2v | 恒在(无真图时诚实 no-op) |
| regenerate_segment | 只重生成缺陷段→下游级联重锚 | i2v + timeline.propagate_repair | 任一视频能力 |
| keyframe_edit_propagate | 改帧→该段重生成→向下传播 | seedream-v4 + i2v + 传播 | 任一视频能力 |
| frame_to_frame | 缺陷段首尾双锚重生成→传播 | seedance-2.0 i2v(image+last_image)或 wan-flf2v;锚帧过同镜头预审门 | flf2v 能力 |
| edit_clip | 整镜就地编辑(不重掷) | seedance-2.0/video-edit(默认)\| runwayml/gen4-aleph \| wan-2.1-14b-vace | edit 能力 |
| depth_edit | 深度引导前景/背景替换 | wan-2.1-14b-vace task=depth | depth 能力 |
| style_edit | 风格重渲染 | gen4-aleph(风格化 prompt) | style 能力 |
| extend_clip | 续拍(整段视频条件延长) | seedance-2.0/video-extend 专用端点 | extend 能力 |
| simulate_reference | brain 写 scene_spec→刚体仿真出正确运动参考→条件重生成 | GenesisSimClient + seedance-2.0 reference_videos | sim 客户端 + ref_video 能力 |
| retrieve_replace | 用用户源片段整段替换 | RetrievalTool + AssetMemory | 素材库有源视频 |
| accept | 停止修复 | — | 恒在(有缺陷且有回合时被 override) |

## ② 窗口 brain 的策略盘(pipeline/window_loop.py)

**keyframe 策略(§B)**:t2i(flux-kontext-pro 文生图)/ asset_image(用户
身份/风格图)/ video_extract(源视频抽中间帧)/ none(诚实降级)。

**条件策略(§C,共 7 个)**:t2v / i2v_keyframe(本镜 keyframe 当首帧)/
ti2v_prev_last(上镜尾帧当首帧)/ flf2v_bridge(上镜尾帧→本镜 keyframe
双锚)/ tiv2v_window(上镜尾段视频当运动参考,`window.tail_seconds` 配置)/
ti2v_prev_plus_keyframe(多图:上镜尾帧当首帧 + keyframe 进
reference_images,@Image1 提及——续接且引导画面、不锁结尾)/
multi_image_fusion(多图融合:[尾帧, keyframe] 进 images 数组,无指定首帧)。

决策三层:episode replay(via=episode)→ LLM 严格 JSON(via=llm)→
确定性优先级(via=fallback);执行降级必记 degraded_from。

## ③ 生成后端能力(models/*_backends.py,单一 $WAVESPEED_API_KEY)

| 能力 | 方法 | 端点(POST /api/v3/…) | 关键约束 |
|---|---|---|---|
| t2v | generate() | bytedance/seedance-2.0/text-to-video | duration 4-15 整秒;480p-4k;aspect_ratio 16:9 等 |
| i2v | generate(first_frame=) | …/image-to-video(id 自动推导) | image 走上传 URL |
| flf2v | frame_to_frame() | seedance-2.0 i2v(image+last_image)\| wavespeed-ai/wan-flf2v | 旧模型 duration∈{5,10}, size "832*480" |
| ref_video | generate(reference_video=) | seedance-2.0 reference_videos 通道 | ≤3 视频、共 15s;legacy 模型无此通道(loud) |
| ref_images | generate(reference_images=) | seedance-2.0 reference_images 通道 | ≤9 图;@Image 提及语法;与 reference_videos 可同用(12 文件总限) |
| multi_i2v | multi_image_to_video() | kwaivgi/kling-v1.6-multi-i2v-standard | images 数组 ≤4;duration 5\|10;aspect 1:1\|16:9\|9:16 |
| edit | edit_video() | seedance-2.0/video-edit \| runwayml/gen4-aleph \| wan-2.1-14b-vace | gen4-aleph 只收 URL(base64→400);输入≤15s |
| depth / style | depth_modify() / style_transfer() | vace task=depth / gen4-aleph | — |
| extend | extend() | bytedance/seedance-2.0/video-extend | 整段视频条件,4-15s 增量 |
| t2i | text_to_image() | wavespeed-ai/flux-kontext-pro/text-to-image | 窗口 keyframe 阶段用 |
| image edit | WaveSpeedImageEditClient.edit() | bytedance/seedream-v4/edit | 拒收非图片桩;输出强制图片后缀 |
| foley | WaveSpeedAudioClient.foley() | wavespeed-ai/hunyuan-video-foley(默认)\| mmaudio-v2 | 视频走上传 URL |
| TTS | .speech() | minimax/speech-2.6-hd(默认)\| 2.5-turbo-preview | voice_id 区分大小写 |
| repaint | repaint() | — 诚实骨架(需 Sa2VA/SAM 分割,GPU) | 调用即 loud RuntimeError |

协议:提交 POST /api/v3/{model-id}(三段式 id)→ 轮询 GET
/predictions/{id}/result → outputs[0];本地媒体一律 POST
/media/upload/binary(≤300MB)换 URL。400 必带响应正文透传。

## ④ 确定性工具箱(tools/,无 LLM,ffmpeg/本地计算)

| 工具 | run() 签名要点 | 用途 |
|---|---|---|
| VideoConcatTool | (clips, out_path) | §E 合成 + 传播拼接(ffmpeg concat) |
| AssemblyTool | (script, out_path, music_path) | 全片装配(可带音轨) |
| FrameExtractTool | (video, timestamps, out_dir) | 按秒抽帧 |
| ImageOpsTool | (op, src, out, size/box) | resize/crop 等图像操作 |
| VideoProbeTool | (path) → dict | ffprobe 时长/fps/分辨率 |
| CaptioningTool | (media, kind) → str | 字幕/描述(mock;真 VLM 待换) |
| DetectionTool | (media, query, max_results) | GroundingDINO 零样本检测封装 |
| AudioGenTool | (prompt, out, kind=music/tts/foley) | ③ 音频后端的工具面 |
| MetricTool | (clip, spec, …) → scores | 指标套件(weighted_total 的来源) |
| RetrievalTool | retrieve_source_shots / identity_refs / style_refs | 素材检索(video_retrieval 技能的底座) |
| timeline.propagate_repair | (timeline, defect, video_gen, …) | 段修复+级联重锚+早停 |
| timeline._cut_tail / extract_frame / _fit_to_seconds | — | 尾段截取 / 抽帧 / 时长回贴 |

## ⑤ 物理测量链(非 AI 评审,critics/physics_consistency.py)

GroundingDINO(定位,第0帧质心种子)→ CoTracker(归一化轨迹)→
certify(可靠性门,不合格降级 VLM 层)→ laws.fit_best_law(静止/匀速/匀加速
自由重力)+ 4 异常检测器(瞬移/空中反向/能量增/加加速度尖峰)→
PhysicsVerdict{entity, frame_range, severity, source=law_verifier}。
Demo:scripts/test_physics_review.py(全轨迹落盘)。

## 缺口台账(发现即登记,补齐即划掉)

- [x] keyframe_edit 真实后端(seedream-v4/edit)— 2026-07-11 补
- [x] t2i(flux-kontext-pro)— 2026-07-10 补
- [x] 专用 video-extend 端点 — 2026-07-08 补
- [x] seedance-2.0 `reference_images` 通道接线(≤9,@ImageN)— 2026-07-13 补
- [x] kling-v1.6-multi-i2v(images≤4,multi_image_to_video)— 2026-07-13 补
- [x] 窗口多图条件策略 ×2(ti2v_prev_plus_keyframe / multi_image_fusion)— 2026-07-13 补
- [ ] kling-elements-advanced(角色元素注册→element_list 引用)— 登记,待做
- [ ] vidu reference-to-video / wan-2.7 多图 — 官方页未核验(UNVERIFIED),核验后再动
- [ ] repaint 需分割后端(Sa2VA/SAM)— 挂起(GPU)
- [ ] CaptioningTool 仍为确定性 mock — 挂起(非关键路径)
