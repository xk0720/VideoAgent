# 总结 — 窗口式生成 + 双层记忆 + 工具库(2026-07-10 ~ 07-13 这一轮)

> 需求文档:docs/REQUIREMENTS_window_memory.md(你逐条裁决过)。
> 本文回答:每条需求/裁决 → 实现在哪 → 怎么验证的。当前 379 测试全绿。

---

## 1. 你的裁决 → 落地对照

| 裁决 | 落地 | 验证 |
|---|---|---|
| **Q1** 多图输入模型:深度调研,有则都实现 | 调研报告 docs/research/wavespeed_multi_image_2026_07.md(agent 18+ 官方页逐页核验)。实现:① seedance-2.0 `reference_images` 通道接线(≤9 图,@ImageN)——**仅 t2v 端点验证**;i2v+refs 是未验证 schema → 代码主动丢弃+loud 告警,不硬编码;② `multi_image_to_video()` → **kling-video-o1/reference-to-video**(≤7 图,带 video 参考自动收缩到 4;legacy v1.6 保留可配);③ 窗口新策略 `ti2v_prev_plus_keyframe`(t2v+refs 软锚:@Image1 续接起点 + @Image2 目标构图,一次两张图、不锁任何帧)和 `multi_image_fusion`;兜底优先级按硬锚>软锚重排。kling-v3.0-4k 双硬锚+element_list、pixverse-c1 标签 refs、vidu r2v 登记缺口台账 | test_video_gen_backends(t2v refs/9 图截断/i2v 丢弃/legacy loud/o1 7↔4 收缩/v1.6 legacy)+ test_window_loop(门控、执行、降级链) |
| **Q2** 尾段 2s 不写死 | `window.tail_seconds`(configs/basic.yaml)+ demo `--tail-seconds` | 参数流到 `_cut_tail` |
| **Q3** 强制 VLM 输出 frame_range | assess_semantic 提示要求逐失败项给 [frame_start, frame_end](原片帧号);ChecklistItem 加 `frame_range`;DefectReport 用它把语义缺陷也变成段级可修;VLM 没给/给废 → 诚实回退整镜级;mock 无像素证据永远不给帧号 | test_defect_report + semantic critic 日志记 localized 数 |
| **Q4** Summarizer/brain/Verifier 三角分工保留;子循环逻辑要准 | 分工未动。子循环核对出一个真问题并修掉:窗口造的候选没带 keyframes,内层 keyframe_edit 工具会空转——现在窗口候选挂上本 shot 的真实 keyframe | test_window_loop(cand_keyframes) |
| **skills 用 folder+files** | 全部改成 `<skill>/SKILL.md` 标准形态(loader 支持 frontmatter/文件夹名解析、跳过空草稿);你建的 `scene_write`、`video_retrieval` 两个空文件已填成真实操作手册 | test_skills_loader |
| **先建完整工具库** | docs/TOOL_LIBRARY.md:五层权威清单(修复 brain 12 工具 / 窗口 brain 4+7 策略 / 后端 14 能力+端点 / 确定性工具箱 12 件 / 物理测量链)+ 活缺口台账(发现即登记、补齐即划掉)。盘点直接抓出两个真缺口并当场补掉:reference_images 未接线、keyframe_edit 无真实后端(现为 seedream-v4/edit,拒收非图片桩) | 每条能力都有对应单测 |

## 2. 需求 R1/R2/R3 的最终形态

**R1 任务台账(StoryboardMemory)**:按时间顺序的 shot 条目
(label/描述/keyframe+来源/视频/状态机/生成条件/追加式评审轨迹/修复动作/
物理轨迹),每步原子落盘 `storyboard.json`,`to_brain_json()` 喂 brain。

**R2 长期记忆(EpisodeMemory)——"可执行化"的实现**:每条 episode 存
replay 表(Verifier 接受过的 per-shot 策略 → 相似新任务直接采纳,
via=episode,零 LLM 消耗)+ avoid 表(失败策略+原因 → 注入 brain 当禁令)。
good/bad 完全由客观收敛状态判定;bad episode 里收敛镜头的策略仍进 replay。
现在 replay/avoid 行还带 `decided_strategy` / `degraded_from`,能区分
"策略本身不行"和"策略没跑成、降级顶上"。

**R3 窗口大循环(generate_movie_windowed)**:
§A playwriting→台账 → §B 逐 shot keyframe 策略(4 选 1,能力+素材门控)→
§C 逐镜条件策略(7 选 1,episode→LLM→确定性三层决策,每层留痕)→
§D 每镜进现有评审-定位-修复-裁决子循环(initial_candidates 接入,评审意见
嵌入台账轨迹)→ §E 时间顺序合成 → §M 收工蒸馏 episode。
对抗审查确认的 2 个记账诚实性 bug 已修:异常降级必写 degraded_from;
条件按锦标赛胜出 seed 归因(新增 SelfImproveResult.initial_winner),
分歧时保留 per_seed 全流水。

## 3. 这一轮的提交

- `5865ecd` 窗口大循环 + 双层记忆(初版)
- `9070ec4` Q2-Q4 裁决 + 2 个诚实性修复 + folder 形态 skills + 真实 keyframe 编辑
- `2d6dfcc` Q1 落地:多图条件(调研核验)+ 工具库盘点

## 4. 已知边界(诚实声明)

- seedance-2.0 i2v 端点页对 reference_images 的显式列出未单独核验(博客称
  限额全端点统一);真跑若 400,报错正文透传会直说哪个字段;
- kling-elements(角色元素注册)、vidu reference-to-video 在缺口台账排队;
- ScreenwriterAgent 的 playwriting 仍是确定性拆条(LLM 版的操作手册已写在
  scene_write/SKILL.md,升级时照办);
- 服务器实跑(scripts/test_window_movie.py,三个 key)还没做——那是检验
  这一轮全部管线的最终一步,跑完把台账/决策流水发回来核对。
