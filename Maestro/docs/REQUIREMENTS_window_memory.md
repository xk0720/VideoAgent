# 标准需求规格 — 窗口式生成 + 双层记忆(2026-07,用户第 3/4 点)

> 用户原始需求 → 标准化条款 → 代码落点。原有代码不动,全部新增
> (唯一例外:generate_shot_orchestrated 新增可选参数 initial_candidates,
> 默认 None 时行为逐字节不变)。

## R1 任务级工作记忆(用户 3.(1))

**需求**:brain 在一次全片生成任务中维护【按时间顺序】的条目列表,每条 =
一个 shot:场景/镜头标签、文本描述、keyframe(路径+来源)、视频路径(已生成
时)、轨迹;可持续更新。

**落点**:`memory/storyboard.py` — `StoryboardMemory` / `ShotEntry`。
- 条目字段:label("scene 1 shot 2")、description、keyframe_path+source、
  video_path、status(pending→keyframed→generated→verified|
  generated_with_defects)、condition(本镜用的生成条件)、reviews(追加式
  评审轨迹)、repair_actions(修复动作流水)、physics_trajectory(测量轨迹,
  无测量=None,不伪造)。
- 每次更新原子落盘 run 目录 `storyboard.json`(可查、可续跑);
  `to_brain_json()` 是喂给 brain 的紧凑视图。

## R2 长期可执行记忆(用户 3.(2))

**需求**:任务结束把整条生成轨迹蒸馏进长期记忆,分 good/bad case;
回答"可执行化怎么实现"。

**落点**:`memory/episode_memory.py` — `EpisodeMemory` / `EpisodeRecord`。
- **可执行化的实现** = 每条 episode 存两张直接驱动决策的表:
  - `replay` 表(good 面):每 shot 被 Verifier 接受的 {keyframe 策略,
    条件策略}。下次相似任务命中 → 窗口 brain **直接采纳**(决策记
    via="episode",跳过 LLM 推理)——检索即执行;
  - `avoid` 表(bad 面):未收敛 shot 的 {策略, 失败原因}。命中 → 注入
    brain 上下文当硬约束——检索即禁止。
- good/bad 判定完全客观:全部 shot verified = good,否则 bad;bad episode
  里收敛了的 shot 策略仍进 replay(好的局部经验不陪葬)。
- 检索:prompt 关键词 Jaccard(确定性、无 embedding 依赖);JSONL 持久化。
- 与既有记忆的分工:repair skill(skill_library)= "这类缺陷怎么修好";
  episode = "这类任务怎么排产";storyboard = "本次任务眼前有什么"。

## R3 窗口式生成大循环(用户第 4 点)

**落点**:`pipeline/window_loop.py` — `generate_movie_windowed`。
§号与代码内注释一一对应:

| § | 用户原文 | 标准化 | 实现 |
|---|---|---|---|
| A | 预生成所有 shots 文本描述(playwriting) | prompt→outline→specs→台账 | 复用 Screenwriter+Director;StoryboardMemory.from_outline |
| B | keyframe 三来源:t2i / 素材图 / 素材视频抽帧 | brain 从门控菜单选策略,+第 4 项 none(诚实降级) | _keyframe_menu / _make_keyframe;t2i 为 WaveSpeedClient 新增 text_to_image(flux-kontext-pro,UniVA 验证过的 schema) |
| C | 条件策略:(1)上镜尾帧+keyframe;(2)上镜尾段视频+keyframe | 5 策略:t2v / i2v_keyframe / ti2v_prev_last(用户(1)) / flf2v_bridge(补全:双端锚) / tiv2v_window(用户(2),走 seedance-2.0 reference_videos) | _condition_menu / _generate_with_condition;尾段 ffmpeg -sseof 截取,缺 ffmpeg 整段兜底 |
| D | 每镜过 reviewer 小循环;VLM skill 定维度;定位失败帧/段;工具调用修复 | 原样复用 generate_shot_orchestrated(评审→汇总→DefectReport 定位→brain 修复→Verifier 闸门),新增 initial_candidates 入参把窗口条件产物接进去 | 评审意见+修复动作追加进台账(轨迹嵌入) |
| E | 所有 shot merge | 时间顺序 ffmpeg concat;未收敛 shot 照拼但状态如实 | VideoConcatTool |
| M | (3.(1)+3.(2) 的闭环) | 开工 guidance_for 取历史经验;收工 distill_episode | EpisodeMemory |

**brain 决策的三层回退**(每层都记录 via):episode replay 命中(via=episode)
→ LLM 严格 JSON(via=llm)→ 确定性优先级(via=fallback;菜单非空必有解,
大循环永不卡死)。Mock LLM 必然走 fallback——mock 模式不伪造"brain 决策"。

**用户没想到、我们补的点**(代码注释里逐条标了):
1. keyframe 策略第 4 项 `none`:三种来源都无输入时的诚实降级;
2. 条件策略 `flf2v_bridge`:用户(1)的强化(双端同时锚定);
3. "上一镜"取【最近已生成】而非【已 verified】:带遗留缺陷的上镜尾帧仍是
   时间上唯一正确的续接点;
4. 策略执行失败的逐级降级链(flf2v_bridge→ti2v_prev_last→i2v_keyframe→t2v),
   每次降级在 condition 里写 degraded_from,可审计;
5. 合成失败(无 ffmpeg)→ 保留单镜产物,不放占位文件冒充成片;
6. 台账/episode 全部原子写盘,进程崩溃不损坏记忆。

**Skill**:`skills/brain_skills/window_generation.md` —— 窗口 brain 的选择
逻辑(菜单、决策规则、episode 提示怎么用、严格 JSON 格式、3 个例子),和
orchestrator.md(修复 brain)并列。
