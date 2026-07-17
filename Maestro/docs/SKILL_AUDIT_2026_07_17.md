# Skill 全面梳理(2026-07-17,12 路并行审计 → 全部修复落地)

> 用户令:"完成全部任务之后,把 skill 从头梳理一下……尤其是撰写 prompt 的
> skill,增强不同 shot 之间 character/background 及一切剧本要求 consist 的
> 东西。"审计方法:每个技能一个审计员对照【代码现实】逐条核验 + 跨镜一致性
> 专项 + 技能间矛盾专项(48 条发现;HIGH/MEDIUM 19 条全修,LOW 择要修)。

## 一、最大的发现:跨镜一致性链条缺一根主梁(已补)

**审计结论**:原链条只统一了实体的【名字】(scene_write 实体命名法则),
从未统一【外观】——身份词法则只覆盖用户素材;**生成角色**(t2i 出来的
男孩/小狗)、场景陈设、光线,每条 prompt 各写各的,同一个角色两镜两个长相
是必然,不是偶然。四个 HIGH 缺口:无生成角色的官方外观描述、无场景级陈设+
光线契约、角色离场再入场无外观载体、t2v 无锚路线零一致性注入。

**修复:cast/setting 官方描述符机制(新)**——
1. **剧本一次定稿**:scene_write 新输出 `cast`(每个反复出场的角色一条
   10-20 词官方外观:物种/体型、毛色/衣着含颜色、显著标记)+ `setting`
   (场景陈设+光线一句)。用户素材角色的描述符从素材身份词派生。
2. **台账承载**:StoryboardMemory.cast/setting(影片级,持久化)。
3. **全链注入**:图计划上下文(t2i prompt 必须内嵌描述符)、条件决策
   上下文、enhancer 条件行(kind=cast/setting)、修复 hint 质量条款
   (orchestrator skill:重生成段落必须复述角色描述符)。
4. **skill 铁律**(window_generation 规则 7 / prompt_enhancer):出场角色
   的官方描述符必须【逐字】进每条 prompt —— 视频模型跨调用零记忆,这是
   无锚路线(t2v)和再入场时唯一的身份载体。
5. **评审收口**:clip.conditioning 带 cast/setting → review_shot 注入
   "CANONICAL CAST(全片外观契约)",出场角色逐一出一条外观匹配 check,
   场景延续时加一条陈设 check —— 漂移变成可修缺陷。

## 二、按 skill 的修复清单

| Skill | 主要问题(审计)| 修复 |
|---|---|---|
| **orchestrator** | 6 HIGH:仍教 8 个已退役工具(目录+决策规则+4 个例子);缺 vlm_route_suggestion;regenerate/segment 语义全旧 | **整篇重写**:三分类契约(accept/segment/full)、route 建议采纳纪律、免级联段修复语义、head/tail 特例、帧范围法则保留、hint 质量条款+cast 复述、3 个新例子 |
| **scene_write** | 自家例子违反完整动作法则(mid-air end_state!);实现状态描述陈旧;end_state 下游角色未记;frontmatter 作者错;警告口径过强 | 例子重写(摇晃交棒/一镜内完成坠落);cast/setting 输出契约;end_state 机器用途注明;frontmatter/口径修正 |
| **window_generation** | 规则 4 引用退役 tiv2v_window;例 2 输出退役 multi_image_fusion;两个"最强续接"打架 | 规则 4 重写(extend_prev 优先);例 2 换 ti2v_prev_plus_keyframe(带首帧强锁);ti2v_prev_last 改称"最强单帧锚";新增规则 7 CAST 法则 |
| **prompt_enhancer** | HIGH:仍教退役的 "Continuing from @Video1" 续接式;缺首帧强锁话术 | 改为 FIRST-FRAME PIN 条款("opens EXACTLY on @Image1",实测锁帧);@VideoN 多条+闸门自动补句说明;cast/setting 行使用法;CONTINUATION_SOURCE 列入不可引用 |
| **image_plan** | 角色→模型表仍路由 kling;例 2 理由提 kling | 表格改 seedance t2v @refs;first_frame 角色的三个真实消费者注明;源视频 @VideoN 说明;**t2i prompt 必须内嵌 cast/setting** |
| **video_retrieval** | 三处宣传退役的 retrieve_replace;"中间帧"与代码不符;检索机制措辞错 | 退役注记;**代码修正为真中间帧**(取时长中点,连视频打标同修);机制改为哈希词袋余弦的如实描述;消费者清单更新(媒体目录/@VideoN);"检索返回实拿语义"法则 |
| **semantic_critic** | 缺 junction 块;维度框架与指令不符;路由描述错 | junction+cast/setting 上下文入 Evidence;五维框架+二元路由(physics vs 其余)如实化 |
| **physics_critic** | 抽帧框架陈旧;失效模式漏 penetration/unexplained;"别驳测量链"条款不可执行 | 原生视频主路径框架;完整模式表(+强制映射说明);输出契约补 suggested_intervention;规则改为"诚实报severity,合并下游做" |
| **review_summarizer** | fix_classes 指向退役工具(代码同罪) | **代码 _FIX_CLASSES 重写**(segment_regen/full_regen)+ skill 类表同步 + route 建议主导说明 |
| **verifier** | verdict 回灌路径与 issues 填充条件描述过甚 | 两处如实化(history 只见 outcome+issues;issues 仅 score<0 时有) |

## 三、顺带修的代码问题(审计牵出)

- `_CONDITION_PRIORITY` 兜底顺序与 skill 偏好矛盾 → extend_prev 提到
  flf2v_bridge 之前(bridge 需要"意图到达图",确定性层无从判断)。
- `video_extract` 与视频打标实取【末帧】(idx=10**6),注释谎称中间帧 →
  两处都改为真·中间帧(探测时长中点)。
- review_summarizer `_FIX_CLASSES` 指向退役工具 → 重写。

## 四、诚实边界(修不了/不修的)

- skill 只能教:cast 描述符质量仍取决于剧本 brain;评审 check 是兜底网。
- 无评审器测"跨镜"漂移的量化程度(audit MEDIUM):目前逐镜对照官方描述符
  判 yes/no;逐对镜头的外观相似度测量(embedding 距离)登记为升级路径。
- deformation 模式在类型系统存在但 Gemini 词表未提供(如实注明)。

全套 446 测试通过。逐条审计原文:`~/.claude/jobs/…/audit_findings.json`
(48 条,含全部 LOW)。
