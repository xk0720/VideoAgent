# EpisodeMemory · 技能轨迹版讲解(2026-08-13 定稿)

> 配套示例:`docs/EPISODE_EXAMPLE.json`(晨光面包店
> `outputs/movie_20260811_022309` 蒸馏,8 镜 54s 成片)。
> 代码:`src/maestro/memory/episode_memory.py`;
> 回填工具:`scripts/distill_episode.py --run <目录> --prompt <片名>`。

## 0. 一句话定位

一部片收工 → 台账蒸馏成一条**技能轨迹**:轨迹头(全片共享上下文)
+ 步序列(一镜一步,每步 = 上下文 → 动作 → 评价)。检索时当范文、
训练时当语料、复盘时当卷宗 —— 一份格式三个用途。

## 1. 蒸馏取舍(存的时候就加工,用户裁决)

**留**:决策与结局 —— 每镜真正看到的上下文、选的策略、**完整 prompt
(草稿+终稿,不截断)**、引用图例、VLM 评语与分数。
**丢**:过程与文件 —— 路径、种子、重试流水、评审逐条、修复回合。
**新增**(台账里没有的):检索关键词、成败判词、**引用解读表**。

## 2. 轨迹头 header(全片共享上下文)

```json
"task": "晨光面包店",
"cast": {"面包师": "static: young woman, …beige apron…; dynamic: …"},
"setting": "老街转角的面包店内设木质柜台…",
"scene_layout": {"n_scenes": 1, "bgs": {"bg_1": [0], "bg_2": [1,…,7]}},
"reference_registry": { … 见 §3 … },
"asset_recipe": {"spaces": {"bg_2": {"t2i": 1, "derived": 2, "frame": 7}}}
```

- `cast`:人物外观描述原文(= 肖像引用的语义,写同类角色的范本);
- `scene_layout`:空间怎么切、每个空间管哪几镜(排产结构);
- `asset_recipe`:参考库怎么建的 —— 只记方式与数量(bg_2:一张
  文生图主板 + 两张环视派生 + **七张实拍回流**),不记文件。

## 3. 引用解读表 reference_registry(轨迹内一切引用在此闭环)

用户令:轨迹里出现的每个引用必须能查到"它是什么、图注是什么",
不需要回台账。示例(节选):

```json
"portrait:小女孩":        {"kind": "portrait", "desc": "static: petite schoolgirl…"},
"bg_plate:bg_2":          {"kind": "bg_plate", "src": "t2i"},
"space_view:bg_2/new_0":  {"kind": "space_view", "src": "frame",
   "caption": "A long wooden counter runs across the foreground…(全文图注)"},
"derived_junction_frame": {"kind": "derived_frame",
   "desc": "过渡视频切点后的首帧(由上镜末帧+肖像+空间视图派生);机器承接句把它钉为本镜开场画面"}
```

步里凡是引用 —— action.refs 的值、junction.space_view —— 都是
这张表的键。所以 "`new_0` 是什么" 的答案就在记录内:
`space_view:bg_2/new_0` = 实拍回流帧,图注全文照录。

## 4. 步 step(一镜一条,三格)

```json
{"step": 5, "label": "scene 1 shot 5",
 "context": {
   "shot": "镜头5:老板娘把纸袋递给小女孩…(分镜描述原文)",
   "camera_facing": "从入口一侧反向朝柜台、木架和烤炉,中近景",
   "bg_id": "bg_2",
   "prev_end_state": "小女孩趴在柜台边沿等待,老板娘在木架右侧",
   "junction": {"kind": "derive", "fallback_to": null,
                "space_view": "space_view:bg_2/new_0",
                "stitcher_via": "agent"}},
 "action": {
   "strategy": "ref2v", "decided_strategy": "ref2v",
   "degraded_from": null, "image_plan": "none",
   "prompt": "<终稿全文,一字不截>",
   "prompt_draft": "<草稿全文>",
   "refs": {"image_1": "portrait:小女孩",
            "image_2": "portrait:面包师",
            "image_3": "derived_junction_frame"}},
 "feedback": {"vlm_headline": null, "score": null, "converged": null}}
```

- **context**:字段取舍标准唯一 —— 当时真喂给 brain 的才有资格留
  (本镜描述/朝向/交界判定/上镜末态);
- **action**:策略 +(草稿/终稿)prompt 全文 + 图例;`decided` 与
  实际执行分开记,降级在 `degraded_from` 留痕;
- **feedback**:VLM 评语头条 + 加权分 + 是否收敛;**未评审的片
  诚实全 null**(示例即此),绝不用 0 分占位。

## 5. outcome 三态(客观信号推导,无 LLM 自评)

good = 全镜 Verifier 收敛;bad = 开了评审有镜没过;ungraded =
全程无实证评审(--no-review 的占位评审行不算数)—— 完成的片是
"最佳可得蓝图",不是失败案例。

## 6. 检索与利用

- **检索**:关键词 Jaccard(中文按字二元组切分,跨表述可召回);
  确定性算法,分数可复现;
- **guidance**(开工简报,从轨迹现场推导):good/ungraded 片的步
  → `replay_hints`(策略先验);实证失败的步 → `avoid`(策略+VLM
  评语);各片 (镜数,结局) → `past_task_shapes`(供拆镜决策);
- **注入纪律**:参谋不当司令 —— 检索结果附"以本次现场条件为准"
  注记,拍板权归当次实况(2026-07-31 事故裁决);
- **与 RL 同构**:步的 (context, action, feedback) 与 GRPO 训练样本
  同形状 —— 高分步可直接进训练语料,记忆库与训练集同源。

## 7. 三级记忆边界

| | 存什么 | 活多久 | 进 agent 的方式 |
|---|---|---|---|
| 技能库(程序性) | 跨片通用规矩 | 永久 | 对应角色每次调用全文注入 |
| 分镜台账(工作) | 本片每镜全量事实 | 一部片 | 每次决策压缩注入(to_brain_line) |
| **技能轨迹(本文)** | 跨片的 (上下文→动作→评价) 案卷 | 永久 | 按相似度检索,命中才注入 |
