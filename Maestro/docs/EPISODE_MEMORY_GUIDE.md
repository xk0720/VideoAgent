# EpisodeMemory 剧本级情景记忆 · 讲解文档

> 配套示例:`docs/EPISODE_EXAMPLE.json`(从成功例 晨光面包店
> `outputs/movie_20260811_022309` 离线蒸馏,8 镜 54s 成片)。
> 代码:`src/maestro/memory/episode_memory.py`;
> 回填工具:`scripts/distill_episode.py`。

---

## 0. 一句话定位

情景记忆 = **"这类片子当年怎么排产的"的完整案卷**。一部片收工时蒸馏
一条记录;下部相似片开工时按相似度检索,以"参谋建议"身份注入决策
上下文 —— 参谋提供先验,现场条件永远拥有否决权。

## 1. 记录结构总览(七大块)

```
EpisodeRecord
├── 检索头     episode_id / user_prompt / keywords / outcome
├── replay 表  可执行摘要:每镜验收过的 (image_plan, 条件策略)
├── avoid 表   失败模式:什么策略在什么镜砸了、评审怎么说
├── repair_tool_stats  修复工具接受/拒绝台账
├── screenplay_digest  剧本形状(2026-08-13 新增)
├── asset_build        参考库构建档案(新增)
└── shot_plans         逐镜蓝图(新增)
```

前四块是旧版就有的**镜级可执行层**;后三块是本次新增的**剧本级
档案层**。两层分工:replay/avoid 给 brain 直接抄作业,档案层给
brain(和人)看"整部片的打法"。

## 2. 逐块讲解(对照示例文件)

### 2.1 检索头

```json
"user_prompt": "晨光面包店",
"keywords": ["晨光", "面包", "包店", "小女", "女孩", …],
"outcome": "ungraded"
```

- `keywords`:检索键。英文按词、**中文按字二元组**切分(2026-08-13
  修:原正则把整段汉字当一个词,检索退化成全句精确匹配 ——
  "晨光面包店"检索不到"小女孩清晨到面包店买面包的故事")。取材
  不止片名 —— 全部分镜描述一起进键,所以"雨夜""柜台""收银台"
  这类场景词都能召回。
- `outcome` 三态(全部由客观信号推导,无 LLM 自评):
  - **good**:全镜通过 Verifier 收敛;
  - **bad**:开了评审但有镜没修好;
  - **ungraded**(新增):`--no-review` 跑法,占位评审行不算实证
    —— 完成的片子是"最佳可得蓝图"而非失败案例(修复前它被误判
    bad、8 镜全进 avoid 当负面教材)。

### 2.2 replay 表(可执行摘要)

```json
{"label": "scene 1 shot 1",
 "image_plan": "single_reference",
 "condition_strategy": "ref2v", "decided_strategy": "ref2v",
 "degraded_from": null, "converged": null, "final_score": null}
```

每行 = 一镜的**图计划 + 条件策略**及其下场。`decided_strategy` 与
`condition_strategy` 分开记:分得清"brain 本来选的"和"实际执行的"
(降级会在 `degraded_from` 留痕)。ungraded 片的 `converged: null`、
`final_score: null` 是诚实标注 —— 没评审就没有分,0.0 占位分绝不
入账。

### 2.3 avoid 表与 repair_tool_stats

失败面:哪镜什么策略没修好、最后一条评审的头条意见;修复工具的
接受/拒绝计数。**只收实证**:必须真的评审过才有资格进 avoid。
(示例文件里两者为空 —— 该片未开评审。)

### 2.4 screenplay_digest(剧本形状)

```json
"cast": {"面包师": "static: young woman, …beige apron…;
          dynamic: gentle smile, arranging bread…"},
"setting": "老街转角的面包店内设木质柜台、玻璃门窗…",
"n_scenes": 1,
"bgs": {"bg_1": [0], "bg_2": [1,2,3,4,5,6,7]},
"music_scenes": [1]
```

留的是**排产结构**:人物正典描述符全文(未来同类角色的措辞范本)、
空间怎么切(街景一镜、店内七镜 —— 视区法分配的实绩)、场数与
配乐位。playwriting 的 brain 决定"这类题材拆几镜"时,读的就是
这里(经 guidance 的 past_task_shapes 通道)。

### 2.5 asset_build(参考库构建档案)

```json
"spaces": {"bg_2": {"views": [
    {"view": "master",   "src": "t2i"},      ← 文生图主板
    {"view": "right_90", "src": "derived"},  ← 环视视频派生
    {"view": "left_90",  "src": "derived"},
    {"view": "new_0",    "src": "frame"}, …],← 实拍清场帧回流
  "n_frame_views": 7}}
```

回答"这部片的参考库是怎么建起来的":每个空间的视图清单**连来源
通道一起记**。三种 src 就是空间圣经的三条供给线:t2i 主板一次定稿、
环视视频派生多视角、实拍帧持续回流。`n_frame_views: 7` 一眼看出
这部片后期挑图的主力已经换成实拍回流(实拍 > 生成的设计意图在
数据上兑现)。

### 2.6 shot_plans(逐镜蓝图 —— 信息量最大的一块)

```json
{"label": "scene 1 shot 5",
 "bg_id": "bg_2",
 "camera_facing": "从入口一侧反向朝柜台、木架和烤炉,中近景",
 "image_plan": "none",
 "junction": {"kind": "derive", "fallback_to": null,
              "space_view": "new_0", "stitcher_via": "agent"},
 "strategy": "ref2v", "decided_strategy": "ref2v",
 "degraded_from": null,
 "n_references": 3,
 "prompt_draft": "反打中近景,摄影机静止:首先,<<<image_2>>>…",
 "prompt_final": "…<<<image_2>>>说:"今天开业酬宾,买一送一。"…",
 "score": null}
```

一镜一条,完整回答"这镜当时怎么拍的":
- **junction**:交界走了哪条路(派生/硬切/顺延)、挑中空间库哪张
  视图当锚、两镜描述是缝合师 agent 组的稿还是模板兜底;
- **camera_facing**:挑视图用的朝向证据原文;
- **prompt 草稿 vs 终稿**(各截 400 字):brain 原始输出与过完
  全部闸门(记号化/台词/音频法)后的出门稿 —— 对比可见管线各
  闸门的实际作用;
- **n_references**:挂了几张参考图(肖像+空间视图+派生帧的总数)。

replay 表是它的"可执行摘要";shot_plans 是"完整现场记录"。

## 3. 构建时机与两条入库通道

1. **在线蒸馏**:`generate_movie_windowed` 收工时(§M)自动调用
   `distill_episode(user_prompt, storyboard)` —— 台账里的一切
   (entries/junction_meta/spaces/condition)就地压缩成一条记录,
   JSONL 追加落库(原子重写);
2. **离线回填**(新增):
   ```bash
   python scripts/distill_episode.py \
       --run outputs/movie_20260811_022309 --prompt "晨光面包店" \
       --memory <库路径>            # 不给 --memory 则只出示例不入库
   ```
   历史归档 run(尤其 --no-review 时代被跳过蒸馏的成功例)都能补记。

## 4. 检索与利用(信息如何回到 agent 手里)

开工时管线调用 `guidance_for(user_prompt)`:

1. **检索**:查询关键词与库内每条记录的 keywords 做 Jaccard 相似度,
   top-3 命中(0 分不硬凑)。确定性算法,无 embedding 依赖,分数
   可复现可解释;
2. **打包**:good/ungraded 的 replay 行 → `replay_hints`;bad 的
   教训 → `avoid`;各片的 (镜数, 结局) → `past_task_shapes`;
3. **注入**:整包作为 `episode_guidance` 进入分镜与条件决策的
   prompt,并附机器注记 ——
   > "verified on a similar PAST task — weigh it as advice;
   > **current-run conditions win**"

   **参谋不当司令**(2026-07-31 裁决):早期"检索即执行"直接照抄
   历史策略,曾因历史条件与本次错配翻车;现在检索结果只缩小搜索
   空间、提示已知的坑,拍板权归本次现场条件。

## 5. 与另外两级记忆的边界

| | 存什么 | 活多久 | 进 agent 的方式 |
|---|---|---|---|
| 技能库(程序性) | 跨片通用规矩 | 永久 | 对应角色每次调用全文注入 |
| 分镜台账(工作) | 本片每镜事实 | 一部片 | 每次决策全量压缩注入 |
| **情景记忆(本文)** | **跨片成败案卷** | 永久 | **按相似度检索,命中才注入** |
