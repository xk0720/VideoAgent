# 标准需求文档 — 窗口式生成 + 双层记忆(供用户分析裁决)

> 你的原话 → 标准化条款 → 我替你定的默认(D)→ 需要你裁决的问题(Q)。
> 你批完这份文档之前,代码不再动。

---

## 需求 1(你的 3.(1)):任务台账 —— brain 的短期工作记忆

**你的原话**:brain 维护一个 dict/list,按时间顺序放本任务依赖的
keyframe/video-clip 及描述,例如 `'scene 1 shot 1': 描述; keyframe 或
video + 路径(如果已生成); trajectory 轨迹`;可以不断更新。

**标准化**:一个按时间顺序的条目列表,每条对应一个 shot,字段如下:

| 字段 | 含义 |
|---|---|
| label | "scene 1 shot 2"(场号从剧本文本解析,解析不到全归 scene 1) |
| description | 这一镜的文字描述(剧本产物) |
| keyframe_path + keyframe_source | 关键帧路径 + 来源(t2i / 用户图片 / 用户视频抽帧) |
| video_path | 视频路径(生成后才有) |
| status | pending → keyframed → generated → verified 或 generated_with_defects |
| condition | 这一镜用什么条件生成的(策略名 + 实际输入) |
| reviews | 评审轨迹:追加式列表,每次评审一条(分数/失败项/物理判定/简报头条) |
| repair_actions | 修复循环的逐回合动作(工具、参数、接受/拒绝) |
| physics_trajectory | 物理测量的逐实体轨迹;没测量就是空,不造假 |

**更新规则**:每一步操作立即原子写盘 run 目录 `storyboard.json`;评审记录
只追加不覆盖(这就是你要的"轨迹")。

**读取接口**:① 下一个待生成的 shot;② 某镜之前最近的已生成镜头(窗口的
锚);③ 给 brain 的紧凑视图(一行一镜)。

**我替你定的点**:
- **D1**:"上一镜"取【最近已生成】而不是【已 verified】。理由:上一镜哪怕
  带遗留缺陷,它的尾帧才是时间上正确的续接点;跳过它用更早的完美镜头续接,
  画面必然跳变。
- **D2**:你写的 "trajectory 轨迹" 我按两个意思都存:评审轨迹
  (reviews + repair_actions)和物理轨迹(physics_trajectory),分开两个字段。

---

## 需求 2(你的 3.(2)):长期记忆 —— good/bad 案例,可执行化

**你的原话**:维护 long-term memory,蒸馏一次完整视频生成的轨迹,分 good
case 和 bad case;"记忆可执行化"不知道怎么实现。

**标准化 —— 可执行化的具体实现**(回答你的疑问):
一条 episode(一次完整任务的蒸馏)不存文字回忆,存两张能直接驱动决策的表:

- **replay 表(good 面)**:每个通过验收的 shot 的 {keyframe 策略, 条件
  策略}。下次相似任务检索命中 → brain **直接照抄**这套策略(决策记
  via="episode",不再消耗 LLM 推理)——检索即执行。
- **avoid 表(bad 面)**:没修好的 shot 的 {用过的策略, 失败原因(取自
  最后一条评审的头条)}。检索命中 → 注入 brain 提示当禁令——检索即禁止。

**good/bad 判定(完全客观,不问 LLM)**:全部 shot verified = good;任何
一镜未收敛 = bad。bad episode 里收敛了的镜头,其策略仍进 replay 表
(好的局部经验不陪葬)。

**检索**:prompt 关键词重叠打分(确定性、可复现,无向量库依赖);JSONL
持久化,跨任务累积。

**三层记忆分工**(和已有记忆的关系):
storyboard = 本次任务眼前有什么(任务内);repair skill(已有)= 这类缺陷
怎么修好(修复级,跨任务);episode(新)= 这类任务怎么排产(任务级,跨任务)。

**需要你裁决**:
- **Q-good**:good 的标准现在是"全部 verified"。要不要放宽(如 ≥80%)?
  我建议不放宽——标准干净客观。
- **Q-avoid**:avoid 表现在是软约束(注进提示,brain 决策规则写明"别选")。
  要不要升级成硬门(直接从菜单删掉该策略)?我建议软约束——历史失败不等于
  这次必失败,场景可能不同。

---

## 需求 3(你的 4):窗口式生成大循环

### 3a 剧本(你的 4.(1) 前半)
用户 prompt → 预生成全部 shot 的文字描述(playwriting,复用现有编剧+导演
agent)→ 用它初始化台账(全 pending)。

### 3b keyframe 阶段(你的 4.(1) 的 (1)(2)(3))
brain 给每个 shot 从【门控后的菜单】选一种:

| 策略 | 对应你的原话 | 出现条件 |
|---|---|---|
| t2i | (1) 按描述构建 prompt 文生图 | 后端有 t2i 能力 |
| asset_image | (2) 检索用户素材库的图片 | 素材库有图(identity/style 锚) |
| video_extract | (3) 检索用户视频素材、提取关键帧 | 素材库有视频 |
| none | (你没列,我补) | 上面三种都不可用时的诚实降级 |

### 3c 窗口条件策略(你的 4.(2.1))
生成"下一个未生成 shot"时,brain 从门控菜单选一种搭生成条件:

| 策略 | 对应你的原话 | 说明 |
|---|---|---|
| ti2v_prev_last | 你的 (1) 的一种读法 | 上镜【尾帧】当首帧 + 文本 |
| flf2v_bridge | 你的 (1) 的另一种读法 | 上镜尾帧当首帧、本镜 keyframe 当尾帧,首尾双锚 |
| tiv2v_window | 你的 (2) | 上镜【尾段视频】当运动参考 + keyframe 当首帧 + 文本(seedance-2.0 的 reference_videos 通道,已验证存在) |
| i2v_keyframe | (补)换场硬切用 | 本镜自己的 keyframe 当首帧,故意不接上镜 |
| t2v | (补)兜底 | 纯文本,啥锚都没有时 |

- **Q1(重要,你的 (1) 有歧义)**:你写"使用上一个视频的最后一帧【以及】
  这个 shot 已经生成的 keyframe 进行 text-image-to-video"——两张图一起用,
  普通 i2v 只吃一张。我读成两种策略都要:尾帧单独用(ti2v_prev_last)和
  尾帧+keyframe 双锚(flf2v_bridge)。你的本意是哪种?还是两种都保留?
- **Q2**:尾段窗口长度默认 2 秒(可调)。可以吗?

### 3d 每镜评审小循环(你的 4.(2.0))
- 每生成一镜必过评审;评审员先只用 VLM/Omni(语义+物理观点),物理测量
  critic 做可选开关(--with-physics-measure)。
- VLM 评审的维度和意见格式写在它的 skill 文件里(已有
  reviewer_skills/semantic_critic.md、physics_critic.md)——因为意见要
  嵌入台账轨迹。
- **定位(你点名的难点)**:物理判定自带 frame_range;语义失败目前是
  全镜级(VLM 没被要求给帧号)。
  **Q3**:要不要强制 VLM 评审输出 frame_range(提示里要求"指出问题在第几
  秒/哪一段"),让语义缺陷也能段定位?我建议要——"定位准确才能
  edit/regenerate"正是你说的关键。
- **Q4(角色命名)**:你说"verifier agent 进行工具调用"。现架构是三角分工:
  Summarizer 汇总评审 → brain 选修复工具 → Verifier 只裁决接受/拒绝。
  我强烈建议维持(职责分离防自我偏袒,有文献依据);如果你坚持让 verifier
  管工具调用,说一声我改。

### 3e 合成(你的 Final)
全部 shot 按时间顺序 ffmpeg 拼接 → movie.mp4。未收敛的镜头照拼(交付最优
可得),但台账状态如实。

### 3f 决策机制(贯穿 3b/3c,你说"window-based 可以作为一个 skill")
- brain 决策三层回退,每层记录在案:episode replay 命中直接采纳
  (via=episode)→ LLM 严格 JSON(via=llm)→ 确定性优先级兜底
  (via=fallback,菜单非空必有解,大循环永不卡死)。
- 窗口生成写成 brain 的技能文件 skills/brain_skills/window_generation.md
  (策略菜单、决策规则、episode 提示怎么用、输出格式、例子)。

---

## 现状声明(透明)

初版代码已按本文档实现并提交(commit 5865ecd;370 测试绿;原有代码未动,
唯一例外:内层循环加了一个默认不改变行为的可选参数 initial_candidates)。
刚完成的多智能体对抗审查确认了 2 个记账诚实性 bug,待你批完需求后一并修:
1. 策略执行崩溃降级到 t2v 时,台账没记 degraded_from,还把决策来源记成
   brain(via=llm/episode)——假记录会污染 episode 记忆;
2. 多候选(n_candidates>1)时台账只记最后一个 seed 的条件,最终胜出的可能
   是另一个 seed 的策略——张冠李戴进 replay 表。
另有一批审查项因会话额度耗尽未完成对抗验证,修复时我会人工复核。


---

## 追加裁决(2026-07-13):小循环轮数控制 + 物理评审开关

**裁决原文**:"可以,越早停越好。然后我们有可能不用纯 physics 来评审,
这个分支要做成开关控制的。"

**轮数控制(已实现)**:max_turns 只是成本天花板(超参保险丝),实际停止
自动决定,先到先停,`result.stop_reason` 留痕:
| stop_reason | 含义 |
|---|---|
| converged | 评审全过(唯一 converged=True 的情形) |
| quality_bar_met | 总分 ≥ compose.quality_bar 且无 severity ≥ 0.7 的缺陷;converged 如实为 False |
| no_improvement | 连续 compose.patience(默认 2)轮被 Verifier 拒 → 止损 |
| brain_accept | brain 末轮主动 accept |
| turns_exhausted | 天花板兜底 |
Verifier 保持纯相对闸门("比上一版好吗"),停不停是循环的事——
"接受 ≠ 完事"这一语义不变。stop_reason 同时写进分镜台账的评审记录。

**物理评审开关(已实现)**:configs `review:` 段——
`physics_vlm`(VLM 物理观点)与 `physics_measure`(非 AI 测量链)各自
独立开关,默认全开(行为不变);双关 = 纯语义/一致性/节奏评审。关掉的
评审员不出判定,其指标读作"未发现违规"(未评审 ≠ 扣分),权重可按需重调。


---

## 追加需求(2026-07-13,已裁决并实施):Image Plan + 素材检索

> 裁决:Q-A = 语义字段方案;Q-B/Q-D 同意;Q-C = 不写死场景规则,把
> brain 的技能(brain_skills/image_plan/SKILL.md)写完整、教它对任意素材
> 场景推理。全部已实现(见 git log)。

### 要点 1:关键帧升级为 Image Plan(数量 + 角色 + 来源,brain 决策)

现状问题:keyframe 阶段固定"每镜一张图",角色(当首帧还是当参考)被后面的
条件策略隐式决定 —— brain 没有"要几张图、每张干什么用"的显式决策。

标准化:生成视频前,brain 先出【Image Plan】,三件事一起定:

1. **数量**:0 / 1 / 2(暂定最多 2)。
2. **角色**(提前设定,锁死角色→模型族映射):
   | 数量 | 角色 | 视频生成必须调用 | payload 图片字段 |
   |---|---|---|---|
   | 1 | first_frame(首帧锚) | seedance-2.0 i2v(ti2v) | `image` |
   | 1 | reference(人物/物体/场景参考) | seedance-2.0 t2v+refs 或 kling-video-o1 | `reference_images` / `images` |
   | 2 | first_last(首尾帧) | seedance-2.0 i2v / wan-flf2v / veo3.1-lite / vidu q3 | `image`+`last_image` |
   | 2 | reference_pair(双参考) | kling-video-o1(images ≤7;可再带 `video`) | `images` |
3. **来源**:每张图独立选 t2i / 用户素材 / 素材视频抽帧(允许混搭,
   例:ref1=用户人物图,ref2=t2i 场景图)。

**最关键的要求**:brain 必须【按图片的角色】写视频 prompt,并输出能精确
解码成官方 payload 的 JSON。以 kling-video-o1 为例(用户给的目标形态):
prompt 要写成 "Use reference image 1 as the female character and reference
image 2 as the male character…" 这样的角色化描述;seedance 则用 @Image1
语法。执行器解码出的 payload 必须与官方 schema 逐字段一致
(aspect_ratio/duration/images/keep_original_sound/prompt/video)。

**分工提案(Q-A,请裁决)**:brain 只输出【语义字段】(策略、每张图的角色
与描述、按角色写好的 prompt、要不要带上镜尾段 video);机械字段
(aspect_ratio、duration、keep_original_sound、图片上传成 URL)由执行器
确定性补齐 —— LLM 不碰机械字段,格式永远不会错。
(替代方案:brain 直接输出完整 payload —— 更接近你的例子,但 LLM 填错
机械字段的风险高,需要逐字段校验器兜底。)

### 要点 2:素材检索逻辑(现状交底 + 背景图场景方案)

**现状(诚实交底)**:
- 源视频片段:`retrieve_source_shots(query)` = caption/标签的关键词重叠
  打分,确定性,无向量检索;
- 图片素材:目前【没有真正的检索】—— asset_image 策略是"按顺序拿第一张
  路径存在的 identity/style 锚",单图没问题,多图会拿错;
- 素材入库没有自动打描述:用户不给 description,检索无从匹配。

**背景图场景方案(用户 prompt + 一张关键背景图),四步(Q-C 请裁决)**:
1. **入库打标**:上传时给素材定 kind(background/character/object/style)
   + 一句描述。来源优先级:用户给的 description > VLM 自动 caption(有
   key 时)> 文件名(诚实降级,并警告检索质量受限)。(Q-D)
2. **剧本感知素材**:screenwriter/director 的输入带素材清单摘要("用户
   提供:客厅夜景背景图 ×1"),分镜描述必须围绕素材可用的场景写 ——
   否则写出海边剧本,背景图白给。
3. **镜头级分配**:同一张背景图对该场景【所有】shot 都可用:
   - 场景首镜(establishing shot):背景图直接当 first_frame;
   - 后续镜头:当 reference image(场景一致性),或经 seedream-v4 图像
     编辑"把人物/物体摆进背景"生成该镜专属 keyframe(编辑底图用法);
4. **逐镜检索**:shot description 关键词 vs 素材描述重叠打分选图
   (多素材时不再"拿第一张");CLIP 向量检索列为升级项登记缺口台账,
   本轮不做。

### 要点 3:技能文件同步(格式必须与 API 一致)

- 新增 brain 技能 `image_plan`(或并入 window_generation):数量/角色/来源
  的决策规则、角色→模型族映射表、**每个模型的 payload 模板**(与
  TOOL_LIBRARY 和多图调研报告逐字段一致)、两种引用语法的 prompt 写法
  规范(kling "reference image N" / seedance "@ImageN")、以用户的
  kling-video-o1 例子作为 worked example。
- 窗口条件策略与 Image Plan 对齐:角色决定条件菜单(role=first_last →
  只开 flf2v 族;role=reference_pair → 只开 kling-o1/seedance-refs 路线),
  避免"图是按首尾帧生成的、却被当参考图用"的错配。

### 待裁决问题汇总

- **Q-A**:brain 输出语义字段+执行器补机械字段(我推荐),还是 brain 直接
  输出完整 payload?
- **Q-B**:双图允许混来源(一张用户素材 + 一张 t2i)?我建议允许。
- **Q-C**:背景图的默认用法按"首镜当首帧、后续当参考/编辑底图"来?
- **Q-D**:素材入库描述的优先级(用户描述 > VLM caption > 文件名)同意?
  VLM caption 需要消耗 QWEN key。
