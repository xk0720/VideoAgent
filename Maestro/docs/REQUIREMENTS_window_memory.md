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


---

## 追加需求(2026-07-14,已裁决并实现):原生视频评审 + NEWTON 式盲测 Verifier

### R-A:GeminiVLM 改为原生视频输入(废除抽帧)

- A1 传输:评审调用把【整段视频】以 inline_data(video/mp4, base64)喂给
  Gemini(NEWTON _video_part 形态:每个媒体 = 一段文本标签 + 一段
  inline_data);删除 GeminiVLM 的抽帧路径。加 NEWTON 同款重试
  (3 次,指数退避)。
- A2 评审上下文 = 生成条件:prompt 部件按顺序装载 ——
  [SHOT VIDEO 本体] + [CONDITION: first/last 关键帧图(如用)] +
  [CONDITION: 参考图(如用)] + [CONDITION: 上镜尾段参考视频(如用)] +
  文本指令(shot 场景描述 + 生成时的 video_prompt)。
  语义评审的定义升级为:视频是否贴合【文本 + 全部多模态条件】——
  条件贴合度成为显式评审维度(每个条件出一条 checklist 项)。
  实现路径:窗口循环把 clip.conditioning = {images(role), reference_video,
  video_prompt} 挂在候选上,评审后端读取(mock 不受影响)。
- A3 输出契约不变(checklist 项 PASS/FAIL + fix + 定位;物理 verdict
  mode/severity + 定位),下游 DefectReport/修复管线零改动。定位单位改为
  【秒】(原生视频下 Gemini 报时间戳比帧号可靠),后端按 fps 换算回帧。
- A4 compare()(锦标赛用)同步改为双视频原生输入。
- A5 OpenAICompatVLM(gpt-4o/qwen 备选)保留抽帧路径(chat/completions
  无视频通道),文档注明差异。
- A6 体积护栏:inline 上限 ~18MB(NEWTON MAX_INLINE_MB)。超限视频先用
  ffmpeg 转码压到限内(480p/低码率,仍是原生视频);ffmpeg 缺失才退回
  抽帧,并大声记日志。【U1 待批】

### R-B:Verifier 改为 NEWTON 式盲测 A/B(修改版 vs 原版)

- B1 机制照搬 NEWTON verify_relative:两段视频随机顺序、只标 Video 1/2、
  评委不知来源;输出带符号分 [-10,+10] + 各维度一句话 + 输家的 issues
  列表 + summary;按随机分配把分数映射回 candidate-vs-baseline。
- B2 评审维度(NEWTON 是 SA+PC 两维;按"越全面越好"扩到五维,【U5 待批】):
  1. Semantic adherence — 是否符合场景描述的事实(数量/颜色/材质/身份/
     场景/动作顺序);
  2. Physical correctness — 运动物理合理性(重力/惯性/碰撞/接触;无穿模/
     漂浮/瞬移/因果乱序);
  3. Condition adherence — 是否贴合提供的关键帧/参考图/参考视频(我们的
     增量维度,NEWTON 没有);
  4. Temporal consistency — 时序一致性(无闪烁/身份漂移/背景跳变);
  5. Visual quality — 画面质量(伪影/肢体畸变/纹理崩坏)。
- B3 接入点(【U2 待批,推荐方案一】):
  方案一(推荐):盲测 A/B 成为 Verifier 的【主闸门】——修复候选 vs 当前
  最优,remapped score ≥ +1 才接受(0 = 保守拒绝,保持单调精神);
  weighted_total 继续计算并记台账,但只作观测不再裁决。
  方案二:保持现状(指标主闸 + 边际盲测确认),仅把确认器升级为完整
  五维指令。
- B4 输家 issues 回灌:评委给出的具体问题列表进 brain 下一回合的上下文
  (与 review 简报合流),NEWTON 的 "objective results feed back" 同款。
- B5 停止条件不变:converged 仍由 checklist 全过判定;patience 统计的
  "被拒"改为 A/B 拒绝。【U3 确认】

### 用户裁决(2026-07-14)与落地记录

- **U1** 超限视频 → ffmpeg 转码压体积。实现:MAX_INLINE_MB=18,超限
  `scale=-2:360 -crf 33 -an` 转码为 `*_inline360.mp4`(仍是原生视频,
  永不退回抽帧;ffmpeg 缺失/失败 → 大声日志 + 本次拒评)。
- **U2** → 方案一:盲测 A/B 是 Verifier 的【主闸门】。verify_pair 输出
  至少含各维 notes + 总分 + conclusion(参考 NEWTON,外加创新——见下);
  weighted_total 只观测记账。verify_pair 返回 None(传输失败/mock 桩)
  才落回指标闸,且大声记 `blind_ab_unavailable`。
- **U3** 确认:converged 仍由 checklist 全过判定;patience 统计 A/B 拒绝。
- **U4** 确认:VLM 报秒,后端按探测到的 fps 换算回帧。
- **U5** Reviewer 五维保留但定位为【参考框架】;核心交付是"有价值的问题
  定位":issues[] 每条 = type(frame/segment/global)+ 秒级时间片 +
  category(semantic/condition/physics/temporal/visual)+ entity +
  severity + problem + reason +(suggestion)+ check_ref(挂回失败的
  check)。suggestion 描述"修好后内容长什么样",永不点名工具——工具
  选择归 brain。Verifier 用四维:semantic/physics/temporal/visual
  (条件贴合度归 reviewer 管——单片评审才有条件可对照)。
- **U6** → 直接合并:review_shot 一次上传服务语义+物理两个 critic
  (多模态大模型本来就同时看得到物理),按 (path, mtime) 缓存;
  assess_semantic / assess_physics 都切同一份包。

### Verifier 创新点(在 NEWTON verify_relative 机制之上)

1. 各维带符号分(-10..+10)独立输出,"修了 A 坏了 B"必须在对应维度上
   显式为负,不许平均掉;
2. 缺陷探针:评委逐视频回答"目标缺陷是否仍在" → target_fixed
   (本次修复到底修没修它声称要修的东西);
3. 维度不回退守卫:accept 当且仅当 总分 ≥ +1 且 min(维度分) ≥ -2
   (单调契约的维度化);
4. 败方 issues 回灌 brain 下一回合历史(NEWTON "objective results
   feed back" 同款,brain 据此不重复被拒动作)。

### 落地位置

- `models/mllm_backends.py`:`_SHOT_REVIEW_INSTRUCTION` /
  `_VERIFY_PAIR_INSTRUCTION` 模板;GeminiVLM.review_shot(合并缓存)、
  verify_pair(盲测随机槽位 + 回映射)、compare(双视频原生)、
  `_looks_like_video` 诚实闸(mock 文本桩静默、真视频缺 key 则
  RuntimeError)。
- `agents/verifier.py`:verify_pair 主闸 + 指标闸兜底;verdict 挂
  candidate.verifier_verdict。
- `pipeline/window_loop.py`:clip.conditioning = {video_prompt,
  images(role), reference_video} 挂到每个候选,评审调用据此装配
  条件对照 parts。
- `pipeline/generate_loop.py` / `agents/orchestrator.py`:repair_context
  传给闸门;verdict 的 issues/score 进 brain 历史。
- 测试:`tests/unit/test_native_video_review.py`(单次上传断言、
  checks/issues→checklist/verdict 映射、秒→帧、盲测回映射与接受规则、
  A/B 主闸与指标兜底)。

---

## 追加需求(2026-07-14 第二轮,已裁决并实现):模型映射确定化 + 调用日志 + 时长

- **任务 0(模型映射 + debug 可见性)**:"所有可能的情况与调用模型名称的
  对应关系肯定都是确定的 —— 先调研、写文档、再按文档改代码。"
  → 权威映射表 `docs/CONDITION_MODEL_MAP.md`(9 条件策略 + 12 修复工具 →
  模型 id + payload 字段 + 时长规则,全部官方 schema 核验);
  → 每次 WaveSpeed 调用(模型名 + 参数)打终端 INFO + 落
  `<out_dir>/wavespeed_calls.jsonl`(WaveSpeedClient._run_task 唯一出口);
  → 路由修正:tiv2v_window 一律走 seedance-2.0 **text-to-video**
  (尾段 = reference_videos/@Video1;本镜图 = reference_images/@Image1
  软锚,绝不再把图当 first_frame 切到 i2v 端点 —— i2v schema 没有
  reference_videos,未验证组合,后端 generate() 现在任何上传前直接拒绝)。
- **任务 1(duration)**:brain 逐镜自判时长,范围写死 **4-10s**(第一版
  裁决 3-8s,同日改 4-10s);**brain 没输出 → None = 生成调用不传
  duration 字段,用模型自然默认**(绝不 feed 预设值);规划值原样传进
  每个生成调用(窗口策略 / generator / extend / propagate)。seedance
  [4,15] ⊇ [4,10] 原样直传;kling {5,10} 向上 snap(call log 可核对)。
- **debug 追加**:brain 的原始输出(策略/工具决策)也要落日志(见
  window_loop/_decide 与 orchestrator 的 brain_log)。

---

## 追加需求(2026-07-15,已实现):基线锚点 + Prompt Enhancer(均超参开关)

- **需求 1(基线锚点)**:任务开始单独调用 brain 按用户指令一次直出视频
  作 anchor,收尾与成片对比,目的"不扰乱原方法、量化框架增益"。
  → `--baseline-anchor`(+ `--anchor-duration`);路线确定性:无素材 =
  t2v;仅图 = ti2v(首图当首帧);有视频(可带图)= seedance-2.0 t2v +
  reference_images/videos(视频裁到 ≤15s/条,≤3 条)。
  **同日追加裁决:锚点只生成、不做机器对比/verifier 裁决、不接
  enhancer —— 用户自己看片对比**;锚点任何失败只记日志。
- **需求 2(Prompt Enhancer)**:可选 agent,skill 按各模型官方 prompt
  guide 蒸馏(`skills/brain_skills/prompt_enhancer/SKILL.md`:通用结构
  主体→动作→场景→镜头→光;seedance @ImageN/@VideoN、kling
  "reference image N"、i2v/flf2v 只写运动不写静态、忌否定句/抽象词)。
  输入 = shot 描述 + 执行器收集的条件事实清单(_conditions_for_prompt,
  增强器只能利用不能发明)+ 策略→家族映射;输出 STRICT JSON,校验失败
  保留原 prompt。`--prompt-enhancer` 开启;原始输出进 brain_calls.jsonl。

---

## 追加需求(2026-07-16 第三轮,attempt2 复盘,已裁决并实现)

- **(1) 完整动作法则**:每镜描述完整动作单元,一次性动作(跳/坠/撞)禁止
  切半;切点交接只许可持续运动(小跑/行走/滚动);end_state 禁 mid-air。
  scene_write 另加 LINKING NARRATION(描述写明如何接棒/交棒)与
  caption 粘贴修正(只取身份词,照片姿态/场景词禁止照搬)。
- **(2) 真续接换代**:tiv2v_window(t2v reference_videos 参考通道,实证
  接不上画面)退役 → `extend_prev`(bytedance/seedance-2.0/video-extend,
  schema 复核:video 必填从末帧续画、last_image 可选目标尾帧、输出=
  输入+续段拼接须裁头);skill 写明"只写接下来 + 维持清单"。
- **(3) Brain I/O 报告**:scripts/render_brain_log.py(任何
  brain_calls.jsonl → markdown);attempt2 全量 I/O + 问题分析在
  docs/analysis_attempt2_brain_io.md。
- **(4) 修复重做**:keyframe_edit_propagate 禁用(中间帧编辑致前后失调);
  frame_to_frame 摘除(重复);regenerate_segment = interior flf2v 双锚
  免级联 / tail i2v / head 用条件首帧图作锚(无锚诚实降级)。
- **(追问)不修的机制**:converged(实证 shot3 零修复)/quality_bar 原有;
  新增 `--repair-severity`(最坏缺陷低于阈值不修,
  stop_reason=minor_defects_tolerated,默认 0 关闭,荐 0.6)。

---

## 追加需求(2026-07-17,已裁决并实现):生成三分类定稿 + 修复三分类

- **生成**(用户实测:t2v 的 @Image1 可基本固定首帧):① 尾段续接 →
  extend_prev(video-extend);② 尾帧 + 无素材 → ti2v_prev_last(i2v 硬
  锁);③ 尾帧 + 有素材(生成图/检索图/源视频)→ ti2v_prev_plus_keyframe
  (t2v:@Image1=尾帧 + prompt 强锁 "opens EXACTLY on @Image1";
  @Image2..=图;@VideoN=用户源视频,新接,≤3 条逐条裁 15s)。首镜/换场
  四路保留;multi_image_fusion 退役;flf2v_bridge 护栏(身份照片绝不当
  收场锚)。已调研未启用备选(kling elements 双锚路线 / seedream 组合
  首帧)登记缺口台账。
- **修复三分类**:不修(converged/quality_bar/repair_severity,后者默认
  不动)/ 局部(regenerate_segment,内段 flf2v 免级联)/ 全修
  (regenerate = **严格按原始条件方法重生成**,窗口层闭包)。菜单裁到
  3 + simulate_reference 门控;方案 B:确定性 vlm_route_suggestion
  (覆盖 ≥90% → 全修)注入 brain 上下文作强建议。

---

## 追加裁决(2026-07-17 第二轮,已实现):视频素材入库原生打标

视频入库不再依赖抽帧打标 —— shot 可能直接续用用户片段,标签必须描述
**整段内容**(身份词+场景+运动/运镜)。假设素材段不长,Gemini-flash 原生
看整段(`GeminiVLM.caption_video`,>18MB 自动转码,桩文件诚实沉默)。
降级链:原生视频 caption → 中间帧 caption(无视频通道的 VLM)→ 文件名
(末端,响亮)。**抽帧能力保留**:图计划的 `video_extract` 仍抽中间帧当
key image(需要某个物体/精确帧时)。skill 三处同步(video_retrieval 打标
链、scene_write "可写直接续用素材的镜头"、image_plan video_extract 定位)。

---

## 追加裁决(2026-07-17 第三轮,已实现):ViMax 借鉴七项(用户:"都做吧")

来源:精读本地 ViMax 仓库 shot planning(storyboard_artist / character_
extractor / script_enhancer 等)。P1 五项 + P2 两项全部采纳:

- **① <角括号> 出场标记(代码+skill)**:剧本每次提到 cast 角色都写
  `<name>`(名字照抄 cast 键)。机器解析(`_MARKER_RE`/`_cast_in_shot`)
  确定本镜出场角色 → cast 注入与评审 check 只对出场者
  (clip.conditioning["cast"] = 子集)。诚实降级:无标记/标记全不匹配 →
  全量注入。**出口一律剥标记**(`_strip_markers`):spec.prompt、brain
  的 video_prompt、enhanced prompt、t2i prompt/检索词(_make_keyframe 与
  _execute_image_plan 收口)—— 生成模型永远看不到角括号;台账 description
  保留标记供解析。
- **② cast 静/动拆分(skill)**:描述符定型 "static: …; dynamic: …" ——
  static 半句 = 逐字不变的身份契约,dynamic 半句 = 允许变的部分(表情/
  姿态/持物);易混角色必须有响亮区分特征;完整度按"角色设定图"标准。
- **③ 画面地理(skill)**:出场者写画内方位与朝向("left of frame,
  facing right");特写点名入画身体部位;**不可见者不入描述**(画外音/
  暗示存在都禁止)。scene_write 规则 8 + enhancer 规则 8。
- **④ 机位复用(skill)**:同场景优先回到已确立机位("same framing as
  shot 1"),只在叙事需要时换角度。scene_write 规则 9。
- **⑤ 重复即精确(skill)**:关键视觉事实(角色身份词、关键物外观、
  承重空间关系)在最要紧处再说一遍 —— enhancer 规则 7,由角色扩展到
  物体与空间关系。
- **⑥ variation 变化幅度(P2,代码+skill)**:剧本逐镜声明首尾帧变化
  large|medium|small(词表校验,非法 → 空);入台账 ShotEntry.variation
  → to_brain_line → 条件决策上下文。策略提示:小变偏续接单锚
  (extend/i2v),大变偏双锚/自由(flf/t2v);是提示不是闸门。
- **⑦ opening_frame 开场静态快照(P2,代码+skill)**:仅首镜/换场镜输出
  纯静态开场快照(无进行中动作)。入台账 → image_plan 上下文;t2i 首帧
  prompt 以它为底稿(静态句适合单帧,动作句会出运动模糊/半程姿态);
  brain 不给逐张 spec 时确定性兜底同样优先用它。续接镜必须留空
  (开场 = 上镜 end_state,重复声明招矛盾)。

测试:tests/unit/test_vimax_borrowings.py(10 条:标记剥除/出场过滤/
诚实降级/词表校验/兜底底稿/出口收口)。全套 458 通过。

---

## 追加裁决(2026-07-18,已实现):attempt3 循环病因 —— 锚定路线 prompt 瘦身

现场证据:shot3/4(ti2v_prev_plus_keyframe,@Image1 软锁)输出循环、不接
上镜尾帧;用户手删 prompt 至「PIN + 身份 + 一句动作 + preserve 句」后同镜
成功 —— 软锁经不起噪声稀释,重述的每个事实都在和锚竞争。三类噪声全部
定位:setting 整句重述(被 t2v 当建景指令)、cast 契约连 "static:/dynamic:"
标签逐字入 prompt、全修 " Fix: hint" 追加(动作×2、身份×3,越修越长)。
shot1/2 同样带噪却成功 = i2v 通道级硬锁不吃 prompt 噪声,反证软锁病因。

- **P0-A 瘦身法则**:锚定路线(i2v_keyframe / flf2v_own_pair / flf2v_bridge
  / ti2v_prev_last / ti2v_prev_plus_keyframe / extend_prev,集合
  `_ANCHORED_STRATEGIES`)prompt 固定四段式 ≤70 词:PIN + 一句身份 +
  一句新动作 + 一句 preserve;禁建景句、禁开场布局重述。setting/cast 条件
  行按锚定/无锚带确定性 `note`(_conditions_for_prompt),无锚路线维持
  "逐字复述唯一载体"。redundancy/画面地理两条 ViMax 规则同步限定无锚。
- **P0-B 全修合成**:`_regen_prompt(strategy, base, hint, slots)` —— hint
  【替换】原动作,不追加;PIN 句仅 ti2v_prev_plus_keyframe 加(常量
  `_PIN_SENTENCE`,与兜底模板同源);漏提槽位引用闸门自动补;hint 引用
  未知编号 → 回退 base。台账记 `regen_prompt_mode`。
- **P0-C 标签清洗**:`_scrub_cast_labels` —— cast 契约值整串出现 → 换成
  static 半句;裸 "static:" 剥除;改写后的 "dynamic:" 残留无法安全定界 →
  响亮告警不动刀。出口收口:brain video_prompt / enhanced / 全修 hint /
  t2i query(_execute_image_plan 传 cast)。skill 同步:static 半句自然
  语序,标签永不入 prompt(enhancer/window_generation/image_plan)。
- **P1-D log 口径**:episode/fallback 决策 brain_log 补 context+menu ——
  attempt3 排查曾误读 "junction 是 null",实为重放路径不记 context。
- **P1-E 时长-密度法则**(scene_write):动作必须填满秒数;"继续 X" 型
  过渡不成镜,并入所导向的镜(shot3 整镜 = 再跑一个身长 → 6 秒循环)。
- **P2-F 漂移不延续**(enhancer):junction 实况只取位置/运动事实;外观
  与 cast 契约矛盾(蓝项圈)写契约不抄漂移,漂移留给评审当缺陷;禁止
  发明任何条件行里没有的视觉细节(blue-and-white bowl)。

测试:tests/unit/test_prompt_diet_attempt3.py(11 条)。全套 469 通过。

---

## 追加裁决(2026-07-18 二轮,已实现):警戒线取代硬帽 + hint 动作保底

用户两问定案:「≤70 是否太绝对」「hint 若没有 motion 描述呢」。

- **≤70 降级为警戒线**(分段预算按用户裁决暂不加):长度是代理指标,
  真病是内容类型。skill 措辞改为「四段式自然落在 55-95 词;过 ~100 先
  自查禁止内容(建景句/布局重述/重复身份),砍那个,永远不砍动作词」
  (enhancer DIET + window_generation 规则 6,PIN 条款同步去数字)。
- **建景句确定性拦截**(`_scrub_setting_sentence`):锚定路线出口,
  canonical setting 原句(大小写不敏感)整句出现 → 替换为
  `_PRESERVE_CLAUSE`(整句替换安全);只出现改写片段(≥70% 内容词命中
  但句子变了,无法安全定界)→ 响亮告警不动刀。无锚路线不拦(setting
  是唯一场景载体)。收口:brain video_prompt / enhanced / 全修合成。
- **hint 动作保底**(P0-B 二轮):hint 可能只写外观(纯身份缺陷)——
  PIN + 纯外观 = 无运动指令 → 静止/循环。不做启发式动作检测(不可靠),
  `_regen_prompt` 新参 action/end_state,【无条件】接剧本动作锚
  "This shot's scripted action: <剧本句,剥 'Shot N: scene N —' 前缀>,
  ending as: <end_state>." —— 起点(PIN)/过程(剧本句)/终点(end_state)
  三件套永远齐;动作是唯一重复有益的内容类别。
- **hint 契约升级**(orchestrator skill):regenerate 的 hint 就是新
  prompt 正文(替换非批注),必须自足 —— 完整动作 + 一句身份(static
  半句自然语序,标签禁入)+ preserve 句;动作锚只是保底,缺动作的 hint
  仍是不合格 hint。

测试:test_prompt_diet_attempt3.py 增至 21 条。全套 474 通过。

---

## 追加需求(2026-07-18,dev-rl 分支,已实现):S0 RL 数据管道

用户批准 RL 方案(docs/RL_TOOLCALLING_RESEARCH_2026_07_18.md)后的第一步:

- **decision_id 贯通**:brain_log 每条记录自发 16 位 uuid 并返回;
  _brain_pick/_decide(三条路径)/orchestrator.decide 把 id 写进决策
  dict。决策与延迟结局从"时序猜"变"id 连"。
- **结局记录**(与决策同文件 brain_calls.jsonl):`repair/outcome`
  (verifier accept/reject/stop,靠 decision_id 连修复决策;
  generate_loop 两处)+ `window/shot_outcome`(每镜 converged/
  stop_reason/repair_turns/条件决策 id;窗口主循环)。orchestrator
  日志补 tools_menu(完整菜单条目,重建 prompt 用)。
- **prompt 单源**:决策 prompt 构建抽为 `window_loop.decision_prompt`,
  生产与数据集构建器 import 同一函数 —— 训练分布=生产分布逐字符一致
  (Crayotter "一处 schema 三处消费" 原则)。
- **scripts/rl/build_dataset.py**:run 目录 → sft/kto/dpo_pairs/
  eval_holdout/excluded 五个 jsonl。标签规则 v1 保守("不怪它的失败
  不进它的坏样本";修复=verifier 判决、条件/润色=零修复收敛、结构层
  失败=坏、归因不清=排除且写明原因)。成对样本两路:enhancer 拒/过
  重试对(confidence 1.0)、修复拒-收相邻轮对(0.7 如实标注)。旧
  格式日志诚实排除不崩(attempt3 实测:1 坏样本 + 19 条带原因排除)。
- **scripts/rl/eval_replay.py**:holdout 重放 + 生产同款校验器打分
  (parse/in_menu/refs_ok/agree/pass^k),对任意 OpenAI 兼容端点,
  零生成费;三基线纪律(原始底座/仅 SFT/仅格式奖励)写入 README。
- 训练脚本(TRL SFT/KTO/DPO)属 S1,不进本仓库依赖。

测试:tests/unit/test_rl_dataset.py(6 条,全离线)。全套 480 通过。
## 追加需求(2026-07-29,dev-music,已实现):音频线两条极简策略入核心

用户批准的两条(playground 实验版先行,现入管线;`--audio` 总开关):

- **对白音画同步(prompt-only,零加价)**:剧本逐镜可选 `dialogue`
  (≤6 词,仅角色近景开口时;skill 明确禁旁白滥用)→ ShotEntry.dialogue
  入台账/to_brain_line/conditioning。执行器出口**确定性追加**口型子句
  (`_with_dialogue`:引号台词 + "嘴随词动" + **压制背景音**话术 ——
  生成端只出人声,BGM 由 §F 统一配,两层不打架;引号串去重防重复);
  该镜生成调用临时开 `generate_audio`(try/finally 还原,非对白镜保持
  静音即经济);全修闭包同款(hint 替换正文后子句重追加,修复不丢对白)。
  brain/enhancer 均不写台词(三处 skill 注明,防重复冲突)。
- **scene 级 BGM(一 scene 一曲,曲内自洽 = 跨段一致性的构造性解法)**:
  剧本新输出 `music_plan`("scene N" → 情绪+流派+BPM 一句;缺省=刻意
  静场),归一化后存 StoryboardMemory.music_plan(持久化)。新模块
  `pipeline/audio_stage.py`:§E concat 前音轨统一(对白镜有声、静音镜
  补无声 AAC 轨,否则 -c copy 拼接出坏文件)→ §F `add_music`(逐 scene
  text-to-music(sonilo 首选/ace_step 备选,走 `_run_task` 自动进调用
  日志)→ 按 scene 起止铺音乐床 → 有人声 sidechain 闪避(0.02/9/
  200ms/500ms)→ 两遍 loudnorm -14 LUFS → movie_scored.mp4)。诚实链:
  计划空 → 响亮记录静音片;任一步失败 → 保留无配乐正片,增强层绝不毁片。

对抗审查修正(提交前专项审查,6 处):对白镜异常兜底 t2v 补口型子句
(否则音频开着台词丢了);口型子句移到引用闸门之后(闸门丢 prompt 不再
陪葬子句);scene 号无标注时**沿用上一镜**(旧"归 1"会把续接镜错标、
音乐床错位);逐镜时长探测失败 → 拒绝配乐(不铺错位的床);终混人声轨
apad + duration=longest + -t 收口(防截掉画面尾巴);两遍 loudnorm 测量
脆断 → 单遍兜底。ffmpeg ≥ 4.4 依赖已注明。

测试:tests/unit/test_audio_line.py(9 条:解析/持久化/子句去重/scene
跨度/逐 scene 生曲与偏移/诚实静音)。全套 485 通过。
## 追加事故修正(2026-07-29,已实现):接缝闪烁根因与两道防线

现场:movie_20260729_150307,三镜(i2v + extend×2),接缝处画面闪烁。
取证链:shot001/002 各多出 48 帧(=2.0s 尾段)且容器时长照裁后声称
(239 帧@24fps ≈9.96s vs 声称 7.96s);movie.mp4 总帧 599 = 全部未裁
帧数;concat 按谎报时长排偏移 → 每个接缝 2 秒区间两镜帧交错 = 闪烁,
且接缝先回放上一镜尾段内容。

根因:`_trim_head` 旧实现 `-ss` 前置 + `-c copy` 是流拷贝,只能在关键
帧下刀;AI 生成短片常整段一个关键帧(实测 shot001 唯一关键帧在 1.83s)
→ 一帧没裁,仅时长元数据改小。

- **防线一 `_trim_head` 重写**:解码级精确裁(`-ss` 后置 + libx264
  重编码,音轨一并裁齐,avoid_negative_ts)+ 【裁后自检】(输出时长
  ≉ 原时长-裁量 → None,说谎文件绝不放行,调用方按既有降级带痕使用
  未裁版)。
- **防线二 concat 完整性闸**(tools/video_concat.py):拼接前逐文件
  校验"帧数×帧率 ≈ 容器时长"(±0.15s)+ 跨文件 编码/分辨率/帧率/
  像素格式一致;任一不过 → 响亮告警并改走 concat FILTER 重编码拼接
  (全解码重排时间戳,对元数据说谎免疫)。副作用同时消除:此前任何
  参数不齐的输入静默产出坏拼接的隐患。
- 连带自愈:scene 级音乐床的偏移取自 ffprobe 时长 —— 裁准后自动对齐
  (本次 run 的配乐原本也错位 2-4 秒)。

回归:tests/unit/test_flicker_fixes.py(5 条:稀疏关键帧精确裁【事故
同款前提,-g 999 复现】、自检拒谎、说谎时长/参数漂移判定、混帧率
重编码拼接)。全套 481 通过。

---

## 追加规矩(2026-07-30,用户口述,已实现):§E 终版清单与接缝分诊

用户两条规矩:"第一步确定每镜最终视频路径(错了就全完)";"extend
剪掉之前的视频片段、首帧生成的剪掉首帧;检查你看着办"。落地为
`_final_cut`(§E 第一步,decisions 留痕):

- **终版路径确定并核验**:台账 video_path 是唯一权威;逐镜打印
  "label → 文件 [策略]" 清单;文件缺失响亮告警 + decisions 记
  skip_missing 后跳过,绝不静默拼错片。
- **接缝按策略分诊**:extend_prev 生成时已裁头(裁后自检把关,拼装
  不再动);`_PREV_FRAME_LOCKED`(ti2v_prev_last / ti2v_prev_plus_
  keyframe / flf2v_bridge,首帧=上一镜尾帧)的镜 → 拼装时裁掉重复的
  第 0 帧。
- **两道检查(我方裁量)**:①先量后裁 —— `_first_last_mad` 实测
  上一镜末帧 vs 本镜首帧的平均像素差,< 8/255(实测标定:同帧不同质
  ≈5-6,正常相邻帧 ≈1-1.5,真不同帧 >12)才裁,量不出/不像 → 不裁;
  ②裁用 `_trim_head`(解码级 + 裁后自检)。首镜永不裁;接缝比较永远
  用上一镜【原片】末帧。

二次简化(同日,用户提议):切割全部前移到【生成时】——
`_drop_first_frame` 在 _generate_with_condition 内完成:硬锁路线
(ti2v_prev_last / flf2v_bridge,首帧由 API 参数锁死在上一镜尾帧)
**无条件切一帧**;软锁路线(ti2v_prev_plus_keyframe)先量后切
(junction_mad 记入 cond);extend 裁头维持生成时既有。下游评审/修复/
拼装看到的都是切好的版本;拼装层 `_final_cut` 只留"终版路径确定并
核验"清单,不再动文件。

回归:tests/unit/test_final_cut.py(5 条:清单缺失跳过/清单不动文件/
硬锁不量直切/软锁量后分流/裁失败诚实保留)。全套 500 通过。

---

## 追加需求(2026-07-30,用户批准,已实现):运镜衔接 + 音画同步评审

- **运镜交接(镜头也是运动物体)**:① scene_write 新增 CAMERA HANDOFF
  LAW —— 每镜 end_state 必须带镜头状态("camera: static / slowly
  pushing in / tracking right at walking pace"),切点上镜头运动只许
  延续或静止,**禁止方向反转**(推近收尾接拉远开场 = 跳切感头号来源);
  剧本 STRICT JSON 指令同步(window_loop);例子三处补镜头状态。
  ② window_generation junction 规则加 CAMERA HANDOFF(开场延续实况
  报告的镜头运动);③ prompt_enhancer 的 opening_state_actual 消费面
  从"位置+运动"扩到"+镜头运动"。④ 评审(mllm_backends):实况/交接棒
  文本含 camera 时自动注入"开场运镜是否延续,方向反转即败"检查项
  (无据不查)。场记的视频版指令本就要求报告镜头运动 —— 至此闭环。
- **音画同步入评审**:有台词的镜(conditioning.dialogue 非空),评审
  指令自动注入三查:台词说了且与剧本一致 / 口型与语音同步 / 人声之外
  干净(生成端压制了背景音,评审验证压制生效;BGM 由 §F 统一混)。
  Gemini 原生视频输入自带音轨,零额外成本。

回归:test_audio_line.py 增至 11 条(运镜检查注入与无据不注入、
台词三查注入与无台词不注入)。全套 502 通过。

---

## 追加进展(2026-07-31,dev-rl,已实现):S1 训练三件套 + 合成冒烟数据

- dev-music 全量合并入 dev-rl(闪烁修复/音频线/运镜交接均在,508→509)。
- `make_synthetic_runs.py`:按 S0 真实格式伪造 run 日志(条件/润色/修复
  决策 + 结局记录,好坏齐备)——训练链路冒烟专用,不用于真实收益;
  6 run × 4 shot 实测产出 38 KTO / 18 SFT / 15 DPO 对 / 6 holdout。
- 训练脚本(训练机运行,依赖 requirements-rl.txt,不进包依赖):
  `train_sft.py`(completion_only_loss)→ `train_kto.py`(权重自动配平
  1:1~4:3,batch≥4)→ `train_dpo.py`(rpo_alpha=1.0 防 chosen 坍缩,
  label_smoothing 可调);统一 `train_config.yaml`(Qwen3-8B + QLoRA
  全线性层;max_prompt 6144 / completion 512;truncation keep_end)。
- **token mask 口径写入 README**:单步转移 → 题干整体不计损失、只训
  completion;无 ReAct 式观测 mask 需求;长程唯一手动项 = 从左截断
  (keep_end)保住 THIS TURN JSON;信用分配在标签层不在 token 层。

---

## 追加需求(2026-07-31,用户裁决 1-4,已实现):角色视觉锚 + episode 降级

现场取证先行:movie_20260729 的坏 keyframe **不是** episode 继承所致
(该 run 图计划 via=llm)——真凶是 brain 用中文写了 t2i prompt(模型
I/O 英文法度无运行时闸)。两案并修:

- **裁决 1:episode 记忆只作 guidance,绝不直接继承**。_decide 的
  via="episode" 短路整体废除:replay 命中改为 `episode_recommendation`
  注入上下文(策略名 + "past task 建议,本次条件优先"),决策一律由
  brain 做;window_generation skill 规则 1 同步改写(可跟可推翻,
  reason 里说明)。三条旧契约测试改写为新契约。
- **裁决 2(第一版)+ 3(不固定 seed)+ 4(跨片库现在做):角色官方
  肖像(单视图视觉锚)**。新 §A' 阶段 `_ensure_cast_portraits`:逐 cast
  角色按 用户素材(描述符词重叠≥0.5)> 跨片库 > t2i 取像;t2i prompt =
  static 半句逐字 + 全片 setting/光线(不学 ViMax 白底,免重打光);
  产物三处登记 —— storyboard.portraits(持久化)/ asset_memory
  (identity 前缀 portrait:,媒体目录与检索即刻可见)/ 跨片库
  `memory/character_library.py`(目录 + index.json;命中 = 同名 且
  描述符词重叠 ≥0.6,同名不同长相绝不误配;uses 台账)。
- **肖像流入三处**:参考通道策略(t2v_own_refs / ti2v_prev_plus_
  keyframe)自动追加 @ImageN 肖像槽位(清单编号与装配顺序严格一致,
  出场角色子集);conditioning 带 role=identity_portrait → 评审收
  "IDENTITY PORTRAIT(外观按此判)"图证 —— 外观检查从对文字升级为
  对图;window_generation skill 新增 portraits 节。
- **英文法度落闸**:决策 prompt 后缀加 "ALL output text in ENGLISH";
  image_plan skill 明写(含 field bug 注记);t2i/检索词含 CJK →
  运行时响亮告警。

回归:test_character_portraits.py(6 条)+ 三条 episode 契约测试改写。
全套 515 通过。

## 2026-07-31(下午)肖像双大 bug 修复 + 裁决:ViMax 式关键帧替换修复

事故(outputs/movie_20260731_144652 实锤):
1. shot2 起所有镜的首帧成了肖像照,人物又和肖像对不上 —— image_plan 把
   官方肖像当素材检索回来做本镜图:同一张图双通道进引用列表(计划图 +
   自动附挂),正面全身像支配开场构图;
2. 肖像本身背景全错(影棚白布 + 错误服装)—— 三因叠加:§A' 生肖像时
   storyboard.setting 还没赋值(拿到空场景,掉进 "the film scene" 空话
   兜底);中文全角分号";"绕过 static/dynamic 拆分器,契约标签原文进
   t2i;scene_write 输出了中文描述符(语言律没管到这层)。

修法(全部落地 + 525 测试全绿):
- 肖像专用通道封死:_asset_catalog/_media_catalog 排除 portrait: 前缀;
  _execute_image_plan 出口按路径守卫(撞肖像 → 响亮丢弃如实降级);
  image_plan skill 明写"肖像自动附挂,绝不自己计划";
- own 图为空但有肖像 → 参考路线照常(菜单肖像感知;t2v_own_refs /
  ti2v_prev_plus_keyframe 装配 [尾帧]+肖像,编号与槽位清单一致,兜底
  模板肖像槽位写"match appearance, do not copy pose/framing");
- 肖像质量链:setting 赋值移到 §A' 之前;_static_half 全角兼容 + 关键词
  兜底(标签绝不外漏);肖像背景 = 影片 setting 具体词(空则响亮告警 +
  中性底,空话兜底废除);scene_write 增 LANGUAGE LAW(视觉字段必须英文,
  名字/台词可留用户语言)+ 解析层 CJK 确定性告警;
- 角色库消污:data/character_library 两条坏条目隔离进 quarantine_20260731,
  index 清空(不清会同名命中复用坏图);
- 【裁决 1】人物漂移 → ViMax 肖像替换:repair_keyframe_identity 进修复
  菜单(三重门控:编辑客户端 + 关键帧 + 官方肖像),执行 = seedream-v4
  多图编辑(images[0]=关键帧, images[1]=肖像)→ 修好的帧正式顶替台账
  首帧(replaced_from 留痕)→ 原条件重跑;image_edit.edit 增 references
  参数,size 按被编辑图宽高比推导(写死方幅会逼模型重构图)。

真 API 验证(scripts/playground/portrait_fix_validation.py,三轮迭代):
- 新肖像 ✅ 背景即影片场景(雨天街角面包店);
- 肖像替换指令迭代出三条铁律(单一事实源 identity_repair_instruction):
  显式绑定"第一张=底片、第二张=参考"(缺了 → 模型自由发挥成 3D 卡通);
  钉住 photographic live-action photorealistic(防风格漂移);
  full-bleed no borders(防胶片边框装饰)。第三轮:换人保景完全成功。

另:frame-2 瞬移实锤(shot000_w_s301 帧 1→2 MAD 13.58,正常 1~1.5)——
i2v 钉帧只服从约两帧后按文字先验重画人+景;钉帧完整性 MAD 闸门已列入
待裁决方案(骨架 §G),本轮未实现。

## 2026-08-02 假成片事故 + extend 雪崩 + REFERENCE-FIRST 裁决

事故(outputs/movie_20260731_180541,另一台机器,ffmpeg 未安装):
1. movie.mp4 打不开(moov atom not found)—— 真凶不是分支合并:git 实证
   audio_stage.py 自落地起零改动(e611962..HEAD diff=0),§E/§F 完好;
   是 video_concat.py 的远古"沙箱兜底"在 ffmpeg 缺失时写了 683 字节
   "MOCK CONCAT" 文本冒充 movie.mp4,§F 拿到假文件配乐全灭;
2. extend 雪崩:裁尾(_cut_tail)与裁头(_trim_head)都依赖 ffmpeg,
   全失败 → 每镜以整段上镜原片为续接源且不裁头,8s→16s→25s→30.9s,
   30.9s 冒充 6s 镜、内含前两镜画面(台账 untrimmed=True 如实记了,
   但瘸腿产片仍到了用户手里)。

修法(全部落地,525 测试全绿):
- video_concat 假产物兜底拔除:ffmpeg 缺失/输入非真视频 → 响亮
  RuntimeError,绝不写假 mp4(旧测试改写为锁新契约);
- 窗口管线入口硬预检:ffmpeg/ffprobe 不在 PATH → 当场拒跑(附安装
  指引);cv2 缺失 → 响亮警告(段修降级);
- §E 对账:终版逐镜实测时长 vs 计划时长,偏差 > max(1.5s, 30%) 响亮
  告警 + decisions 记录;合成失败从 info 升为 WARNING + decisions;
- 【裁决】REFERENCE-FIRST 法则(用户 2026-08-02):有人物/场景参考图
  (含自动附挂肖像)时必选参考通道(续接→ti2v_prev_plus_keyframe,
  切换→t2v_own_refs);extend_prev 看不见任何参考图,仅限无参考图的
  无缝同镜续接 —— skill 决策规则 4 重写 + extend_prev 菜单描述同步;
- 该次运行已就地修复:repair_20260802/ 按台账 extended_from 逐镜裁头
  (7.96/8.96/5.96s 全对上计划),重拼 38.96s 可播 movie.mp4,真 sonilo
  BGM 混出 movie_scored.mp4(h264+aac,-14 LUFS)。

## 2026-08-02(二)§G 钉帧完整性闸门落地(用户批准,默认关闭)

用户裁决:"这个你可以先加,然后默认关闭"。实现:
- `_pin_frame_mad(video, work_dir)`:第 1 帧 vs 第 2 帧平均像素差(/255),
  ffmpeg+numpy 纯算术零成本;算不出 → None(不猜不拦)。体温表:健康
  相邻帧 ≈1~1.5,同瞬间重画 ≈5~6,"第二帧瞬移"实录 13.58;
- 适用路线 _PIN_GATE_ROUTES = 开场被图钉住的五条(i2v_keyframe /
  ti2v_prev_last / flf2v_own_pair / flf2v_bridge / ti2v_prev_plus_keyframe),
  按 cond 实际路线判定(降级后不误判);
- 触发(阈值默认 0=关;开启荐 8.0):超阈 → decisions 记 pin_gate/reroll
  → 同条件 seed+1000 免费重掷一次 → 复测;仍超阈 → 保留 + still_tripped
  留痕(交给评审,不空手);重掷异常 → 保留被拦原片 + reroll_failed;
- 台账诚实:重掷后 cond.seed 改记实际 seed(s+1000),pin_gate_mad 记
  重掷后的干净测量;对白镜重掷同步 generate_audio 开关;
- 入口:generate_movie_windowed(pin_gate_mad=0.0) + test_window_movie
  --pin-gate-mad;测试 4 条(真 ffmpeg 合成瞬移片实测 + 布线三态)。

### §G 对抗核查修正(同日,多智能体三视角审查,6 条实锤全修)
- 【高】裁头路线盲区:ti2v_prev_last / flf2v_bridge /(去重触发时)
  ti2v_prev_plus_keyframe 的交付片已被 _drop_first_frame 裁掉重复钉帧,
  撕裂前移到帧 0→1 甚至只在接点可见 —— 测量升级为【开场撕裂度 =
  max(接点差, 帧0→1, 帧1→2)】,上镜锚定三路线连接点(上镜末帧 vs
  本片帧 0)一起量;末帧取法改"截尾 0.5s 倒放取首帧"(低帧率下
  -sseof -0.05 会落空);
- 未裁路线帧 0→1 撕裂同被覆盖(原实现 0 基索引量的是第 2↔3 帧);
- 重掷后复测按重掷的【实际】路线门控(内部降级到无钉路线不复测,
  不给无钉片记假 pin 失败);
- 开闸却测不了 → log.warning + decisions 记 measure_failed(哑火必响);
- pin_gate_mad 逐候选噪声不计入 per_seed 分歧判定;
- 措辞纠偏:测量零成本,但重掷是一次真金生成费(换掉更贵的评审→
  修复弯路)—— CLI 帮助与注释不再写"免费"。
测试 7 条(含 0→1 撕裂、接点撕裂、measure_failed、per_seed 噪声),
全套 532 通过。
