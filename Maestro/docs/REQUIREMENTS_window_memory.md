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
