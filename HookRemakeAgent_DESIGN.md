# HookRemakeAgent — 爆款 Hook 视频复刻（换人 + 换品）Agent 框架设计

> 给 Claude Code 看的项目搭建说明书，风格对标 [`LongVideoEditAgent_DESIGN.md`](./LongVideoEditAgent_DESIGN.md)。
> 目标：输入「一条爆款视频 + 一组 hook 素材（目标人物 / 目标商品）」，输出「同节奏、同动作、同运镜，但人和商品都换成我们自己的」复刻成片。
> 分支约定：按仓库惯例落 `dev-music`，其他分支须合回。
> 框架骨架借鉴 `ViMax/`（agent 类 + 注入 chat_model + Pydantic 结构化输出 + duck-typed 生成器协议），**但不要 ViMax 的分镜创作策略**——爆款视频本身就是现成的分镜，我们的任务是"翻拍"，不是"创作"。

---

## 0. TL;DR — 一段话讲清楚

**比喻：我们是一个翻拍剧组。** 原爆款视频 = 一份已经排练好的"完整演出"——舞蹈动作、机位、剪辑节奏、卡点全都是现成的；hooks = 我们自己的"新演员"（比如穿粉色 T 恤的女生）和"新道具"（指定商品）。我们要做的不是重新编舞，而是：**把原片拆成施工图 → 给新演员分配角色 → 按施工图逐段重拍 → 按原时间轴逐帧装回去。**

术语版：对爆款视频做 shot 检测与人物身份聚类，把碎镜头**重组回拍摄单元（take）**；用选角表（cast plan）把 hook 素材映射到每个身份；按镜头特性走三条生成路线（**MIX 视频换人 / MOVE 首帧重绘+动作迁移 / STILL 首帧改绘+图生视频**），生成单元是"拼带（reel）"而不是碎片；最后按原始帧级时间轴 conform 回去，保住剪辑节奏和音乐卡点。质检 agent 对每个片段打分，不合格走重试梯子。

**为什么能落地**：主力模型阿里云百炼 `wan2.2-animate-mix`（视频换人：保留原视频动作/表情/环境，把人换成参考图中的人）单次调用支持 **2s–30s** 输入视频——爆款的碎镜头拼成拼带后恰好在这个窗口内，一条 60s 的爆款 ≈ 12–18 次生成调用。

---

## 1. 输入输出契约

### 1.1 输入

```python
inputs = {
    "viral_video": "<url|path>",   # 爆款原片，30s–3min，竖屏为主，切镜快（0.5–3s/镜），可能多人
    "hooks": {
        "person_hook_1": "<url>",  # 目标人物素材：图或短视频（如：穿粉色T恤的女生，正面/侧面/全身……）
        "object_hook_1": "<url>",  # 目标商品素材：白底图/实拍图/短视频
        "person_hook_2": "<url>",
        "object_hook_2": "<url>",
        # ... person_hook_N / object_hook_N
    },
    "config": "remake.yaml",       # 选角策略、预算上限、路线开关、字幕策略等
}
```

**hooks 的语义约定**（关键，写死在 MaterialLibrarian 的分类逻辑里）：

| 键名 | 是什么 | 用途 |
|---|---|---|
| `person_hook_k` | 目标人物的参考素材。**理想形态是"人已经穿着/拿着目标商品"的照片**（如穿粉色 T 恤的女生），一图同时携带人和商品的外观 | MIX/MOVE 路线的角色参考图；按景别裁出 face / half / full 变体 |
| `object_hook_k` | 商品的干净参考（白底、多角度、logo 清晰） | 商品特写镜头的改绘参考；QC 阶段商品保真度对照的 ground truth |

两条规则：
1. **成对解释，独立入库。** `person_hook_k` 与 `object_hook_k` 是"第 k 套换装方案"；素材入库后打散成变体库，规划时按镜头需要取用。
2. **N 对 hook = N 个成片变体。** 拆片、身份聚类、施工图只做一次；生成阶段按 k 套 cast 各跑一遍（"一次拆片、N 次换装"），这是本框架相对逐条手工复刻的最大成本优势。

### 1.2 输出

```
outputs/remake_<ts>/
├── remake.mp4                 # 成片（每套 cast 一个：remake_cast1.mp4 ...）
├── deconstruct.json           # 拆片结果：shots / identities / takes（对标 Maestro 的 storyboard.json 地位）
├── cast_plan.json             # 选角表
├── remake_plan.json           # 施工图：每个 reel/fragment 的路线、参考图、生成参数
├── qc_report.json             # 每片段质检得分、重试历史
└── assets/                    # hook 变体库、关键帧、位姿轨迹、生成中间产物
```

每个 stage 落盘一份 checkpoint JSON，**可断点续跑**（对齐 ViMax 的 render checkpoint 与 Maestro 的 storyboard rev 习惯）。

---

## 2. 三个核心难题与对策（先比喻，后术语）

### 2.1 视频太碎 → "碎纸拼页"：拍摄单元重组（take reconstruction）

**比喻：** 爆款视频像一页被碎纸机剪碎又重新排列的纸。逐个碎片临摹（逐 shot 生成）又贵又对不齐笔迹；正确做法是**先按纹理把碎片拼回原来那几页纸，整页重新誊写，再按原来的剪法剪开、按原来的顺序摆好**。

术语版：快切视频的几十个碎镜头，实际来自少数几个**拍摄单元（take）**——同一人物、同一场景、同一机位风格的连续表演（A-roll 舞蹈、商品特写桌拍……），剪辑时被交错排列成 `A1 B1 A2 C1 A3 …`。我们：

1. **聚类**：按「身份聚类结果 + 场景嵌入 + 镜头卡片相似度」把 shots 聚成 takes；
2. **拼带（reel）**：把同一 take 的碎片按剪辑顺序拼接成一条连续视频，作为 `wan2.2-animate-mix` 的驱动视频——**换人模型逐帧跟随驱动视频，拼带内部的硬切只是内容跳变，输出会原样带着这些切点回来**，等于"生成即预剪"；
3. **conform（装回）**：按帧会计表把生成结果切回原时间轴。

为什么必须拼带，而不是可选优化：
- `wan2.2-animate-mix` **输入视频下限 2s**——爆款里大量 0.5–1.5s 的碎镜头单独送根本不合法；
- 同一 take 的碎片共享一次生成 → 人物外观天然一致（不存在跨调用漂移）；
- 计费按输出秒数，拼带不增加计费秒，但调用次数从"每碎片一次"降到"每 take 一到两次"，排队/轮询开销大幅下降。

（拼带在切点处可能出现 1–2 帧的时序伪影——模型的时间先验在硬切处被打断。QC 的 `BOUNDARY_JUMP` 检查项专门盯这个，命中则把该拼带在切点处拆成两次调用重跑。详见 §5。）

### 2.2 不止一个人 → "选角表"：身份聚类 + 按帧内人数选路线

**比喻：** 翻拍剧组开机前先做一张**选角表**：原片里的"红衣主舞"由我们的"粉 T 女生"出演，"背景路人甲"保留原样或换成路人乙。之后每个镜头拍摄时只看选角表办事，不再临场决定——这就是多人、碎镜头下保证全片一致的机制：**替换决策定在"身份"层，执行落在"镜头"层。**

术语版：
1. 用 InsightFace（人脸）+ 服装颜色直方图 + 身体 ReID 特征，把全片所有镜头里出现的人聚成 `identity_A / B / C…`，统计每个身份的出镜时长、商品交互次数、平均画幅占比；
2. CastingDirector（LLM）产出 cast_plan：主角（出镜最长 + 商品交互最多）→ `person_hook`；其余身份按策略 `keep / replace / minimize`；
3. **执行约束（实测 API 决定的）**：`wan2.2-animate-mix` 只替换"视频中的主角"，**没有指定替换第几个人的参数**。因此：
   - 帧内**只有 1 个待换的人**（或待换者显著主导画面）→ 走 MIX；
   - 帧内**≥2 个待换的人**→ 走 MOVE（首帧整幅重绘出所有目标人物，再用原片动作驱动整幅图）；
   - "对 MIX 输出再跑一遍 MIX 换第二个人"的串行方案**降级为实验项**：模型的主角判定可能再次锁定刚换上的人，不可控。

### 2.3 多个 hook 怎么嵌进去 → "戏服间"：变体库 + 按景别配参考图

**比喻：** 好剧组不会拿一张证件照应付所有戏；特写戏用面部定妆照，全身舞蹈用全身定妆照。hooks 进库后先加工成一间"戏服间"，每个镜头开拍时按景别去取对应的那一件。

术语版（这就是"多 hook 参考嵌入"的全部机制）：
1. **入库拆解**：每个 `person_hook` 产出变体 `{face_crop, half_body, full_body}`（视频 hook 则抽帧去重后再裁）；每个 `object_hook` 产出 `{clean_product, in_context}` + 商品要点卡（颜色/形状/logo 文字/材质）；
2. **按景别匹配**：百炼官方指引"确保输入图片与参考视频中的人物**画幅占比相似**"→ RemakePlanner 为每个 take 选参考图变体：特写 take 配 `face_crop`/`half_body`，全身舞蹈 take 配 `full_body`；
3. **prompt 注入**：MOVE/STILL 路线的改绘 prompt 里注入人物外观卡与商品要点卡的文字描述（双保险：图管相似度，文字管细节兜底）；
4. **QC 对照**：`object_hook` 的干净图作为商品保真度检查的 ground truth。

---

## 3. 系统总览

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Inputs: viral_video + hooks{person_k, object_k} + remake.yaml              │
└────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌【S1 资产入库】MaterialLibrarian ──────────────────────────────────────────┐
│ 下载/探针 → 分类(人/物/图/视频) → 变体库(face/half/full…) → 外观卡(VLM)     │
│ 产物: assets/bank.json                                                     │
└──────────┬─────────────────────────────────────────────────────────────────┘
           ▼
┌【S2 拆片】Deconstructor ──────────────────────────────────────────────────┐
│ shot检测(TransNetV2) → 微镜头合并 → 关键帧 → 镜头卡(VLM) → OCR字幕/ASR/节拍 │
│ → 身份聚类(人脸+ReID) → 商品检测 → take 重组                                │
│ 产物: deconstruct.json  (shots / identities / takes)                       │
└──────────┬─────────────────────────────────────────────────────────────────┘
           ▼
┌【S3 选角+施工图】CastingDirector → RemakePlanner ─────────────────────────┐
│ 选角表(身份→hook) → 逐take定路线(MIX/MOVE/STILL/KEEP) → 拼带切块(≤28s)     │
│ → 补齐规则(<2s) → 逐take配参考图变体 → 逐fragment写降级预案                 │
│ 产物: cast_plan.json + remake_plan.json                                    │
└──────────┬─────────────────────────────────────────────────────────────────┘
           ▼
┌【S4 生成】ShotProducer (并行, 异步任务池) ────────────────────────────────┐
│ MIX:  拼带 ──────────────── wan2.2-animate-mix ──→ 已换人拼带              │
│ MOVE: 首帧重绘(qwen-image-edit 多图参考) → wan2.2-animate-move ──→ 整幅重拍 │
│ STILL:首帧改绘 → wan-i2v / 2.5D 运镜 ──→ 商品特写/B-roll                    │
│ 产物: assets/gen/reel_*.mp4                                                │
└──────────┬─────────────────────────────────────────────────────────────────┘
           ▼                       ┌───────────────────────────────┐
┌【S5 质检+合片】────────────────┤ 不合格 → 重试梯子(§10):        │
│ QualityInspector: 人脸相似度/商品│ 换seed → 换路线 → 降级 → 人工  │
│ 保真/动作保真(MAD)/伪影/切点跳变 └───────────────────────────────┘
│ Conformer: 帧会计表切回原时间轴 → 调色统一 → 字幕重写覆盖 → 音轨铺回 → 成片  │
│ 产物: remake.mp4 + qc_report.json                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

数据流关键不变量：**原片的帧级时间轴从 S2 起就冻结**（fragment 的 in/out 帧号），后续所有生成、重试、合片都对着这张表做帧会计，音乐卡点因此天然保住——节奏是"抄"来的，不是"对"出来的。

---

## 4. 五个 Stage 详设

### S1 资产入库（MaterialLibrarian）

干什么：把 hooks 从"一堆 URL"变成"可按需取用的戏服间 + 文字外观卡"。

| 步骤 | 工具 | 说明 |
|---|---|---|
| 下载与探针 | `probe_media` | 时长/分辨率/fps；视频 hook 抽帧（1fps + 峰值清晰帧） |
| 分类 | VLM (`qwen-vl-max`) | person / object / person+object；是否已穿戴商品 |
| 变体加工 | InsightFace + 裁剪 | `face_crop`(1:1) / `half_body`(3:4) / `full_body`(9:16)；分辨率不足则超分一次 |
| 外观卡 | VLM，Prompt P1 | 人物卡沿用 ViMax `CharacterInScene` 的 static/dynamic 二分；商品卡含 logo 文字逐字抄录 |
| 合规检查 | `check_image` 预跑 | 百炼对输入图有内容审核；提前发现会被拒的素材（如无授权名人脸） |

失败处理：某 hook 无法下载/审核不过 → 标记该套 cast 不可用，不阻塞其他套。

### S2 拆片（Deconstructor）

干什么：把原片变成结构化的"施工现场勘测报告"。全部是**离线重计算，一次缓存**（对齐 LongVideoEditAgent 的离线/在线分离原则）。

1. **shot 检测**：TransNetV2 为主（对快切鲁棒），PySceneDetect ContentDetector 兜底交叉验证；输出帧级切点。
2. **微镜头处理**：连续 `<8` 帧的闪切归并为 `burst_group`（视为一个节奏单元，整组同路线处理）。
3. **关键帧**：每 shot 取 first/mid/last 三帧。
4. **镜头卡（shot card）**：VLM 逐 shot 产出结构化 JSON（Prompt P2）：景别、相机运动、在场人物框、商品可见性（worn/held/standalone/none）、动作短语、手-商品交互、叠加文字区域、光线。
5. **音频**：ASR（faster-whisper / paraformer）出口播文稿；librosa 出节拍网格（仅用于 QC 与可选的变奏重剪，不参与主流程——节奏靠 conform 天然保住）。
6. **OCR**：PaddleOCR 逐关键帧提取烧进画面的字幕/贴纸及其区域（S5 字幕策略的输入）。
7. **身份聚类**：InsightFace 人脸嵌入(主) + OSNet ReID + 服装色直方图(辅)，DBSCAN 聚类 → `identity_*`，带出镜统计。侧脸/背面帧靠 ReID 与服装接力。
8. **take 重组**：确定性算法先提案（同身份集合 + 场景嵌入余弦 > 阈值 + 景别相邻 → 同 take），LLM 仲裁边界样例（Prompt P4 的前半部分），输出 `take_*` 与其 fragment 列表。

产物 `deconstruct.json` 骨架：

```jsonc
{
  "video": {"fps": 30, "frames": 1800, "w": 1080, "h": 1920},
  "shots": [
    {"id": "sh_007", "in_f": 312, "out_f": 353, "burst_group": null,
     "card": {"framing": "MS", "camera": "handheld_slight", "persons": [{"identity": "id_A", "bbox_share": 0.42, "facing": "front"}],
              "product": {"visibility": "worn", "interaction": "none"},
              "action": "spins to face camera, hands on hips",
              "overlay_text": [{"text": "3 colors!!", "region": [0.1, 0.72, 0.9, 0.8]}]}}
  ],
  "identities": [
    {"id": "id_A", "shots": ["sh_001", "sh_007"], "screen_time_s": 21.4, "product_interactions": 9,
     "avg_bbox_share": 0.4, "desc": "female dancer, red crop top, ponytail"}
  ],
  "takes": [
    {"id": "tk_01", "identity_ids": ["id_A"], "scene": "studio white wall",
     "fragments": ["sh_001", "sh_007", "sh_012"], "total_s": 11.2}
  ]
}
```

### S3 选角与施工图（CastingDirector + RemakePlanner）

**CastingDirector**（Prompt P3）：输入身份统计 + 资产库 + 策略配置，输出 cast_plan：

```jsonc
{
  "cast_k": 1,
  "mapping": [
    {"identity": "id_A", "action": "replace", "person_hook": "person_hook_1", "reason": "主角: 出镜21.4s, 商品交互9次"},
    {"identity": "id_B", "action": "keep",    "reason": "背景路人, 无商品交互, 占比<8%"}
  ],
  "product_mapping": {"source": "red hoodie", "target": "object_hook_1 (pink t-shirt)"}
}
```

`action` 取值：`replace` / `keep` / `minimize`（能裁掉就通过构图裁掉，用于法务或审美上不想保留的路人）。

**RemakePlanner**（Prompt P4）：逐 take 定路线 + 切拼带 + 配参考图 + 写降级预案。**路线判定表**（写进 prompt，也写进代码里做硬校验）：

| 条件（按优先级） | 路线 | 生成单元 |
|---|---|---|
| 帧内待换人数 ≥2；或手持商品需换且交互复杂；或叠加文字覆盖人物主体 | **MOVE** | 逐 fragment（首帧重绘 + 动作驱动） |
| 帧内待换人数 = 1，人物主导画面，商品为穿戴式（参考图已含商品） | **MIX** | 拼带（同 take 碎片拼接，2s ≤ 长 ≤ 28s） |
| 无人物、商品特写 / B-roll / 文字卡 | **STILL** | 逐 fragment（首帧改绘 + i2v；<1.5s 的用 2.5D 运镜） |
| 素材为用户自有且配置允许 | KEEP | 原样保留（默认关闭） |

拼带切块与补齐规则（约束驱动，全部来自已验证的 API 限制，见 §6）：
- 单条拼带 ≤ **28s**（30s 上限留 2s 余量）；超长 take 在碎片边界处切块；
- 拼带总长 ≥ **2s**；孤儿碎片（该 take 只有一个 <2s 碎片）→ 回文补齐（正放+倒放拼到 ≥2s，生成后掐回原长）或直接降级 STILL；
- 每条拼带记**帧会计表**：`fragment_id → (reel_id, reel_in_f, reel_out_f)`，conform 的唯一依据；
- 每个 fragment 写好 `fallback_route`（MIX→MOVE→STILL 的降级链）。

### S4 生成（ShotProducer）

异步任务池（并发 4–8，尊重百炼 RPS 与排队），三条路线的调用形态：

| 路线 | 调用链 | 文字 prompt？ |
|---|---|---|
| MIX | `upload_public(reel)` → `animate_mix(image=ref_variant, video=reel_url, mode=wan-std/pro)` → 轮询 → 下载 | **无**（该 API 只吃图+视频；一致性全靠参考图选对变体） |
| MOVE | `edit_image(首帧, refs=[person_variant, object_clean], instruction=P5产出)` → `animate_move(image=redrawn, video=fragment_url)` | 改绘 instruction 由 Prompt P5 生成 |
| STILL | `edit_image(首帧, refs=[object_clean], instruction=P5)` → `generate_i2v(image, prompt=P6产出)` 或 `still_2p5d(image, move=zoom_in_slow)` | i2v prompt 由 Prompt P6 生成（**相机指令放第一句**——07151cf 的教训） |

工程细节：
- 百炼要求输入为**公网可访问、纯 ASCII 的 URL** → 所有中间产物先传 OSS 拿预签名 URL（`upload_public` 工具）；
- 任务 ID 与结果 URL **24h 失效** → 生成完成立即回捞落盘；
- 输出 fps 未在文档承诺 → 下载后 `probe_media` 实测，与原片 fps 不一致时在 conform 阶段用 ffmpeg `minterpolate`（或最近帧映射）重定时——**帧会计表以原片帧号为准，重定时只发生在生成素材一侧**；
- 失败即重试（tenacity，指数退避 ×3），内容审核类失败（错误码指向 censor）不重试、直接走降级链。

### S5 质检与合片（QualityInspector + Conformer）

**QualityInspector** 逐 fragment 打分（工具算数值，VLM 出判词，Prompt P7 汇总裁决）：

| 检查 | 工具 | 通过线（v0.1 初值，跑批后校准） |
|---|---|---|
| 人脸相似度 | ArcFace cos(生成帧人脸, person_hook 人脸) | ≥ 0.42（取 fragment 内中位数） |
| 商品保真 | DINOv2 crop 相似度 + VLM 对照 object_hook 要点卡（logo 逐字比对） | VLM 结论 pass 且无 `logo_text_wrong` |
| 动作保真 | DWPose 逐帧关键点，源 vs 生成的归一化 **MAD**（沿用仓库现行验收口径） | MAD ≤ 阈值（对齐 Maestro 现行值） |
| 伪影 | VLM 抽 5 帧看手部/面部/商品形变 | 无 `severe` |
| 切点跳变 | 拼带切点前后 2 帧帧差 + VLM | 无 `BOUNDARY_JUMP` |
| 叠字泄漏 | OCR 生成帧 | 原片品牌字未泄漏进成片 |

不合格 → §10 重试梯子。

**Conformer**（纯确定性工具，不是 LLM agent）：
1. 按帧会计表从生成拼带里切出每个 fragment，落回原时间轴（帧级精确，moviepy/ffmpeg concat）；
2. 跨 take 调色统一：以 person_hook 色调为锚做直方图匹配 / LUT；
3. **字幕策略**：原片烧字所在区域，用我们重写的字幕（Prompt P8：LLM 按新商品改写 OCR 文稿，保节奏保字数）以不透明底条覆盖或直接压上——MOVE/STILL 路线的画面天然无原字，MIX 路线靠覆盖；
4. 音轨：默认铺回配置指定的 BGM（原 BGM 是否可用属版权问题，交给配置项 `audio.bgm`）；原口播若提及原品牌 → ASR 文稿经 P8 改写后走 CosyVoice TTS 重配（v0.2 选配）；
5. 导出 9:16 H.264，帧数与原片逐帧对齐。

---

## 5. 切分策略专章：分钟级碎视频 → 可生成的单元

（这是需求里点名要想清楚的问题，单独成章。）

**分层切分阶梯：**

```
L0 整片(60–180s)
 └─ L1 shots: TransNetV2 帧级切点 (60s 快切片典型 30–60 个)
     └─ L2 burst 归并: <8帧闪切并组 (节奏单元, 整组同路线)
         └─ L3 takes: 身份+场景聚类, 碎片归源 (典型 8–20 个)
             └─ L4 reels: take 内碎片按序拼接, 切成 2–28s 的块  ← 生成单元
```

**为什么生成单元是 L4 而不是 L1**：三个硬理由——① API 下限 2s，碎镜头单送不合法；② 同 take 共享一次生成，人物外观零漂移；③ 调用数从 ~45 降到 ~15，排队时延减半以上。**为什么上限 28s 而不是一口气整片**：API 上限 30s；且块越大，一处伪影重跑的代价越大——28s 是"一致性收益"与"重试粒度"的折中，跨块一致性由同一张参考图变体保证。

**conform 帧会计（装回去的数学）**：

```
fragment sh_007: 原片 [312, 353) 帧, 41 帧 @30fps
  → 位于 reel_tk01_a 的 [96, 137) 帧
  → 生成后从 gen_reel_tk01_a.mp4 取 [96, 137) 帧
  → 放回成片时间轴 [312, 353)
不变量: Σ(成片各fragment帧数) == 原片总帧数; 每个切点帧号与原片逐一相等
```

音乐卡点由此**结构性成立**：切点没动过，动的只是画面内容。librosa 节拍网格仅作为 QC 断言（切点集合与节拍格的偏差分布应与原片一致）。

**边界情况处理表：**

| 情况 | 处理 |
|---|---|
| 孤儿碎片 <2s（take 内仅此一片） | 回文补齐到 ≥2s 生成后掐回；或降级 STILL |
| take 总长 >28s | 在碎片边界切块，各块用同一参考图变体 |
| 切点跳变伪影（QC 命中） | 该拼带在命中切点处一分为二重跑 |
| 同帧多个待换人物 | 该 take 整体改走 MOVE（逐 fragment） |
| 0.3s 级闪切商品特写 | STILL 路线 2.5D：改绘一张图 + ffmpeg zoompan，肉眼无法分辨真伪 |
| 转场特效帧（叠化/滑动） | 归并进相邻 fragment，conform 后用 ffmpeg xfade 按原时长复刻转场 |

---

## 6. 模型选型矩阵

**主力（百炼，与 `dev-music-bailian` 现有账号/封装同源）——已查证的关键约束：**

| 用途 | 模型 | 已验证的关键约束 |
|---|---|---|
| **视频换人（MIX 主力）** | `wan2.2-animate-mix` | 输入视频 **2–30s**、≤200MB、宽高 ∈[200,2048]、比例 1:3–3:1；参考图 ≤5MB、∈[200,4096]；`mode: wan-std / wan-pro`；只换"主角"，**无多人指定参数**；输入输出均过内容审核；按输出秒计费；结果 URL 24h 失效；官方建议参考图与视频**人物画幅占比相似** |
| **动作迁移（MOVE 主力）** | `wan2.2-animate-move` | 输入约束同上；把参考**图**里的人按驱动视频的动作演起来，背景来自参考图（所以先重绘首帧再驱动 = 整幅换掉） |
| 首帧改绘/重绘 | `qwen-image-edit`（指令编辑）；升级位：`qwen-image-2.0-pro` / `wan2.7-image-pro`（官方标注支持**多图参考生成**） | 多图参考 = 人物变体 + 商品干净图同时喂入 |
| 图生视频（STILL） | 百炼 wan i2v 家族（以当期[模型列表](https://www.alibabacloud.com/help/zh/model-studio/models)为准，运行时探针决定型号） | 5–10s 档；短碎片生成后掐帧 |
| 镜头理解 VLM | `qwen-vl-max` / qwen3-vl 系 | 镜头卡、QC 判词 |
| 文本 LLM | `qwen-max`（或仓库当前默认 chat model） | 选角、规划、prompt 生成、字幕改写 |
| TTS（v0.2 选配） | CosyVoice | 口播重配 |

**备选与逃生通道：**

| 场景 | 备选 |
|---|---|
| MIX 质量不满意 / 需 >30s 单段 | 自托管 **Wan2.2-Animate-14B**（开源，ComfyUI/批量队列成熟，无 30s 硬限，重活可下放本地 GPU） |
| 商用第三方对照 | Kling 动作控制、Runway Act-Two、Viggle（作为 A/B 对照基线，不进主链路） |
| 视频编辑端点 | WaveSpeed `seedance-2.0/video-edit`（Maestro `video_gen_backends.py` 已有封装，可直接复用其轮询/下载代码） |

**感知栈（全部本地开源，离线一次）：** TransNetV2 + PySceneDetect（shot）、InsightFace/ArcFace（脸）、OSNet（ReID）、DWPose/RTMW（位姿）、GroundingDINO（商品框）、PaddleOCR（烧字）、faster-whisper（ASR）、librosa（节拍）、DINOv2（商品/场景嵌入）。SAM2 仅在 v0.2 需要掩码级操作时引入。

---

## 7. Agent Crew：六个智能体 + 一个确定性合片器

沿用仓库"六 agent 剧组"的编制传统与 ViMax 的实现风格（普通类 + 注入 `chat_model` + 模块级 prompt 常量 + Pydantic 输出 + tenacity 重试；生成器走 `tools/protocols.py` 的 duck-typing 协议）。

| # | Agent | 角色比喻 | 输入 → 输出 | 主要工具 | LLM? |
|---|---|---|---|---|---|
| 1 | **MaterialLibrarian** 资产管理员 | 戏服间管理员 | hooks → `bank.json` | probe/裁剪/VLM(P1) | VLM |
| 2 | **Deconstructor** 拆片师 | 现场勘测员 | 原片 → `deconstruct.json` | shot/pose/聚类/OCR/ASR/VLM(P2) | VLM |
| 3 | **CastingDirector** 选角导演 | 选角导演 | identities+bank → `cast_plan.json` | LLM(P3) | ✔ |
| 4 | **RemakePlanner** 复刻规划师 | 制片主任(排施工图) | takes+cast → `remake_plan.json` | LLM(P4)+硬校验器 | ✔ |
| 5 | **ShotProducer** 生成执行 | 摄制组 | plan → 生成素材 | animate_mix/move、edit_image、i2v、oss；prompt 由 P5/P6 生成 | ✔(写prompt) |
| 6 | **QualityInspector** 质检员 | 剪辑室审片人 | 素材 → `qc_report.json` + 重试指令 | face_sim/product/pose_mad/VLM(P7) | VLM |
| 7 | Conformer 合片器 | 剪辑台 | 全部素材 → `remake.mp4` | ffmpeg/moviepy/LUT；字幕改写走 P8 | ✘(纯工具) |

**编排**：`HookRemakePipeline`（对标 ViMax `Script2VideoPipeline`）：S1‖S2 并行 → S3 串行 → S4 并行池 → S5 逐件质检、汇总合片。QC 不合格件经 RemakePlanner 的降级链回注 S4（最多 2 轮，预算封顶）。所有 LLM 决策与 tool 调用写结构化 JSONL trajectory log——**直接对齐 `rl/` 现有的 step-level GRPO 训练数据格式**，路线选择（MIX/MOVE/STILL）天然是一个可 RL 的离散动作空间（v0.3+）。

---

## 8. Tool Call 设计（完整 Schema）

命名 snake_case；全部异步；生成类工具返回 `{job_id}` + 轮询获取（与百炼异步任务模型一致）。以下为 LLM function-calling 格式的完整 schema（也是代码里 Pydantic 参数模型的蓝本）。

### 8.1 生成类（核心）

```jsonc
{
  "name": "animate_mix",
  "description": "视频换人: 保持驱动视频的动作/表情/运镜/环境, 把画面主角替换为参考图中的人物。用于 MIX 路线。输入视频须 2-30s、公网URL。",
  "parameters": {
    "type": "object",
    "properties": {
      "ref_image_url":  {"type": "string", "description": "人物参考图公网URL。必须选用与驱动视频人物画幅占比相似的变体(face/half/full)"},
      "video_url":      {"type": "string", "description": "驱动视频(拼带)公网URL, 2-30s, ≤200MB, 宽高∈[200,2048], 比例1:3-3:1"},
      "mode":           {"type": "string", "enum": ["wan-std", "wan-pro"], "description": "std=快/省, pro=画质优。首轮std, 重试或主打镜头用pro"},
      "check_image":    {"type": "boolean", "default": true}
    },
    "required": ["ref_image_url", "video_url", "mode"]
  },
  "returns": {"job_id": "string"}
}
```

```jsonc
{
  "name": "animate_move",
  "description": "动作迁移: 把参考图中的人物按驱动视频的动作/表情演绎, 画面背景取自参考图。用于 MOVE 路线(先用 edit_image 重绘首帧, 再整幅驱动)。",
  "parameters": {
    "type": "object",
    "properties": {
      "ref_image_url": {"type": "string", "description": "重绘后的首帧(含目标人物+目标商品+近似原构图)"},
      "video_url":     {"type": "string", "description": "原片该fragment的视频URL(提供动作), 约束同 animate_mix"},
      "mode":          {"type": "string", "enum": ["wan-std", "wan-pro"]}
    },
    "required": ["ref_image_url", "video_url", "mode"]
  },
  "returns": {"job_id": "string"}
}
```

```jsonc
{
  "name": "edit_image",
  "description": "指令式图像编辑(qwen-image-edit / wan2.7-image-pro)。多图参考: 在保持原图构图/姿态/光线的前提下替换人物身份与商品。",
  "parameters": {
    "type": "object",
    "properties": {
      "base_image_url": {"type": "string", "description": "待改绘的原片关键帧"},
      "ref_image_urls": {"type": "array", "items": {"type": "string"}, "maxItems": 3, "description": "参考图: [人物变体, 商品干净图]"},
      "instruction":    {"type": "string", "description": "英文编辑指令, 由 Prompt P5 生成, 含明确的 REPLACE 清单与 KEEP 清单"},
      "n":              {"type": "integer", "default": 2, "description": "候选张数, QC择优"}
    },
    "required": ["base_image_url", "instruction"]
  },
  "returns": {"image_urls": ["string"]}
}
```

```jsonc
{
  "name": "generate_i2v",
  "description": "图生视频(百炼 wan i2v 家族)。用于 STILL 路线: 改绘后的首帧 + 运动描述 → 短视频。",
  "parameters": {
    "type": "object",
    "properties": {
      "image_url": {"type": "string"},
      "prompt":    {"type": "string", "description": "英文运动prompt, 由 Prompt P6 生成; 第一句必须是相机指令"},
      "duration_s":{"type": "integer", "description": "生成时长档(5/10), 取≥fragment时长的最小档, conform时掐帧"},
      "resolution":{"type": "string", "enum": ["720p", "1080p"], "default": "1080p"}
    },
    "required": ["image_url", "prompt", "duration_s"]
  },
  "returns": {"job_id": "string"}
}
```

```jsonc
{
  "name": "still_2p5d",
  "description": "静帧2.5D运镜(本地ffmpeg zoompan/crop漂移)。用于 <1.5s 的商品闪切: 一张改绘图 + 缓推/缓拉/手持微晃, 零API成本。",
  "parameters": {
    "type": "object",
    "properties": {
      "image_url":  {"type": "string"},
      "move":       {"type": "string", "enum": ["zoom_in_slow", "zoom_out_slow", "pan_lr", "handheld_jitter"]},
      "duration_f": {"type": "integer", "description": "输出帧数(=fragment原始帧数)"},
      "fps":        {"type": "number"}
    },
    "required": ["image_url", "move", "duration_f", "fps"]
  },
  "returns": {"video_path": "string"}
}
```

### 8.2 感知类（确定性，S2 调用；schema 从简，参数即签名）

| 工具 | 签名 → 返回 |
|---|---|
| `probe_media` | `(url) → {kind, w, h, fps, duration_s, frames}` |
| `detect_shots` | `(video, backend="transnetv2") → [{in_f, out_f, confidence}]` |
| `sample_keyframes` | `(video, shot) → {first, mid, last: png_path}` |
| `extract_pose_track` | `(video, in_f, out_f) → dwpose_keypoints.npz` |
| `cluster_identities` | `(video, shots, faces+reid) → identities.json` |
| `detect_product` | `(keyframe, text_query) → [{bbox, score}]`（GroundingDINO） |
| `ocr_overlay` | `(keyframe) → [{text, region}]` |
| `asr_transcribe` | `(audio) → [{t0, t1, text}]` |
| `beat_grid` | `(audio) → {bpm, beat_times[]}` |
| `upload_public` | `(local_path) → {url}`（OSS 预签名，纯 ASCII） |
| `poll_job` | `(job_id) → {status, video_url?}`（统一轮询；完成即回捞落盘，防 24h 失效） |

### 8.3 质检类

```jsonc
{
  "name": "face_similarity",
  "parameters": {"video_path": "string", "ref_face_url": "string", "sample_n": 8},
  "returns": {"median_cos": 0.0, "min_cos": 0.0, "frames_no_face": 0}
}
{
  "name": "product_fidelity",
  "parameters": {"video_path": "string", "ref_product_url": "string", "product_card": "object"},
  "returns": {"dino_sim": 0.0, "logo_text_match": true, "vlm_verdict": "pass|fail", "notes": "string"}
}
{
  "name": "motion_fidelity",
  "parameters": {"gen_video": "string", "src_video": "string", "in_f": 0, "out_f": 0},
  "returns": {"pose_mad": 0.0, "worst_window": [0, 0]}   // 归一化关键点MAD, 口径对齐仓库现行验收
}
{
  "name": "artifact_scan",
  "parameters": {"video_path": "string", "focus": ["hands", "face", "product", "cut_boundaries"]},
  "returns": {"issues": [{"code": "ARTIFACT_HANDS|BOUNDARY_JUMP|OVERLAY_LEAK|...", "severity": "minor|severe", "t": 0.0}]}
}
```

### 8.4 合片类

| 工具 | 签名 |
|---|---|
| `conform_cut` | `(gen_reel, frame_ledger) → fragments/*.mp4`（fps 不符时先重定时） |
| `assemble_timeline` | `(fragments, timeline) → silent.mp4`（断言: 帧数逐一相等） |
| `color_match` | `(video, anchor_ref) → video`（直方图匹配/LUT） |
| `render_captions` | `(video, captions[], regions[]) → video`（覆盖原烧字区 + 压新字） |
| `mux_audio` | `(video, bgm, vo?) → remake.mp4` |

---

## 9. Prompts（完整文本，模型侧一律英文）

沿用 ViMax 的 `[Role][Task][Input][Output][Guidelines]` 模板与"角色卡 static/dynamic 二分"约定；输出全部走 Pydantic `format_instructions`。以下为 v0.1 全文（代码中作为模块级常量放各 agent 文件顶部）。

### P1 · MaterialLibrarian — hook 素材卡

```
[Role]
You are an asset librarian for a video remake production. You catalog reference
materials (people and products) so that downstream agents can pick the right
reference for each shot.

[Task]
Analyze the given hook asset (image, or frames sampled from a short video) and
produce a structured asset card.

[Input]
One or more images of the same asset, plus the asset's declared slot name
(person_hook_k or object_hook_k).

[Output]
{format_instructions}
// AssetCard fields:
// kind: "person" | "object" | "person_with_object"
// person_static: physical appearance that rarely changes (face shape, hair, build, skin tone)
// person_dynamic: attire, accessories, held items — INCLUDING the target product if worn/held
// product_card: {name_guess, colors, shape, material, logo_text (transcribe EXACTLY,
//                character by character), distinctive_marks[]}
// best_use: subset of ["face_closeup", "half_body", "full_body", "product_closeup"]
// quality_flags: subset of ["blurry", "occluded", "extreme_angle", "low_res", "watermark"]

[Guidelines]
- Describe only what is visible. Never invent logo text: if unreadable, write "unreadable".
- person_static / person_dynamic must be concrete and visualizable (colors, lengths,
  shapes), never abstract ("stylish", "pretty" are forbidden).
- If the person wears or holds the product, say so explicitly in person_dynamic and
  fill product_card as well (kind = "person_with_object").
- best_use reflects what this specific image can serve as a generation reference for:
  e.g. a tight face photo cannot serve "full_body".
```

### P2 · Deconstructor — 镜头卡（VLM，逐 shot）

```
[Role]
You are a shot analyst for a video remake production. Your shot cards are the
construction blueprint: downstream agents decide replacement strategy purely
from your cards, without watching the video.

[Task]
Given keyframes (first/mid/last) of ONE shot from a fast-cut vertical video,
produce a structured shot card.

[Input]
Three keyframes in temporal order, shot duration in seconds, and person tracking
boxes precomputed by the perception layer (identity ids with bounding boxes).

[Output]
{format_instructions}
// ShotCard fields:
// framing: "ECU" | "CU" | "MCU" | "MS" | "FS" | "WS"
// camera: "static" | "handheld_slight" | "pan_L" | "pan_R" | "tilt" | "zoom_in" | "zoom_out" | "whip"
// persons: [{identity_id, bbox_share (0-1), facing: "front"|"profile"|"back", is_dominant: bool}]
// product: {visibility: "worn"|"held"|"standalone"|"none",
//           interaction: "none"|"pointing"|"holding_static"|"manipulating"}
// action: one imperative phrase describing the dominant motion, ≤15 words
// overlay_text: [{text, region [x0,y0,x1,y1] normalized}]
// lighting: one short phrase
// replace_difficulty: "easy" | "medium" | "hard" , with reason

[Guidelines]
- CAMERA FIRST: judge camera motion before anything else, from framing drift
  across the three keyframes. Do not confuse subject motion with camera motion.
- bbox_share is the fraction of frame area; is_dominant = true for at most one person.
- interaction = "manipulating" only when hands visibly operate/deform/rotate the
  product (this routes the shot to the expensive full-redraw path — be strict).
- Transcribe overlay text EXACTLY, including emoji and numbers.
- replace_difficulty = "hard" iff: ≥2 non-background persons, OR interaction =
  "manipulating", OR overlay text covers the person's body region.
```

### P3 · CastingDirector — 选角表

```
[Role]
You are the casting director of a remake production. The original viral video
has several recurring people; the client provides replacement references
(hooks). You decide who gets replaced by whom.

[Task]
Produce a cast plan mapping every identity cluster to an action.

[Input]
1. Identity clusters with stats: screen_time_s, product_interactions,
   avg_bbox_share, appearance description, example shot ids.
2. Asset bank cards (from the librarian), one per hook slot.
3. Client policy: {primary_person_hook, extras_policy: "keep"|"replace"|"minimize",
   product_hook}.

[Output]
{format_instructions}
// CastPlan fields:
// mapping: [{identity_id, action: "replace"|"keep"|"minimize",
//            person_hook: str|null, reason}]
// product_mapping: {source_desc, target_hook, appears_standalone: bool}
// risks: [{shot_id|identity_id, note}]   // e.g. two replaced identities co-occur in sh_014

[Guidelines]
- The PRIMARY identity is the one with the highest combination of screen time and
  product interactions — it MUST map to the client's primary person hook.
- Never map two identities to the same person hook if they ever co-occur in one
  shot (they would become twins on screen); flag it in risks instead.
- "minimize" means: prefer routes/crops that exclude this person, keep them only
  when unavoidable.
- Every mapping needs a one-sentence reason quoting the stats.
```

### P4 · RemakePlanner — take 仲裁 + 路线施工图

```
[Role]
You are the line producer of a remake production. You turn the shot cards, take
grouping proposal and cast plan into an executable production plan. Your plan is
consumed by machines: every field must be exact.

[Task]
1. Adjudicate the take grouping proposal (merge/split ambiguous cases).
2. Assign each take (or fragment) a generation route.
3. Cut takes into reels of 2–28 s and pick a reference variant per reel.
4. Write a fallback route for every fragment.

[Input]
1. deconstruct.json (shots + cards + takes proposal + identities)
2. cast_plan.json
3. Asset bank (available reference variants per hook: face/half/full …)
4. Hard constraint table (routing rules + API limits) — you MUST obey it; a
   deterministic validator will reject plans that violate any limit.

[Output]
{format_instructions}
// RemakePlan fields:
// take_fixes: [{take_id, op: "merge"|"split", detail}]
// items: [{unit_id, unit_kind: "reel"|"fragment",
//          route: "MIX"|"MOVE"|"STILL"|"KEEP",
//          fragments: [fragment_id...],
//          ref_variant: {hook, variant},        // e.g. {person_hook_1, full_body}
//          mode: "wan-std"|"wan-pro",
//          pad_rule: "none"|"palindrome",
//          fallback_route, notes}]

[Guidelines]
- Route decision order (first match wins):
  1. ≥2 to-replace persons in frame, or product interaction = "manipulating",
     or overlay text covers the person → MOVE (per fragment).
  2. Exactly one dominant to-replace person, product worn (reference already
     includes it) → MIX (reel of same-take fragments, 2–28 s).
  3. No person / standalone product / text card → STILL
     (< 1.5 s fragments: still_2p5d; else edit + i2v).
  4. KEEP only if client config allows.
- Reference variant must match the take's framing: CU/MCU → face or half variant;
  MS/FS/WS → full_body. Mismatched framing is the #1 cause of identity drift.
- wan-pro only for: the 3 highest-impact reels (opening hook, product reveal,
  finale) or retry attempts. Everything else wan-std.
- A reel must never mix fragments from different takes.
```

### P5 · ShotProducer — 首帧改绘指令生成（MOVE / STILL 用）

```
[Role]
You write surgical image-editing instructions for an instruction-following image
editor (qwen-image-edit class, with reference images attached).

[Task]
Given a source keyframe description, the target person card and the target
product card, write ONE editing instruction in English that swaps identity and
product while keeping everything else.

[Input]
1. Shot card of the fragment (framing, camera, action, lighting).
2. Target person card (static + dynamic) and reference images attached.
3. Target product card (colors, shape, logo_text) and reference image attached.
4. Overlay-text regions to erase, if any.

[Output]
A single instruction string with EXACTLY this structure:
  REPLACE: <who/what to swap, referencing "the person in reference image 1" and
            "the product in reference image 2", with key appearance words>
  KEEP: <explicit list — camera framing, subject pose and position, background,
         lighting, color mood>
  REMOVE: <overlay text / watermarks / original brand marks, or "nothing">
  STYLE: photorealistic, consistent with the original frame's lens and grain.

[Guidelines]
- The KEEP list is a contract, not decoration: always name pose ("keep the exact
  body pose and limb positions"), composition, and lighting direction.
- Logo text: instruct the editor to render the product's logo text EXACTLY as
  "<logo_text>" — misspelled logos are automatic QC failures.
- Never introduce new objects, new people, or text.
- One instruction ≤ 120 words. No lists inside REPLACE; flowing clauses only.
```

### P6 · ShotProducer — STILL 路线 i2v 运动 prompt

```
[Role]
You write motion prompts for an image-to-video model. House rule learned the
hard way: THE CAMERA INSTRUCTION COMES FIRST, or the model invents its own
camera work.

[Task]
Given the fragment's shot card (original camera + action + duration), write the
motion prompt for regenerating this fragment from its redrawn first frame.

[Output]
A prompt string with EXACTLY this order:
1. Camera sentence FIRST: static / slow push-in / handheld micro-shake …
   mirroring the original shot card's camera field. If the original camera is
   static, write "Locked-off static camera. No dolly, no orbit, no zoom."
2. Subject motion: what moves and how much, amplitude-matched to the original
   action phrase ("subtle", "moderate", "energetic").
3. Constraint tail: "No new people, no new objects, no text. The product stays
   fully visible and undeformed."

[Guidelines]
- Never exceed the original motion amplitude: an i2v fragment that moves MORE
  than the original reads as fake immediately; less is safe.
- Duration/fps are API parameters, not prompt text — do not mention seconds.
- ≤ 60 words total.
```

### P7 · QualityInspector — 裁决

```
[Role]
You are the screening-room inspector of a remake production. Numeric metrics
are computed for you; your job is the final verdict and an actionable retry hint.

[Task]
For one generated fragment, combine the metric report and your own reading of
the sampled frames into a verdict.

[Input]
1. Metric report: {face_median_cos, dino_sim, logo_text_match, pose_mad,
   artifact_issues[], boundary_issues[]}  + thresholds.
2. 5 sampled frames of the generated fragment + the matching source frames +
   the person/product reference images.
3. The fragment's route and retry history.

[Output]
{format_instructions}
// Verdict fields:
// pass: bool
// failure_codes: subset of ["FACE_MISMATCH","PRODUCT_WRONG","LOGO_TEXT_WRONG",
//   "MOTION_DRIFT","ARTIFACT_HANDS","ARTIFACT_FACE","BOUNDARY_JUMP",
//   "OVERLAY_LEAK","ENV_BROKEN","CENSOR_REJECT"]
// severity: "retry_same" | "switch_route" | "degrade" | "human_review"
// retry_hint: one sentence for the planner, e.g. "use half_body variant instead
//   of full_body — face too small in reference caused identity drift"

[Guidelines]
- Metrics gate, eyes decide: if metrics pass but a sampled frame shows an obvious
  defect (six fingers, melted logo), fail it and say which frame.
- Be specific in retry_hint: name the reference variant, mode upgrade (wan-std →
  wan-pro), or route switch you recommend.
- FACE_MISMATCH on a CU/MCU fragment is always at least "switch_route" severity
  after one failed retry — do not burn budget re-rolling the same setup.
```

### P8 · Conformer — 字幕改写（合片阶段）

```
[Role]
You rewrite the burned-in captions of a viral video for a product swap remake.

[Task]
Given the OCR'd caption track (text + timing + region) and the target product
card, rewrite each caption for the new product.

[Output]
{format_instructions}
// captions: [{t0, t1, region, text}]  — same count, same timing as input

[Guidelines]
- Keep each caption within ±20% of the original character count (it must fit the
  original region and rhythm).
- Preserve the emotional register (hype, urgency, humor) and emoji density.
- Replace every mention of the original product/brand; never mention the original.
- Keep the language of the original captions (Chinese stays Chinese).
```

---

## 10. QC 阈值与重试梯子

**重试梯子（每 fragment 独立走，全程记入 qc_report.json）：**

```
第0轮  按 remake_plan 生成 (MIX 默认 wan-std)
  │ fail
  ▼
第1轮  同路线重掷: MIX → mode 升 wan-pro / 换参考图变体(按 retry_hint)
  │ fail
  ▼
第2轮  换路线: MIX → MOVE;  MOVE → STILL;  (STILL → still_2p5d)
  │ fail
  ▼
第3轮  降级保底: 该 fragment 用同 take 内已通过的相邻 fragment 定格/慢放填充
  │ fail
  ▼
人工审核队列 (整片其余部分照常交付)
```

预算护栏：单 fragment 最多 3 次生成调用；整片重试调用数 ≤ 计划调用数 × 0.5，超限即停并出报告。`CENSOR_REJECT`（内容审核拒绝）不重掷，直接跳到换路线或人工。

---

## 11. 成本与时延量级（60s 爆款，单套 cast）

| 项 | 量级 | 依据 |
|---|---|---|
| 生成调用数 | 12–18 次 MIX/MOVE + 3–6 次 STILL + 6–10 次图像编辑 | 45 shots → ~14 takes → ~16 reels |
| 计费视频秒数 | ≈ 70–90s（60s 成片 + 15–50% 重试） | animate 按输出秒计费，输入不计费 |
| 端到端时延 | 感知 3–5 min（单 GPU）+ 生成 10–25 min（并发 6，异步轮询）+ 合片 <2 min | 生成是长尾，并发决定墙钟 |
| 第 2..N 套 cast 增量 | 只重复 S4/S5，感知与规划零成本 | "一次拆片、N 次换装" |

---

## 12. 目录骨架与配置

```
hook_remake/
├── pipeline.py                 # HookRemakePipeline: 五阶段编排 + checkpoint/resume
├── interfaces.py               # AssetCard/ShotCard/Identity/Take/CastPlan/RemakePlan/Verdict (pydantic)
├── agents/
│   ├── material_librarian.py   # P1
│   ├── deconstructor.py        # P2 (+确定性感知工具编排)
│   ├── casting_director.py     # P3
│   ├── remake_planner.py       # P4 (+硬校验器 validate_plan())
│   ├── shot_producer.py        # P5/P6 + 任务池
│   └── quality_inspector.py    # P7
├── tools/
│   ├── protocols.py            # 沿用 ViMax duck-typing: AnimateGenerator/ImageEditor/...
│   ├── bailian_animate.py      # wan2.2-animate-mix / -move 封装(提交+轮询+回捞)
│   ├── bailian_image.py        # qwen-image-edit / wan-image 封装
│   ├── wavespeed_edit.py       # 备选: 复用 Maestro video_gen_backends 的轮询代码
│   ├── perception.py           # transnetv2/insightface/dwpose/ocr/asr/beat
│   ├── qc_metrics.py           # face_sim/product_fidelity/motion_fidelity(MAD)/artifact
│   └── conform.py              # 帧会计/assemble/color_match/captions/mux
├── prompts/                    # P1–P8 (模块常量的单一来源, agent 文件 import)
├── configs/remake.yaml
└── outputs/
```

`configs/remake.yaml` 最小示例：

```yaml
chat_model: {provider: bailian, model: qwen-max}
vlm: {model: qwen-vl-max}
animate: {mix: wan2.2-animate-mix, move: wan2.2-animate-move, default_mode: wan-std,
          pro_quota: 3, max_reel_s: 28, min_clip_s: 2.0}
image_edit: {model: qwen-image-edit, candidates: 2}
i2v: {model: auto}          # 运行时读百炼模型列表探针决定
cast:
  primary_person_hook: person_hook_1
  product_hook: object_hook_1
  extras_policy: keep        # keep | replace | minimize
captions: {strategy: rewrite_cover}   # rewrite_cover | strip | keep
audio: {bgm: path/or/original, vo: strip}   # 原口播默认去除, v0.2 支持改写重配
qc: {face_cos_min: 0.42, pose_mad_max: <对齐Maestro现行值>, max_retries_per_fragment: 3}
budget: {max_total_calls_ratio: 1.5}
```

**与既有代码的复用点**：ViMax `tools/protocols.py` 的 duck-typing 协议与 agent 类范式直接沿用；Maestro `video_gen_backends.py` 的 WaveSpeed 提交/轮询/下载代码抽出复用；`rl/` 的 trajectory log 格式作为本框架 JSONL 日志的 schema 基准。

---

## 13. 风险清单与开放问题

| # | 风险 | 缓解 | 遗留 |
|---|---|---|---|
| 1 | animate-mix 的"主角"判定不可控（多人帧） | 路线判定表把多人帧全部导向 MOVE | 串行两遍 MIX 换两个人 = 实验项，需实测主角锁定行为 |
| 2 | 拼带切点伪影（模型时间先验被硬切打断） | QC `BOUNDARY_JUMP` + 命中即拆带重跑 | 切点伪影率需要跑批统计，决定拼带默认粒度 |
| 3 | 手持商品被 MIX 当作"环境"保留 → 商品没换掉 | `interaction=manipulating` 强制走 MOVE | held_static 的归属（人物还是环境）需实测 |
| 4 | 原片烧字在 MIX 输出中原样保留（含原品牌名） | OCR 检测 + 字幕覆盖策略；重字幕镜头走 MOVE | 花字/贴纸形状复杂时覆盖不干净 → v0.2 引入视频 inpainting（ProPainter） |
| 5 | 内容审核拒绝（真人素材敏感性） | S1 预跑 check_image；CENSOR_REJECT 不重掷 | hook 素材需客户提供肖像授权，流程外要有合规单 |
| 6 | 输出 fps/编码与原片不一致 | conform 前探针 + 重定时 | — |
| 7 | 版权：复刻的编舞/分镜本身可能构成实质性相似 | 全片零原始像素输出；BGM 走配置项由用户自担 | 产品层面需要法务口径，框架只保证"不搬运像素" |
| 8 | take 聚类错误（把两个机位并成一个 take） | LLM 仲裁 + 拼带内一致性由参考图兜底，错并的代价是伪影而非错人 | 聚类阈值需在真实爆款集上标 30 条调参 |
| 9 | 24h URL 失效 | 完成即回捞落盘，绝不存远端 URL 进 checkpoint | — |

**开放问题（需小实验定夺，建议各 ≤0.5 天）：**
E1 animate-mix 对拼带硬切的伪影率（20 条拼带跑批）；E2 多人帧串行 MIX 的主角锁定行为；E3 held 商品在 mix 模式下换不换；E4 参考图景别失配对人脸相似度的量化影响（决定变体库粒度）；E5 自托管 Wan2.2-Animate-14B 与百炼 wan-std 的质量/成本交叉点。

---

## 14. 里程碑

- **v0.1（1 周）**：S2 感知链全通（shot/聚类/take/镜头卡落盘可视化）+ MIX 单路线端到端（单人爆款、单套 cast、无 QC）→ 第一条"能看"的复刻片。
- **v0.2（2–3 周）**：三路线全通 + QC 重试梯子 + 字幕覆盖 + N 套 cast 批量 + E1–E4 实验报告回填路线判定表。
- **v0.3+**：QC 分数作 reward，接 `rl/` 的 step-level GRPO 训练 RemakePlanner 的路线选择策略；视频 inpainting 清字；自托管 animate 降本。

---

*已核对的外部依据：百炼 [wan2.2-animate-mix 视频换人 API](https://help.aliyun.com/zh/model-studio/wan-animate-mix-api)、[wan2.2-animate-move 图生动作 API](https://help.aliyun.com/zh/model-studio/wan-animate-move-api)、[qwen-image-edit API](https://www.alibabacloud.com/help/zh/model-studio/qwen-image-edit-api)、[百炼模型总列表](https://www.alibabacloud.com/help/zh/model-studio/models)。API 具体参数以当期文档为准，本文标注的 2–30s / 200MB / 24h 等硬约束均取自 2026-08 版中文文档。*
