# 低成本视频创作:流程漏洞审计与完善方案(2026-07-25,待用户裁决)

> 用户令:广泛调研,找出整个视频生成流程还可完善的策略与潜在漏洞,
> 目标是小成本创作。方法:四路并行(自家逐调用点成本审计 + 业界省钱
> 策略 + 便宜评审打法 + 四个参考仓机制盘点),全部结论带 file:line /
> URL,原文在 `cost_appendix/A-D`。**本文档是提案,未改任何代码。**

## 〇、先看实测账单(三次真实 run 的硬数字)

| Run | 生成调用 | 首次生成 | 修复/重生成占比 | 计费秒数 | 成片秒数 | 开销倍率 |
|---|---|---|---|---|---|---|
| attempt1(级联时代) | 18 | 4 | **78%** | ≈77s | 15s | **5.1×** |
| attempt2 | 14 | 4 | **79%** | ≈69s | 20s | 3.5× |
| attempt3(免级联后) | 11 | 5 | **55%** | ≈69s | 27s | **2.56×** |

attempt3 逐镜复盘更扎心:shot1 的 4s 段修被下一轮 8s 全修**直接作废**;
shot2 的 7s 全修被 verifier 拒(纯浪费);shot3/4 三次"接受"的全修
metric 反而一路下跌(0.772→0.7531)。**修复秒数里约 46% 是纯浪费**,
且四镜最终仍 generated_with_defects。结论:**最大的钱不在单价,在
"坏输入走到了付费那一步"和"不值得修的修了、修了又白修"。**

审计还抓到九个具体漏点(全带 file:line,附录 A §c),最扎眼的四个:
1. **段修复在"头部段"结构性不可用**——t2v 续接策略的上镜尾帧是
   reference 不是 first_frame 角色,propagate_repair 诚实返回 None,
   烧掉一轮还把 brain 推向全修(attempt3 shot2/3 四次段修全部空手);
2. **`video_fps: 5` 是 5 倍 Gemini 计费系数**——官方按帧计 token,
   fps=5 恰是文档里"调高采样"的示例;配合默认分辨率,单镜评审比
   1fps+低清贵约 20 倍;
3. **无预算帽、无成本台账、无跨 run 缓存**——attempt2/3 把同一部
   猫片从头重新买了一遍;image_edit 甚至完全没进调用日志;
4. **异常兜底双倍计费**——条件策略执行抛异常时,同一 seed 立刻再发
   一次裸 t2v,两次都付钱(window_loop.py:2058-2070)。

## 一、方案总览(按 省幅×置信÷工作量 排序)

### P0 —— 直接对着 55-79% 修复开销下刀

**① 修复经济学修正(最大单一杠杆)**
- (a) 修掉头部段修不可用:pin 类策略的上镜尾帧本来就在手上,作为
  head_anchor 传给 propagate_repair —— 段修复活了,全修次数应声降;
- (b) 成本入决策:orchestrator 上下文给每个修复选项标注相对成本
  (全修 = 整镜重买,段修 ≈ 半价,accept = 免费),route_suggestion
  配成本项;verifier 对全修候选提高接受线(+1 边际赢 + metric 下跌
  还接受 = attempt3 实锤的花钱倒退);
- (c) **需你裁决**:`repair_severity` 默认从 0(关)改为 0.5-0.6——
  你 7-17 裁过"暂时不动",但当时没有这份账单;账单显示轻缺陷追修
  是浪费主力之一。

**② 免费确定性预检(评审前的零成本闸)**
- ffmpeg 三件套:黑帧/冻结帧/镜内切换检测 + 时长校验(每镜 ~1-2s
  CPU,零 API);
- **首帧-锚点嵌入距离检测**:DINO/CLIP 算 shot 第 0 帧与 @Image1 锚
  的余弦距离(VBench 同款机制,与人评相关 >90%)——attempt3
  "没按 @Image1 开场"这类缺陷从此**免费秒判**,不用等 VLM。
- 通过预检的才进 VLM 评审;没通过的直接带确定性证据进修复。

**③ Gemini 评审开销分层(5-20× 削减;需你裁决,因 fps=5 是你定的)**
- 常规评审:fps=1 + media_resolution=low(单镜 ~$0.0002 级);
- **升级评审**:仅当便宜档报可疑或缺陷涉运动连续性时,才用 fps=5 全
  配置复审(运动缺陷确实需要帧密度,文献支持);
- verify_pair 的基线视频改走 Files API 复用,不再逐轮重传;
- 附录 C 有完整分层文献(路由/级联文献:~95% 质量下省 75-98%)。

**④ NEWTON 式条件预审门(参考仓现成机制,附录 D §2)**
- 生成**之前**,flash-lite 一次调用(~$0.0001)审"所选策略 + 引用图
  + prompt + 上镜实况"是否合理,带历史(拒绝重复已证死路);
- 原版设计三条铁律照搬:只判运动/数量不判外观、"务实不完美主义,
  只在会浪费生成时才否决"、异常时 fail-open;
- 一次否决 = 省一整次 480p 生成;插在 §C 和 generate 之间。

### P1 —— 结构性省钱

**⑤ 成本台账 + 预算帽**(OpenMontage cost_tracker 移植,挂
`_run_task` 单一出口):estimate→reserve→reconcile 三段账,超预算
成为新 stop_reason;每 run 一张成本快照进台账。
**⑥ fast 档 + 分辨率阶梯**:Seedance fast ≈ 便宜 20-26%(评价普遍
"接近正装质量");修复候选与草稿走 fast,终稿走 standard;720p 需求
改为"480p 全流程 + SeedVR2 上采样收尾"(WaveSpeed 自家,$0.15/5s,
测算省 ~45%)。⚠️ 前提:先按我们的 schema 调研纪律核实 WaveSpeed
是否暴露 fast slug(审计确认当前后端无 fast 配置)。
**⑦ 内容寻址剪辑缓存 + 断点续跑**:(model, prompt, 条件图 hash,
seed, duration, resolution) → 本地缓存;同参调用永不重买(attempt
重跑的最大补贴);ViMax 式 artifact-exists 全阶段跳过,崩溃重跑 $0。
**⑧ 剧本级确定性 lint**(OpenMontage 三校验器移植,已在借鉴清单③):
scene_write 出口做空话短语/镜头密度/时长-动作密度静态检查 + 已知好坏
剧本回归基准——坏剧本 $0 拦下,不再由 18 秒付费视频来发现。
**⑨ 5 秒探针 + extend 续投**(SWIFT 部分回滚思想的 API 版):8-10s
长镜先生成 4-5s 头段,过便宜闸后 extend 补全——探针直接成为成片头段,
不是沉没成本;恰好骑在我们 extend_prev 机制上。

### P2 —— 工程卫生与守护

**⑩ 小修一批**:异常兜底双倍计费修正;legacy 级联路径摘除(attempt1
78% 浪费的元凶,仍可达);image_edit 接入 call_log;enhancer
usable=False 不再静默降级(响亮告警,那正是 attempt3 shot2 白烧修复
预算的前因)。
**⑪ 合成 fixture 免费回归环境**(Crayotter 模式):moviepy 假视频 +
脚本化 brain/reviewer 应答,把 window loop 全阶段跑成零成本 E2E 测试
——以后管线改动先免费验证再上真钱。
**⑫ 明确不做**:本地开源模型草稿档(续接/参考通道语义跨模型不保,
破坏连续性机制;Seedance fast@480p 已够便宜)、盲目 best-of-N
(2 候选 = 保底 2×,只对历史高失败率镜头值得,做成自适应留待有数据)。

## 二、需要你裁决的四件事

1. `repair_severity` 默认值是否翻到 0.5-0.6(推翻 7-17"暂不动"裁决,
   依据是本次账单);
2. 评审 fps 分层(常规 1fps 低清、升级 5fps)——fps=5 是你亲手定的,
   分层方案保留了它在"运动复审"里的位置;
3. fast 档与 SeedVR2 收尾:先做 schema 核实再决定,还是直接跳过;
4. 实施顺序:建议 P0①② → P2⑩ → P0③④ → P1(⑤⑦优先),每批跑一次
   attempt 验证省幅再进下一批。

## 附录索引

- `cost_appendix/A_maestro_cost_audit.md` — 全调用点成本模型 + 三 run
  实测 + 九漏点(全 file:line)
- `cost_appendix/B_industry_strategies.md` — 业界策略目录(草稿-精修
  阶梯/快档/关键帧先行/best-of-N 经济学/缓存/本地模型,带价目)
- `cost_appendix/C_cheap_review_playbook.md` — 便宜评审手册(ffmpeg
  预检/fps 计费数学/级联评审/嵌入检测/开源打分器/帧密度证据)
- `cost_appendix/D_reference_repo_mechanisms.md` — 四参考仓 16 机制
  (NEWTON 预审门原文机理、OM 台账/缓存/校验器、ViMax 续跑/限流、
  Crayotter fixture)
