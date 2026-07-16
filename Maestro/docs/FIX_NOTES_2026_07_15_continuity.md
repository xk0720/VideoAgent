# 修复记录:语义线索进 agent + 镜间动作连续性(2026-07-15,已实现)

> 对应两大问题:(一)brain 写 prompt 时看不全条件内容;(二)镜间动作
> 断裂(苹果停了又自己滚)。每条小问题 → 怎么解 → 代码在哪。全部有单测。

## 问题一:brain 能不能看到条件内容

### 1.1 图片描述被截断到 80 字
**解法**:取消截断,全文进 brain 上下文。
**代码**:`memory/storyboard.py to_brain_line`(`[:80]` 删除)。

### 1.2 素材图记的是"搜索词",不是"实际拿到的图"
**解法(语义跟着图走,四步)**:
1. 检索返回 (路径, **素材真实标签**):`_retrieve_asset_image` 返回二元组;
   目录项新增 `desc` 字段 = 干净语义(用户描述 > 入库 VLM caption > 文件名,
   Q-D 链),打分仍用带前缀的 label。
2. 台账双字段:`_execute_image_plan` 存 `description` = 实拿语义,
   检索词另存 `retrieval_query`(审计:"搜的"和"拿到的"分开记)。
   t2i 图的 description = t2i prompt;视频抽帧 = 源片段 caption。
3. 三条下游全部吃到真语义:brain 条件决策(to_brain_line 全文)、
   润色 agent(_conditions_for_prompt 读 description)、兜底模板
   (`_mention` helper:"@Image2 shows: an orange tabby cat — keep it
   consistent",语义缺失诚实退化,绝不编)。四个策略的 fallback prompt
   全部改为内容感知(t2v_own_refs / ti2v_prev_plus_keyframe /
   tiv2v_window / multi_image_fusion)。
4. skill 硬规则(window_generation):每个 @ImageN/"reference image N"
   必须说清它是什么、在本镜演什么,禁止裸写编号。
**代码**:`pipeline/window_loop.py`(_asset_catalog/_retrieve_asset_image/
_make_keyframe/_execute_image_plan/_desc_of/_mention/四个兜底模板)。

### 1.3 brain 的输入没日志
**解法**:`brain_calls.jsonl` 每条记录新增 `context` 字段 = 喂给 brain 的
完整 THIS TURN JSON(技能全文不重复存,`skill_chars` 已证明在场)。
四个决策点全覆盖:scene_write / image-plan / generation-condition(含
junction)/ repair decide(orchestrator `_build_user` 重构)/ 润色 agent。
**代码**:`window_loop._brain_pick/_write_outline`、
`agents/orchestrator.py`、`agents/prompt_enhancer.py`。

## 问题二:镜间动作连续性(ViMax 调研结论:它也没解这个,须自研)

### ① 剧本交接棒(治本)
**解法**:scene_write 逐镜多输出 `end_state`(一句话:切点瞬间谁在哪、
动/停、方向)。写作法则进 skill:下一镜开头必须从上一镜 end_state 继续;
**要接续运动就不许在切点停**(保持滚动过切点或滚动出画);静止物再动必须
写明新外力事件,否则剧本物理错误。例子重写为"交接棒相连"的三镜示范。
brain 没输出 = 空串,不编造。
**代码**:`window_loop._write_outline`(4 元组返回)、
`memory/storyboard.py ShotEntry.end_state`、`scene_write/SKILL.md`。

### ② 接点实况(治"照想象写")
**解法**:生成第 N+1 镜前,VLM 看上一镜**真实尾帧**出一句实况
("the apple is at rest at the center of the floor"),按 (帧, mtime)
缓存,一镜一次。实况 + 上一镜剧本 end_state + 本镜 required end_state
一起进条件 brain 上下文(`junction` 字段)和润色 agent 的 conditions
(kind="state" 三条)。skill 规则:**prompt 从实况起笔**;实况与剧本矛盾
→ 按实况写并在 reason 说明。诚实链:无上镜/无 VLM/尾帧抽不出/调用失败
→ 空串跳过,绝不编。
**VLM 双模式(用户裁决)**:`describe_junction` 由 GeminiVLM(API)与
**新后端 LocalQwenVLM**(本地 transformers 加载 Qwen2.5-VL,
`models.mllm.name: "qwen-local"` 切换)同名实现;LocalQwenVLM 惰性加载、
缺 torch/transformers 响亮报错;评审职责不归它(assess 返回 [] 并警告,
绝不冒充评审员)。
**代码**:`window_loop._junction_state/_JUNCTION_CACHE` + 主循环接线
(generate_movie_windowed 新参 mllm)、`mllm_backends.py`
(GeminiVLM.describe_junction/_JUNCTION_INSTRUCTION/LocalQwenVLM/注册表)、
`scripts/test_window_movie.py`(传 mllm)。

### ③ 防"镜尾刹车"
**解法**:skill 规则(不硬编码)——`required_end_state` 说仍在运动的,
prompt 结尾必须写 "still rolling as the shot ends — it does not slow
down or settle"(生成模型默认在片尾把运动刹停,不明说必翻车)。
**代码**:`window_generation/SKILL.md` 规则 7、`prompt_enhancer/SKILL.md`
state 条件说明。

### ④ 镜间衔接进评审
**解法**:`clip.conditioning` 新增 `end_state` + `junction_prev_actual`;
`review_shot` 构建指令时注入 junction 块——"上一镜实际结束状态是 X,
开头必须延续它(出一条 check)"+"剧本要求结束状态是 Y,按最后一刻判
(动/停有别,出一条 check)"。不符 → 定位到头段/尾段的 issue →
现有 regenerate_segment 即可修,衔接不再靠运气。
**代码**:`mllm_backends._SHOT_REVIEW_INSTRUCTION`(+{junction_block})、
`review_shot` 装配、`window_loop` conditioning 写入。

## 测试

`tests/unit/test_semantic_flow_and_continuity.py` 7 个:台账双字段、
兜底模板带语义、输入 context 落日志、junction 诚实链+缓存、状态条件
清单、qwen-local 注册+评审沉默、评审指令带接点检查。全套 424 通过。

## 已知边界(如实)

- 接点实况看的是尾帧【单帧】,"动/停"靠运动模糊和姿态推断 —— 单帧对
  匀速慢滚可能误判为静止;更准的做法是看尾段视频(成本更高,登记待议)。
- end_state 是 brain 的声明,剧本层可能仍写出矛盾(skill 只能教,不能
  保证);评审 ④ 是最后一道网。
- @ImageN 编号一致性(brain 自己在 prompt 里引用素材时如何保证编号与
  执行器装配一致)—— 用户已点名,方案另文,未动代码。

---

## 追加(2026-07-16,方案 A 已实现):@ImageN 编号一致性

**问题**:编号由执行器装配 payload 时决定,brain 在 prompt 里引用素材时
编号靠 skill 教它猜 —— 猜错无人拦截(用户点名,方案 A 获批)。

**解法:把编号从"要遵守的规则"变成"发给它的数据",出口再上确定性闸。**

1. **槽位清单**(`window_loop._slot_manifest`):纯函数,按策略算出执行器
   将装配的引用槽位 [{slot, content(实况语义), referenceable}],与
   `_generate_with_condition` 的装配顺序一一对应(单一事实源契约,
   `test_slot_manifest.py` 锁行为)。FIRST_FRAME/LAST_FRAME/kling 参考视频
   = referenceable=False(该路线无引用通道,prompt 只描述运动)。
2. **清单发给写 prompt 的人**:条件 brain 的上下文带
   `slots_by_strategy`(菜单里每个策略一份,选哪个用哪份,照抄 ID);
   润色 agent 的 conditions 媒体行即清单投影({slot, referenceable,
   description})。skill 改写:"NUMBERING IS GIVEN, NEVER GUESSED"。
3. **出口闸**(`pipeline/ref_slots.validate_references`,确定性正则,
   不靠 LLM 自觉):
   - 引用清单外编号 → 整条 prompt 作废 → 内容感知兜底模板顶上,
     **错编号永远到不了 API**(decisions 记 ref_validate 留痕);
   - 可引用槽位漏提 → 自动补一句("@Image2 shows: … — keep it
     consistent"),素材不白传;
   - 大小写不敏感,同时覆盖 @ImageN/@VideoN 与 kling "reference image N"。
4. **enhancer 重试一次**:输出引用了错编号 → 带着"错在哪、只许用哪些 ID"
   的反馈重试一次;仍错 → None(保留原 prompt,主循环闸再兜一层)。
5. **顺带修正**:t2v 策略的条件清单现在为空(旧行为会向润色 agent 谎称
   有图 —— t2v 根本不装配图)。

**测试**:`tests/unit/test_slot_manifest.py` 4 个 —— 清单与装配顺序一致
(tiv2v/ti2v_prev_plus/multi_fusion/i2v/t2v)、闸门三行为(放行/拦截/补
漏)、enhancer 重试一次、主循环端到端(brain 写 @Image1 于无引用通道的
i2v 路线 → 拦截 → 兜底,错编号未到 API)。全套 429 通过。

---

## 追加(2026-07-16 第二轮,已实现):剧本提及素材 → enhancer 格式化编号

**用户方案**:剧本(scene_write)必须把"用户点名的素材出现"写进 shot
description(但不写编号 —— 那时编号还不存在);enhancer 拿 description +
槽位清单,把自然语言提及翻译成正确的 @ 引用。

**采纳 + 三修正**:
1. 修正 A(强制力):skill 只能教 —— 加确定性警告:有素材但全剧本无一
   description 提及任一素材关键词 → log.warning "wasting the assets"
   (不阻断,终端当场可见)。`_write_outline` 内实现。
2. 修正 B(匹配锚点):槽位清单给 asset_image 来源的图加
   `"user asset: "` 前缀 —— enhancer 翻译"提供的图片中的猫"时一眼锁定
   哪个槽位是用户的东西(`_slot_manifest._c`)。
3. 修正 C(职责叠加不转移):enhancer 是可选开关,翻译能力不独占 ——
   条件 brain 仍按方案 A 持清单写引用(window_generation 规则 3 补:
   用户点名素材的槽位绝对不可丢,description 提及绑定到该 slot ID);
   enhancer 开着时做最终规范化。素材类任务建议常开 enhancer。

**skill 新法则**:scene_write "ASSET MENTION LAW"(点名素材必须用与素材
目录一致的措辞写进 description,不写编号);prompt_enhancer "FORMALIZE
ASSET MENTIONS"(逐条提及 → 匹配 "user asset:" 槽位 → 照抄 slot ID 重写;
清单没有对应槽位 → 保留文字描述,绝不发明编号 —— 发明会被闸门整条拒掉,
且这本身是上游图计划/策略丢素材的信号)。

**测试**(test_slot_manifest.py +3):清单 user-asset 前缀、剧本浪费警告
(提及则安静)、三个 skill 新法则在场防丢。全套 432 通过。
