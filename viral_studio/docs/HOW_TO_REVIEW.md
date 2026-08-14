# 如何分析一次 `run_plan.py` 的产出

> 给人看的审片说明书。目标：在**没花一分钱生成费**之前，判断这份分镜脚本值不值得执行。
> 比喻：这是开机前的**剧本围读**——导演念一遍，制片算一遍钱，谁都别等拍完了才喊卡。

```bash
python tools/check_memory.py                              # 0. 记忆库体检(本地免费)
python run_plan.py --product examples/product_pink_tee.yaml   # 1. 出方案(约6次LLM调用)
```

产物在 `outputs/plan_<ts>/`，**按这个顺序看**：

| 顺序 | 文件 | 回答的问题 |
|---|---|---|
| 1 | `validation.json` | 机器挑得出的毛病还剩几个？(先看这个，省时间) |
| 2 | `creative_direction.json` | **创意对不对**：结构像不像爆款、卖点有没有落地 |
| 3 | `shot_script.json` | **执行靠不靠谱**：每段用什么模型、挂什么图、prompt 怎么写 |
| 4 | `tool_plan.json` | **值不值这个钱**：几次调用、多少计费秒 |

---

## 第一层：validation.json —— 机器已经替你查过的

`ok: true` 只代表"没有硬伤"，不代表"好"。真正要你读的是 `warnings`：

| 警告 | 含义 | 该不该管 |
|---|---|---|
| 总时长偏离目标超15% | 策划自己加戏了 | 看你要不要控时长；要就改 brief 的 `duration_target_s` 或直接删段 |
| prompt 开头未见相机指令 | 违反"相机先行"家规 | 建议管——不写相机，模型会自己发明运镜 |
| 资产 animate 兼容性未实测 | 导演赌了一把 | 想省钱就打回改自创；想探路就放行，结果回填记忆卡 |
| animate 不接受文本 prompt | 正常，该字段只是台账 | 不用管 |

`errors` 不为空 = 修复轮没收敛，**这份脚本不能执行**，直接看是哪段出问题（错误前缀 `[seg03]`）。

---

## 第二层：creative_direction.json —— 审创意

逐段问四个问题：

1. **结构像爆款吗？** 每段的 `reason` 必须引用记忆库证据（视频卡结构谱 / `asset_ref` / `pattern_ref`）。
   出现"我觉得这样比较好"这类无证据表述 = 策划在编，打回。
2. **hook 段撑得住吗？** 前 3 秒决定生死。多张人物图却没用 `multi_person_reveal`、
   或强节奏商品却没用 `beat_pose_swap`，都值得质疑。
3. **卖点落地了吗？** brief 里的 `selling_points` 应该能在段落 `idea` 里逐条找到出处。
   全片没提"三个配色"却卖三色卫衣 = 失职。
4. **节奏合理吗？** 段数 3–6、每段 2–8 秒整数；hook 短、talk 长、outro 干脆。

---

## 第三层：shot_script.json —— 审执行（最重要）

一段一段过，五个检查点：

### ① mode 选得对不对（省钱的关键）

| 看到 | 该确认的事 |
|---|---|
| `reuse_motion` | 对应资产卡的 `compat.animate_preflight` 必须是 `pass_verified`；`bgm_source` 必须是 `asset_bgm`（借动作必带 BGM）；`speech_text` 必须为空 |
| `self_create` | 为什么不能借素材？`decision_reason` 应给出理由（资产被拒/未实测/场景不匹配） |
| `self_create_multiwindow` | 一次调用管多个时段——只在"总长≤12s 且参考≤4张"时才划算 |
| `vo` | 台词必须同时出现在 `speech_text` 和 `prompt` 里；`bgm_source` 必须 `none` |

**经验判据**：`reuse_motion` 段越多越省钱且越稳（动作是真实拍摄的）；全片一个 reuse 都没有，
通常说明策划挑的资产都不兼容——回头看是不是记忆库该补正例了。

### ② 参考图挂对没有

`person_hook_refs` + `product_image_refs` 的**顺序就是 @Image 编号**。自己数一遍：
prompt 里写 `@Image3`，这两个列表加起来就得有 3 项。（程序会自动补挂/钳制，但钳制过的
prompt 语义可能歪掉——日志里出现"@Image 编号钳制"就重点看这段。）

### ③ prompt 质量（照这五条挑刺）

1. 第一句是相机指令（`Locked-off static camera, vertical 9:16 ...`）
2. 无高危动作词：`360 / full turn / spin / rotate quickly`
3. **所有时长/时间戳是整数秒**（`3 seconds`、`Shot 1 (1s)`、`0-2s`），不得出现 `1.1s`
4. 多人/多时段有防串脸约束（`Each woman keeps her own face ... No morphing`）
5. 结尾有约束尾巴（`no text, no extra people, no camera movement`）

### ④ 整数铁律的连带后果（容易被忽略）

自创路线最快只能 **1 秒一切**。所以看到"快切""卡点""0.7s 一镜"这类创意落在 `self_create` 上时，
要么接受它变成 1 秒一切（节奏变慢），要么改成 `reuse_motion` 借真实素材——
**亚秒级碎切只能靠借素材实现**，这是当前的物理上限。

### ⑤ 时长与素材的匹配

`reuse_motion` 段时长不能超过驱动素材本身（校验器会拦），但也别浪费：
素材 6 秒只用 2 秒，不如换个短素材或者干脆用满。

---

## 第四层：tool_plan.json —— 审成本

```bash
python -c "import json;p=json.load(open('outputs/plan_<ts>/tool_plan.json'));\
print(sum(i['billed_estimate_s'] for i in p),'计费秒 /',len(p),'次调用')"
```

- **计费秒** ≈ 生成时长之和（自创段不足 4 秒也按 4 秒生成后剪回，这部分是必然浪费）
- 经验值：20 秒成片 ≈ 25 计费秒、5–8 次调用、墙钟 15–30 分钟（2 并发）
- 明显异常：某段 `billed_estimate_s` 远大于 `duration_s`（说明短段太多，考虑合并成 multiwindow）

---

## 审完之后：结论怎么落地

| 你的结论 | 该改哪里 |
|---|---|
| 创意方向不对 | 改 `studio/prompts/planner.md`（策划的判断标准） |
| 执行决策不对（选错模式/挂错图/prompt 差） | 改 `studio/prompts/director.md` |
| 某类错误反复出现 | 加进 `studio/validate.py` 变成硬规则——**别指望靠提示词管住，机械问题用程序兜底** |
| 缺好素材可借 | 往 `memory/assets/cards/` 加卡，跑 `tools/check_memory.py` 验一遍 |
| 实测出了新结论（某素材能过/被拒） | 回填对应资产卡的 `compat.api_verdict`，这是记忆库越用越准的机制 |

---

## 一句话总结

**看 validation 省时间，看 creative 判创意，看 shot_script 判执行，看 tool_plan 判成本。**
其中最该较真的是 `mode` 选择和 prompt 的五条纪律——前者决定花多少钱，后者决定生成出来能不能用。
