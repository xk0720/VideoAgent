# viral_studio — 记忆库驱动的爆款带货视频工作室

> 输入「商品信息 + 人物/商品参考图」, 输出「带音乐的带货短视频」。
> 创作不是凭空来的: 多模态理解 agent 把爆款拆成**记忆库**(结构谱/动作资产/策略),
> 策划与导演基于记忆混编出新片。独立项目——不 import 本目录之外的任何代码, 需要就 copy。

## 四个 Agent(两条流水线)

```
[离线入库]  omni理解agent(qwen3.5-omni-plus, P3): 爆款视频 → memory/ 三层卡片
[在线创作]  商品brief ─→ 策划agent(创意方向) ─→ 导演agent(分镜脚本+安检门)
                         └─ 读 memory 摘要        └─ 读被引用卡片全文
            ─→ 执行层(P2=dry工具计划; P3=真调用+装配) ─→ 成片
```

## 记忆库(memory/)——三层卡片

| 层 | 谁读 | 内容 |
|---|---|---|
| `videos/` 视频总卡 | 策划 | 段落结构谱、BPM、strategy takeaways |
| `assets/cards/` 段级资产卡 | 导演 | 切片路径+**BGM切片**+英文动作描述+`compat`(animate实测兼容性)+适用对象 |
| `patterns/` 策略卡 | 两者 | 可复用编排套路 + 分镜式 prompt 模板 |

当前池: 3 条真实爆款 → 3 视频卡 + 10 资产卡(含 6 张真实 API 判决、2 张负例) + 3 策略卡。

## 快速开始(P2: 出分镜脚本, 不花视频生成费)

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY=sk-xxx        # 或写进 ./.env
python tools/check_memory.py           # 记忆库体检(本地免费)
python run_plan.py --product examples/product_pink_tee.yaml
# 产物: outputs/plan_<ts>/{creative_direction, shot_script, validation, tool_plan}.json
```

**产出怎么审 → [`docs/HOW_TO_REVIEW.md`](docs/HOW_TO_REVIEW.md)**(四层审片法: 先看
validation 省时间, 再看 creative 判创意、shot_script 判执行、tool_plan 判成本)。

## 硬规矩(全部来自真实调用的教训, 校验器强制)

- animate 只吃 `compat: pass_verified` 的资产; 实测被拒(NoHuman/FullFace)的一律自创。
- 借资产动作必须带走它的 BGM 切片; `reuse_motion` 段**不写台词**(口型跟随驱动素材,
  prompt 仅作台账); 口播用 `vo` 段, 台词写进 prompt 且不配 BGM。
- **整数铁律**: 生成时长与时间戳只能整数秒(窗口最小 1s) —— 连带后果: 亚秒级碎切
  只能靠 `reuse_motion` 借真实素材实现, 自创路线最快 1s 一切。
- prompt: 相机指令第一句; 禁 360/spin; `@ImageN` 编号=参考图传入顺序; 多人段写防串脸约束。
- seedance 时长域 4–15s 整数, 短段生成后剪回目标时长(帧级)。
- 一次调用多时段(multiwindow) vs 逐段调用: 导演按"总长≤12s 且参考≤4张"逐案决定。

## 路线图

- **P2(当前)**: 策划+导演+校验器+dry 工具计划 ✅
- **P3**: 执行层真调用(copy 进 bailian animate 客户端与 WaveSpeed seedance 客户端)+帧级装配+BGM铺回; omni 入库 agent
- **P4**: 口播音画同出、QC 重试梯子、批量与成本报表
- 待实验回填: seedance 时间控制(multiwindow 可靠性)、`untested` 资产的 animate 预检
