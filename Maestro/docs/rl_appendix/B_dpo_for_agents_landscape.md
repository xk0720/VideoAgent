# 附录 B:DPO 族偏好优化 for agent tool calling 全景(调研代理原文,2026-07-18)

> 主报告 `../RL_TOOLCALLING_RESEARCH_2026_07_18.md` 第五节的完整依据。

## 1. 变体族谱与选型

- **DPO**(2023):成对数据 + 冻结参考模型,拉大 chosen/rejected 对数
  似然差;梯度实际主要在压 rejected;须先 SFT 温启动。
  https://huggingface.co/papers/2305.18290
- **IPO**:sigmoid 换平方回归目标,有界 —— 偏好近确定(我们的确定性
  闸门)时防过拟合。https://huggingface.co/papers/2310.12036
- **KTO**(2024,**对我们最重要**):不要成对!每条 (prompt, completion,
  好/坏) 即可;前景理论非对称损失;1B-30B 匹敌或超 DPO;类不平衡用
  desirable/undesirable_weight(目标 1:1~4:3)。生产日志二值结局的
  设计场景。https://huggingface.co/papers/2402.01306
- **ORPO/SimPO**:免参考模型但无 KL 锚 —— 窄域适配器有坍缩风险,跳过。
- **cDPO/rDPO**:标签噪声(VLM 判官对约 60-70% 准)→ label_smoothing
  ≈0.1 / 无偏损失 loss_type="robust"。https://arxiv.org/abs/2403.00409
- **RPO**(NVIDIA):有量化分差时用。RainbowPO(ICLR'25)结论:变体
  选择不如数据质量与 on-policy 程度重要。

## 2. 步/段级信用分配

- **Step-DPO**:共享正确前缀、在首个错误步分叉的步级对;~10K 对即大
  幅提升;响应级 DPO 在长链上定位不了错步。
  https://arxiv.org/abs/2406.18629
- **DMPO**:多轮下 DPO 数学上次优(配分函数不再消掉),占用测度约束
  + 轨迹长度归一。https://arxiv.org/abs/2406.14868
- **SDPO(段级,最可迁移)**:定位错轮 → 从错前历史重采样得正会话 →
  取首个分歧轮起的最小关键段成对(共享前缀)。段 ≈ 我们一个 shot 的
  修复序列。https://arxiv.org/abs/2501.01821
- 只有轨迹级结局时造步级对:Math-Shepherd 式 MC 补全(每步 N 条续写
  按成功率打分)/ MCTS 节点值(Agent Q/SVPO)/ DPO 隐式奖励免费当
  过程奖励(https://arxiv.org/html/2412.01981v1)。

## 3. agent 专用偏好优化先例

- **ETO**(ACL'24):SFT → 探索收集失败 → 同任务失败 vs 专家成功成对
  → 轨迹级 DPO → 迭代;ScienceWorld OOD +22% over SFT。
  https://arxiv.org/abs/2403.02502
- **IPR**(EMNLP'24):步级版 ETO,MC 步奖励 + 与专家步对照出动作级对。
- **Agent Q**:MCTS + 自评 + off-policy DPO;Llama-3-70B 订票成功率
  18.6%→81.7%(一天自主采集)→95.4%。树搜索偏好数据无人标可行。
  https://arxiv.org/abs/2408.07199
- **DiaTool-DPO**(SIGDIAL'25,工具调用对话控制):5 状态 MDP 自动
  构造对(过早调用 vs 正确追问槽位);追问 44→94.8%、拒调 9.6→91%。
  https://arxiv.org/abs/2504.02882
- **LoopTool**(2025):探测能力 → 判官复核标签(去噪内置)→ 围绕
  失败合成难样本 → 重训迭代;8B 超过给它造数据的 32B。
  https://arxiv.org/abs/2511.09148
- **TL-Training**(EMNLP'25):工具错误集中在少数类别(错工具名/错
  参数值/格式),关键 token 加权 + 错误类别奖励,**1217 条**追平闭源。
  https://arxiv.org/abs/2412.15495
- **PLUM**(确定性校验器→偏好对的最近先例):测例执行 pass/fail 当
  标签;**KTO 常胜 DPO**(DPO 压 chosen 似然的失稳);on-policy 数据
  至关重要。https://arxiv.org/html/2406.06887v1

## 4. 迭代式 DPO 与失效模式

- 循环:当前策略采 K 条 → 判官排序 → 成对 → DPO → 换新策略重采
  (Iterative RPO 等);第一轮收益最大,2-3 轮见顶。
- 失效:离线对随策略漂移失效(on-policy 重挖是解);DPO 隐式奖励 OOD
  泛化差;**chosen 似然坍缩**(margin 涨但双双下降)→ 加 NLL 辅助项
  (loss_type=["sigmoid","sft"],GSM8K 73.1 vs 61.8 的决定性差);
  参考模型陈旧 → 每轮重锚(TR-DPO sync_ref_model)。
- 数据量级(7-14B):~1.9K 干净 on-policy 对 → +5%(8B QLoRA);
  Step-DPO 10K;泛化对齐 ~60K;~20K 饱和;工具专项 1.2K 见效。
  经验律:窄技能低千级干净对即可动,宽行为要数万。

## 5. 从日志挖对的配方

- 同上下文成败对(ETO/LoopTool/PLUM):共享最长前缀、在翻转结局的
  决策处分叉的对最强。
- Best-of-N + 判官(West-of-N):best-vs-worst 最大化 margin;分差
  接近的对过滤掉。
- **verifier accept/reject 修复对 = 近零噪声、天然同上下文**(PLUM/
  SPORT 模式)—— 我们的盲测 A/B 台账免费产出。
- 判官标签走 label 复核(LoopTool JGLV)+ label_smoothing。

## 6. TRL + LoRA + Qwen3 实操

- DPOTrainer:beta=0.1,LoRA 时 lr≈1e-5;PEFT 下**无需单独参考模型**
  (禁用 adapter 的底座即参考),precompute_ref_log_probs 省显存;
  **原生工具调用数据格式**(messages+tool_calls+tools 列)。
- KTOTrainer:{prompt, completion, label};per-device batch ≥4 顺序
  采样(批内估 KL);lr 严格 5e-7~5e-6;类权重配平。
- 实测配方(Schmid'25):8B QLoRA 4bit 单张 24G,r16/alpha16/全线性
  层,~1.9K 对,+5%。14B LoRA ≈35-48G。
- Qwen3:思考/非思考模式统一在模板;**按部署模式训练**(决策头用非
  思考模式,序列短、评估便宜)。https://arxiv.org/abs/2505.09388
- 评测:BFCL v3/v4(AST 校验选择/参数,防幻觉函数)、τ-bench pass^k
  (重复试验一致性)、held-out 日志重放(我们的真指标)。

## 7. 代理的综合建议(原文结论,含信号→目标映射表)

| 日志信号 | 结构 | 噪声 | 最适目标 |
|---|---|---|---|
| 确定性闸门(JSON/菜单/槽位) | 不成对二值,逐决策 | ~0 | **KTO** |
| 盲测 A/B 修复判决 | 天然成对同上下文 | 低 | **DPO(sigmoid+NLL)** |
| VLM 缺陷报告 | 成对或二值,判官产 | 中 | **DPO robust + smoothing 0.1** |
| shot 收敛 | 延迟轨迹级 | 低但归因难 | **SDPO 式段级对** |

顺序:SFT 温启动(只用闸门通过+verifier 接受的决策)→ KTO 全量不成
对台账 → DPO 成对流(A/B 对 > 跨 run 成败对 > 未来 best-of-n)→
延迟信号用"首个分歧决策"定位段(我们菜单约束,diff 解析后的决策即
得,比论文用 MCTS 找便宜得多)→ 迭代重挖。评估:日志重放决策准确率
+ pass^k + BFCL 抽样回归 + 在线 A/B 胜率;训练中盯 logps/chosen 防
坍缩。跳过:ORPO/SimPO、初期 token 级方案;ArCHer 是 DPO 天花板后的
升级路径。