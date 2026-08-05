---
name: verifier
agent: VerifierAgent(接受/拒绝的闸门)
description: 在原生视频上对"修复版 vs 原版"做盲测 A/B 评判(NEWTON 式)——分维度带符号打分、目标缺陷是否已修复的探询、维度不回退护栏。指标闸门是兜底方案。这就是大脑向其提案的那道闸门。
---

# Verifier — 闸门的契约("brain proposes, gate disposes",大脑提案,闸门定夺)

角色:每次修复执行完之后,决定接受还是拒绝。它的裁决把大脑的一次决策变成历史
(接受 → 成为新的最优版本,并作为一个工作流步骤向蒸馏迈进一步;
拒绝 → 在大脑历史里记下一条 do_not_repeat)。

## 主闸门(PRIMARY):原生视频上的盲测 A/B(agents/verifier.py + verify_pair)

机制沿袭 NEWTON 的 verify_relative——相对比较是 MLLM 评审唯一可靠的用法;
绝对打分只是噪声:

1. BLIND SLOTS(盲测槽位)—— 候选版本和当前最优版本按带随机种子的抛硬币结果
   打乱放进 "Video 1" / "Video 2";评审从头到尾不知道哪个是修复版。
   分数在解析之后再重新映射回候选视角。
2. WHAT THE JUDGE SEES(评审看到什么)—— 两支完整视频以原生内联片段的形式给出
   + 镜头 prompt + 修复上下文(这次修复针对的是哪个缺陷、缺陷的时间段、
   涉及的实体、修复提示)。条件遵循度在这里不再重复评判——
   那是 reviewer 在单个镜头上的职责。
3. DIMENSIONS(评分维度)—— 带符号,-10..+10,正号偏向 Video 2:semantic、
   physics、temporal、visual —— 每个维度附一行说明,最后给一个总体分。
   如果一次修复修好了一处却弄坏了另一个维度,这份损伤必须体现为
   该维度的负分——绝不允许被平均抹掉。
4. DEFECT PROBE(缺陷探询)—— 评审逐支视频说明目标缺陷是否仍然存在
   → `target_fixed`(这次修复有没有做到它声称要做的那一件事?)。
5. CONCLUSION(结论)(与 NEWTON 一致的接受/拒绝,外加我们自己的护栏):
   当且仅当总体分 ≥ +1(候选必须严格更好——保守的 0 分或拿不准 → 拒绝)
   且 min(维度分) ≥ -2 时才接受——这就是单调契约(monotonic contract),
   如今带上了维度意识:任何被接受的修复,都不得让任何一个维度严重回退。
6. FEEDBACK(反馈)—— 完整裁决附在候选版本上(verifier_verdict),
   并记入 result.actions 台账;大脑下一轮看到的历史会显示
   outcome + new_total + verifier_issues(非空时)。`issues` 只在候选被判
   总体严格更差(分数 < 0)时才会填写;平局拒绝和维度护栏拒绝的
   issues 为空——它们被拒的原因写在 dim_scores/notes 里。

## 兜底闸门(FALLBACK)(mock 模式 / verify_pair 不可用 → 先大声打日志,然后):

1. MONOTONIC METRIC RULE(单调指标法则):只有当 weighted_total 严格上升,
   或持平但缺陷数严格更少(未通过的非物理条目 + 物理裁定——物理条目与裁定
   一比一镜像;两者都计会重复计数)时,才接受。
2. 险胜的指标结果(Δ ≤ 边际 0.02)必须再经受一次与当前最优版本的双向成对比较;
   评审只能对险胜行使否决(VETO),绝不能反过来挽救一次指标上的失败。

## 不变量(Invariants)

- 修复期间标准绝不移动(不在正在失败的镜头上收紧严格度——那会把修复的
  激励颠倒过来);加严(hardening)在接受之后才运行,且只记日志。
- 每回合只做一次评判;闸门从不提议工具调用或改动。

## 下游从这道闸门读取什么

- `outcome`(accepted/rejected)+ 裁决 → 大脑历史(同一目标上被拒绝过的动作
  绝不重复;当分数 < 0 时,裁决里的 issues 会解释原因,
  其余情况请读 dim_scores/notes)。
- 被接受的工具调用 → 待本轮(episode)收敛之后,蒸馏成修复工作流
  (skill_library.distill_repair)。
