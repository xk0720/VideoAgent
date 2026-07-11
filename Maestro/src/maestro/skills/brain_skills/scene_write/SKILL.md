---
name: scene_write
agent: ScreenwriterAgent + DirectorAgent (playwriting, 窗口大循环 §A)
description: 用户 prompt → 按时间顺序的 scene/shot 文本描述列表 → ShotSpec;是 StoryboardMemory 台账的种子。
---

# Scene Write(playwriting)— 剧本拆解技能

## 角色
把用户的一句话/一段话拆成【按时间顺序】的 shot 描述列表。这个列表就是
窗口式生成的骨架:每一行变成台账(StoryboardMemory)的一个条目,后续的
keyframe 策略、条件策略、评审修复都挂在这些条目上。

## 拆解规则
1. 每个 shot 一句完整、可拍的描述:主体 + 动作 + 场景(+ 镜头语言可选)。
   坏例:"然后它掉下来"(主体是谁?在哪?);
   好例:"Shot 2: the glass tips over the table edge and falls (kitchen, close-up)"。
2. 分场:场景/地点/时间跳变 = 新 scene。描述里显式写 "scene N",台账靠它
   解析场号(解析不到全归 scene 1 —— 单场剧本这是对的)。
3. shot 数量:默认跟配置(plan.n_shots,默认 3;上限 max_shots);素材库
   有音乐 profile 时按乐段数。宁少勿碎——每镜 4-15 秒(生成模型的时长域)。
4. 时间顺序即生成顺序:窗口循环严格按列表顺序推进,跨镜连续性(上镜尾帧/
   尾段当锚)只对相邻 shot 成立——把需要连续的动作放进相邻的 shot。
5. 实体一致性:同一角色/物体在所有 shot 描述里用同一个名字(检测、追踪、
   素材检索都按名字对齐)。

## 输出去向
outline(每 shot 一行)→ DirectorAgent 逐行展开成 ShotSpec(时长、镜头
语言、identity/style 引用、物理标注、事件图)→ StoryboardMemory.from_outline
建台账(全 pending)。

## 当前实现状态(诚实声明)
ScreenwriterAgent 目前是确定性拆条(按分号切子句、按 n_shots 循环),LLM
调用是占位;本技能文件是它升级成真 LLM playwriting 时的操作手册,规则
以此为准。
