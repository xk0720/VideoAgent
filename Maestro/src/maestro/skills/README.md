# skills/ — 每个 agent 的预定义技能（NEWTON planner_skills 式）

每个 agent 的"操作手册"都是一个带 frontmatter（name / agent / description）的
markdown 文件，按角色分组。这与 memory/skill_library.py 的**学习型技能**是两回事：
学习型技能是运行时从收敛的修复里蒸馏出来、按缺陷签名检索的；这里是每个 agent
**开工前就有**的手写技能。

```
brain_skills/orchestrator.md          brain：完整工具目录 + 决策规则 + 严格 JSON 输出
reviewer_skills/semantic_critic.md    VLM 语义评审：维度 / 工具 / checklist 输出契约
reviewer_skills/physics_critic.md     VLM 物理观点评审：失效模式 / 工具 / verdict 契约
reviewer_skills/physics_measure.md    non-AI 测量链：定位→追踪→认证→定律→verdict
summarizer_skills/review_summarizer.md 整理员：合并/排序/冲突/进度规则 + LLM 润色指令
verifier_skills/verifier.md           闸门：单调硬规则 + 盲测边际确认（只否决）
```

## 加载机制（loader.py）

`load_skill_catalog()` 扫全部 `**/*.md`，剥 frontmatter，`body` 即可入 prompt。

| 技能 | 谁在消费 | 方式 |
|---|---|---|
| orchestrator | OrchestratorAgent | **每回合整体载入 prompt**（_load_skill_prompt） |
| review_summarizer | ReviewSummarizerAgent | **LLM 润色指令取自 body**（_polish） |
| semantic_critic / physics_critic | SemanticCritic / PhysicsCritic | 契约规范：真实 VLM 指令目前在 `models/mllm_backends.py`（assess_semantic / assess_physics），以本文件为准绳；下一步把指令文本迁到这里 |
| physics_measure | PhysicsConsistencyCritic | 契约规范：链路是纯代码（无 prompt），本文件是它的权威文档 + demo 入口 |
| verifier | VerifierAgent | 契约规范：规则是纯代码（无 prompt），本文件是权威文档 |

`src/maestro/prompts/` 里剩余的 *.txt 是 v0.1 占位（无人加载），后续迁并到这里。
