# Maestro 增量改进日志（借鉴现有方法 × 突出创新点）

本轮在 v0.1 骨架上补了 4 个有明确文献依据、又强化差异化的模块。全部**向后兼容**（默认行为不变，已通过的测试仍应绿；新增模块各带单测）。

| # | 新模块 | 借鉴（引用） | 我们的创新 / 差异化 |
|---|---|---|---|
| 1 | **GEST 事件图 IR** `planning/event_graph.py` | Event-Graph (arXiv:2604.10383)：GEST 表示 + separation-of-concerns + 可验证构造 + Relation Subagents | 事件图不交给 3D 引擎渲染（那会丢真实感），而是作为**约束/grounding 层**：驱动物理草图 + 播种语义 checklist，像素仍由神经生成器出 |
| 2 | **规划 Validate→Correct 闭环** `agents/plan_validator.py` | FilmAgent 的 Critique-Correct-Verify；UniVA (arXiv:2511.08521) 自反思；旧仓库 Orchestrator | 在**生成之前**就校验每个 ShotSpec 的可 grounding 性（引用是否存在、事件图是否合法），新增**规划级**自改进闭环，远比生成后才发现问题便宜 |
| 3 | **双向消偏锦标赛** `critics/tournament.py` | VISTA (arXiv:2510.15831) 的 Binary Tournament + 双向比较消除 MLLM 位置偏置 | 替代 ViMax"生成多张 VLM 挑一张"的单次多选，作为更鲁棒的候选选优，给自改进闭环播种 |
| 4 | **资产检索 grounding** `tools/retrieval_tool.py` | ViMax asset indexing；DIRECT 的 CLIP 检索；旧仓库 narrative memory | 把"检索"从 *editing* 移植到 *generation*：Generator 拉取身份/风格 anchor 图像做条件，保证跨镜头一致性（落地 E1） |

## 接入位置（数据流变化）
- **Stage 1**：`Director` 现在给每个 `ShotSpec` 挂 `event_graph`（GEST）；`plan_shots` 增加 `PlanValidatorAgent` 的 Validate→Correct 循环（`max_plan_iters`，默认 3）。
- **Stage 2**：`generate_shot` 现在用 `Tournament`（双向消偏）做候选选优；用 `RetrievalTool` 取身份/风格 anchor 作为生成的 `reference_images`（初始与每轮局部修正都带）。
- 配置新增 `plan.max_plan_iters`；`GeneratorAgent.run` 增加 `reference_images` 参数。

## 新增测试
- `tests/unit/test_planning.py`：事件图构造/校验（温度序、缺 actor、悬挂边）、PlanValidator 标记缺失引用、Director.revise 纠正。
- `tests/unit/test_tournament_retrieval.py`：锦标赛选最优/单候选、身份/风格检索、源镜头语义检索。
- `tests/integration/test_end_to_end.py`：新增断言 trajectory 出现 `validate_plan`。

## ⚠️ 测试状态
本会话隔离沙箱多次未能启动（`ERR_NETWORK_CHANGED` / `ERR_CONNECTION_RESET`），**新增测试未由我执行**。已做多轮静态自查（导入无环、签名/默认值向后兼容、事件图 verb 词形匹配 bug 已修）。请在服务器 `pytest -q` 复跑；若有红贴报错即修。

## 与四大核心创新的关系
- 这 4 项都不改变 C1–C4 的主张，而是**加固论证链**：事件图让物理草图更有据（C1），规划级闭环 + 锦标赛让"自改进"更完整（C3），检索让多模态 grounding 落地（E1）。每个文件头部都分列了"借鉴谁 / 我们创新在哪"，便于写论文时引用。
