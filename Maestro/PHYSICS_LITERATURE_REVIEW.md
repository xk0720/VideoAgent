# Physics Grounding 文献综述与 C1 重新定位

> 调研时间 2026-06。起因：Kevin 质疑"把物理草图/仿真轨迹当 condition 喂给冻结视频模型"是否有文献依据。
> 结论先行：**质疑成立。该做法没有任何一篇工作完整验证过，是空白而非成熟技术；证据指向的正确角色是"仿真当验证 oracle + world-model reward 驱动 test-time 搜索"。** 本文档是重构（v0.3 physics 模块）的依据。
>
> 日期说明：当前为 2026-06，arXiv 25xx/26xx 编号属合法近期论文；个别标注【核】的建议引用前点开二次确认。

---

## 1. 核心问题的直接答案

**Q：有没有人在冻结的通用视频模型上、用抽象仿真/草图轨迹做 condition、training-free 地证明物理变好？**

**A：没有一篇同时满足全部条件。** 最接近的工作各缺一角：

| 工作 | 满足 | 缺 |
|---|---|---|
| MotionCraft (2405.13557, NeurIPS'24) | 零训练 + 仿真光流注入冻结扩散 | 基于**图像**扩散+光流warp，非视频底模；物理证明偏定性 |
| PhyRPR (2601.09255) | training-free + 规划 scaffold 注入冻结 Wan2.2 | 评测规模小(40场景/12评委)；scaffold 是确定性规划非物理仿真 |
| SDG (2509.24702) | training-free + 冻结模型 + **硬指标**(PhyGenBench/VideoPhy 上提升 CogVideoX/Wan) | condition 是 **LLM 反事实引导**，根本不是仿真轨迹 |
| PSIVG (2603.06408, CVPR'26) | 最纯正 physics-engine-in-the-loop | 含 test-time 优化 + 4D重建+mesh，重管线非即插即用 |

**这正是一个清晰的空白点——但空白的原因是该路线有结构性硬伤（见§3），不是没人想到。**

## 2. "仿真信号当 condition"的真实格局：以 trained 为主

- **训练专门控制分支**：Motion Prompting (2412.02700, CVPR'25, 训 ControlNet)、Force Prompting (2505.19386, NeurIPS'25, Blender 仿真造数据微调力条件)、PhysAnimator (2501.16550, 草图条件需训练)。
- **逐物体 SDS 优化**：PhysDreamer (2404.13026)、Physics3D (2406.04338)、DreamPhysics (2406.01476)——方向还反了：用视频先验学物理参数。
- **重多模块管线**：PhysGen (2409.18964, ECCV'24, 仅刚体)、PhysGen3D (2503.20746)、PhysMotion (2411.17189)、WonderPlay (2505.18151)、PSIVG——都需要感知/重建/仿真模块，不是冻结底模即插即用。
- **轨迹/拖拽控制**（DragNUWA 2308.08089、MotionCtrl 2312.03641、Tora 2407.21705、DragAnything 2403.07420、Motion-I2V 2401.15977、Go-with-the-Flow 2501.08331）：**几乎全部需要训练专门分支，且训练轨迹来自真实视频的光流/点追踪**（SG-I2V 2411.04989 摘要明言）。training-free 的仅 SG-I2V/FreeTraj/DiTraj 少数，且 motion fidelity 自承弱于监督法。**对解析抛物线这类 OOD 合成轨迹的服从性：文献空白，无人测过。**

## 3. 为什么轨迹 conditioning 修不了物理（结构性硬伤）

1. **轨迹欠定物理（几何事实）**。VideoPhy (2406.03520) / VideoPhy-2 (2503.06800) 归纳的失败模式——穿模、形变、流体本构、**质量/动量守恒**（最弱项，hard 子集 joint 仅 22%）——全部活在质心轨迹的零空间里。一条 (x,y,t) 线不携带质量/体积/接触面/材料信息。
2. **"视觉真实 ≠ 物理理解"有大规模实证**。Physics-IQ (2501.09038, DeepMind)：Sora 视觉最真但 Physics-IQ 仅 10.0%，最佳模型 ~24%；二者**统计上不相关**。
3. **控制方法自承只控粗运动**：MotionCtrl 主动丢形状信息、DragAnything 自承限 2D 不能转身、Tora 承认稀疏-稠密鸿沟。
4. **业界共识是 densify**：要管接触/形变必须把稀疏轨迹稠密化成 mask/depth/flow（STANCE 2510.14588、VHOI、MagicMotion 2503.16421）或直接补物理参数（PhysCtrl 2509.20358 用仿真生成 55 万条物理正确轨迹**训练**）。

## 4. Training-free 提升物理：被验证有效的手段（按证据强度）

1. **World-model reward + 推理时搜索（最强）**：WMReward (2601.10553, Meta FAIR)——V-JEPA-2 物理先验当 reward，BoN+梯度引导搜索去噪轨迹，**ICCV 2025 PhysicsIQ Challenge 第一，62.64%（+7.42%）**，开源 github.com/facebookresearch/WMReward。关键发现：**world-model reward 显著优于 VLM-as-critic reward**。
2. **物理 reward + Best-of-N**：VJEPA-2 Reward (2510.21840)——16 候选选最低 surprise，PhysicsIQ 和 VideoPhy 各约 +6%。
3. **VLM/LLM 迭代自改进 prompt**：2511.20280 (MM-CoT)——PhysicsIQ 56.31→62.38，挑战赛第三。
4. **解耦推理 + scaffold 注入**：PhyRPR——有效但小规模。
5. **反事实引导**：SDG (2509.24702)——冻结模型上 PhyGenBench/VideoPhy 硬指标提升。
6. caveat：Seeking Physics in Diffusion Noise (2603.14294)【核】指出早期去噪步 reward 与最终物理相关性差——BoN/搜索要在合适的步做。

对照（号称物理但要训练）：PhysCorr/PhyDPO (2511.03997, 提升仅 ~2-3%)、NewtonRewards (2512.00425)、PIRF (2509.20570, 科学 PDE 非通用视频)、VChain (2510.05094, 需逐实例 test-time training)。

## 5. 仿真/轨迹当"验证 oracle"：有成形的研究线（我们的新 C1 落点）

三种范式，全部有现成工作：

1. **真值/仿真逐帧对比**：Physics-IQ（预测续帧 vs 真实拍摄，Spatial/Spatiotemporal IoU + MSE，全自动开源）；PISA (2503.09595, ICML'25)——自由落体，**Trajectory L2 / Chamfer / IoU** 对照含仿真的真值。
2. **抽取轨迹/物理量 → 对照守恒律**（最贴合 Maestro）：Morpheus (2504.02918)——物体追踪 + PINN 抽速度/加速度，按**能量/动量/加速度守恒**打分；NewtonBench (2512.00425) 把"光流当速度代理"做成可验证 reward。
3. **光流/点追踪当评测特征**：PhyCoBench (2502.05503)——预测光流 vs 生成光流的 deviation，与人评一致性最好；Direct Motion Models (2505.00209)——BootsTAPIR 点轨迹直接当评测特征。

自动物理评测选型：无参考大规模跑 → **PhyGenEval（三层VLM, 2410.05363）/ VideoPhy-2-AutoEval / VBench-2.0 (2503.21755)**；客观可作 reward → **Physics-IQ / Morpheus / PISA / PhyCoBench**。

现有 oracle 类工作的公开短板（= Maestro 的空间）：大多限于受约束/单物体场景；3D、多体接触、真实材料的真值稀缺；**没有人把"仿真 oracle + 失败模式定位 + agentic 修复闭环"串起来**。

## 6. C1 重新定位（v0.3 起生效）

**旧（撤销）**："物理草图 → control signal → condition 冻结生成器"（engine does physics, diffusion does rendering）。
**新**：**仿真是预言机不是控制器**——

```
PhysicsPlanner 仿真出"期望轨迹"(oracle reference)
        │
Generator 正常生成（conditioning 只用模型真听得懂的：
        │   I2V 首/末帧锚点 + 参考图；物理通过"摆对的关键帧"进入生成）
        ▼
TrajectoryOracle: 点追踪/光流(v0.3: CoTracker/RAFT)从生成视频抽"观测轨迹"
        → 期望 vs 观测偏差 + 守恒检查（PISA/Morpheus 式）
        ▼
PhysicsConsistencyCritic 出可定位 verdict ──┐
WorldReward (V-JEPA-2, 对标 WMReward) ──────┤→ test-time 搜索:
VLM PhysicsCritic (PhyGenEval 式) ──────────┘  BoN/tournament + 重写/局部regen
```

诚实版差异化主张：**"以物理仿真为 oracle、以 world-model reward 为评判、由多 agent 编排的 test-time 物理自改进闭环"**。每一块都有 SOTA 背书（PISA/Morpheus/WMReward/VISTA），**组合成 agentic 闭环 + 失败模式定位 + 跨任务记忆是我们的增量**。

对应代码变更：`physics/oracle.py`（新）、`critics/physics_consistency.py`（重写为 oracle 对比，去掉读 metadata 的代理逻辑）、`models/world_reward.py`（新，Mock→v0.3 V-JEPA-2）、`control_render.py` 降级为"oracle 参考 + 可选关键帧锚点提示"。

## 7. 引用清单

PhysGen 2409.18964 · PhysGen3D 2503.20746 · PhysMotion 2411.17189 · WonderPlay 2505.18151 · PSIVG 2603.06408 · PhysDreamer 2404.13026 · Physics3D 2406.04338 · DreamPhysics 2406.01476 · Motion Prompting 2412.02700 · Force Prompting 2505.19386 · PhysAnimator 2501.16550 · MotionCraft 2405.13557 · SDG 2509.24702 · PhyRPR 2601.09255 · DragNUWA 2308.08089 · MotionCtrl 2312.03641 · Motion-I2V 2401.15977 · DragAnything 2403.07420 · Tora 2407.21705 · SG-I2V 2411.04989 · Go-with-the-Flow 2501.08331 · FreeTraj 2406.16863 · DiTraj 2509.21839 · MagicMotion 2503.16421 · STANCE 2510.14588 · PhysCtrl 2509.20358 · VideoPhy 2406.03520 · VideoPhy-2 2503.06800 · PhyGenBench/Eval 2410.05363 · Physics-IQ 2501.09038 · VBench-2.0 2503.21755 · PISA 2503.09595 · Morpheus 2504.02918 · PhyCoBench 2502.05503 · Direct Motion Models 2505.00209 · WMReward 2601.10553 · VJEPA-2 Reward 2510.21840 · VLM-Self-Refine 2511.20280 · PhysCorr 2511.03997 · NewtonRewards 2512.00425 · PIRF 2509.20570 · VChain 2510.05094 · VDAWorld 2512.11061 · SVD 2311.15127 · Wan 2503.20314 · Framer 2410.18978 · VideoComposer 2306.02018 · SparseCtrl 2311.16933 · Ctrl-Adapter 2404.09967 · CameraCtrl 2404.02101
