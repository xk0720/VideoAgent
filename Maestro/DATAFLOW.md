# Maestro 数据流与运行配置说明

配合图看。本文回答三件事：数据怎么从输入流到视频、为什么现在看不到真实视频、服务器上怎么配怎么跑。

---

## 1. 端到端数据流（带类型与函数）

入口：`scripts/run_pipeline.py` → `maestro.pipeline.run.run_maestro()`。一次调用顺序经过 4 个 stage。

### 输入
- `--prompt`（必填）：自然语言指令，`str`。
- `--source`（可选，可多个）：源视频路径，做 grounding 素材。
- `--image`（可选，可多个）：参考图，提供身份/风格 anchor。
- `--music`（可选）：音乐，驱动镜头节奏与结构。

### Stage 0 · 素材理解 → `AssetMemory`
`pipeline/understand.py: build_asset_memory()`。把素材转成结构化记忆 `AssetMemory`：
`video_shots`（源视频拆 shot）、`identity_anchors`（人物/物体身份）、`style_anchors`（风格）、`music_profile`（bpm/beats/段落）。
> v0.1 是 **mock 感知**（按文件名造占位条目，不真读像素）。v0.2 在此换成 CLIP / 镜头检测 / InsightFace / all-in-one。离线可缓存。

### Stage 1 · 多 agent 规划 → `ShotSpec[]`
`pipeline/plan.py: plan_shots()`，三个 agent 串行：
1. `ScreenwriterAgent` → 分镜大纲 `outline: list[str]`（镜头数默认跟随音乐段落数）。
2. `DirectorAgent` → 每镜头一份 `ShotSpec`（prompt、镜头语言、引用哪些 identity/style anchor、按 beat 的 `rhythmic_pacing`）。**同时检索 `LessonLibrary` 把历史经验作为约束注入**（C4）。
3. `PhysicsPlannerAgent` → 给每个 `ShotSpec` 挂 `PhysicsAnnotation`（v0.4）：从 prompt 抽实体 + **运动类别**（ballistic/rigid/fluid/agentive/static）+ 交互 + 预期失效模式。只是**验证种子**——不含轨迹、不含控制信号（sketch-as-controller 线已废弃）。

### Stage 2 · 生成 + 自改进闭环（核心）→ `accepted CandidateClip[]`
`pipeline/generate_loop.py: generate_shot()`，**逐 shot** 跑闭环：
1. `GeneratorAgent` 在 first_frame（关键帧锚）/ 参考图条件下生成 `n_candidates` 个候选 `CandidateClip`；物理从不注入，只在生成后从像素验证（C6 v0.4）。
2. **Tournament 选优**取最强候选（E3）。
3. `ReviewBoard`（`critics/board.py`）并行跑 4 个 critic：
   - `SemanticCritic`（prompt 对齐 checklist）
   - `PhysicsCritic`（**按失败模式定位到帧** → `PhysicsVerdict`，C1 critic 层）
   - `ConsistencyCritic`（身份/风格一致性）
   - `RhythmCritic`（卡点/能量）
   再由 `MetricTool` 算出 metric 套件（`m1_semantic / m2_temporal / p1_physics / id1_identity / m5_rhythm / aesthetic / weighted_total`）。
4. `VerifierAgent` 用 `weighted_total` + 失败项数判断**是否严格变好**；只有变好才 accept（**单调改进**，C2）。
5. 不达标 → `RefinerAgent` 把"哪条定律第几帧违反"翻译成**可执行修正**：对失败关键帧做 `image_edit` 局部编辑，再以其为 first_frame 局部 regen（**不是整段重生成**）。`k_retries` 次仍不过 → escape hatch 跳过该项防死循环。
6. 收敛后把"失败+成功修法"蒸馏成 `Lesson` 写进 `LessonLibrary`（C4），供未来任务检索。

每个 shot 返回 `SelfImproveResult`（含 `score_history`、`revisions_used`、`converged`、`gen_calls`）。

### Stage 3 · 拼接 → 最终输出
`pipeline/assemble.py` → `tools/assembly_tool.py`。把 accepted 片段（+音乐）用 ffmpeg 拼成 `demo.mp4`。
> **降级逻辑**：当片段是 v0.1 的 mock 文件、或环境无 ffmpeg 时，自动写成一个 **manifest 占位 `.mp4`**（文本）+ `.manifest.json`，保证流程总能产出路径、测试能过。

### 产物
- `demo.mp4` —— **v0.1 是 mock 占位**（不是真视频）。
- `demo.report.json` —— 每镜头修正轮数、是否收敛、生成调用数、score 曲线、最终 metric。
- `demo.trajectory.jsonl` —— 每个 agent 的决策（state/action/observation），为未来 RL/reward 预留。
- `lessons.jsonl` —— 跨任务经验记忆。

---

## 2. 为什么现在看不到真实视频（重要）

v0.1 是**脚手架**，目标是把"多 agent 规划 + 自改进闭环 + 物理模块"的**控制流和数据流**先跑通、可测试、CPU 可运行。因此**所有重模型都是 mock**：

| 环节 | v0.1 (mock) | 产出 |
|---|---|---|
| 感知/理解 | 按文件名造占位 | 占位 AssetMemory |
| LLM 规划 | `MockLLMClient` 回 ack；agent 用确定性 Python 算结构 | 真实结构的 ShotSpec |
| VLM critic | `MockMLLMClient` 确定性打分（随 revision 收敛） | 真实的 verdict/metric 流 |
| **视频生成** | `MockVideoGenClient` 写一个文本占位 `.mp4` | **占位文件，非像素** |
| 关键帧编辑 | `MockImageEditClient` 写占位 | 占位 |
| 物理仿真 | `MockSimulator` 解析重力轨迹 | 真实轨迹 JSON（可用） |

所以闭环、报告、日志、经验库都是**真实有效**的；唯独"像素"是占位。要看到真实视频，必须进入 **v0.2**：把 `MockVideoGenClient` 换成真实视频生成模型（接口已留好，见下）。

---

## 3. 输入配置详解（`configs/default.yaml`）

```yaml
models:                      # v0.1 全 mock；v0.2 在这里改 name/backend
  llm:        {name: mock-llm}        # 规划大脑 (DeepSeek/GPT/Claude/vLLM)
  mllm:       {name: mock-mllm}       # judge/critic (Qwen-VL 等)
  video_gen:  {name: mock-video-gen}  # 视频生成 (OmniWeaving/Wan/Veo)
  image_edit: {name: mock-image-edit} # 关键帧编辑 (Qwen-Image-Edit)

plan:
  n_shots: 3            # 默认镜头数（有音乐时跟随音乐段落数）
  max_shots: 6          # 镜头数上限
  shot_duration: 3.0    # 每镜头秒数

compose:
  fps: 8                # 帧率（mock 仿真/生成用）
  n_candidates: 2       # 每镜头初始候选数（tournament 池）
  max_revisions: 5      # 自改进最大轮数（收敛上限）
  k_retries: 2          # 每轮局部修正重试次数，超出触发 escape hatch

metrics:
  weights:              # 驱动 Verifier 单调判断的加权；物理 p1 权重最高之一
    m1_semantic: 0.25
    m2_temporal: 0.15
    p1_physics: 0.25
    id1_identity: 0.15
    m5_rhythm: 0.10
    aesthetic: 0.10

physics:
  simulator: mock              # v0.2: mujoco / newton / 粒子仿真
  acceptance_severity: 0.30    # 物理失败 severity 低于此即视为已解决
```

调参直觉：
- 想**更追求质量**→ 调高 `max_revisions`、`n_candidates`（更慢更贵）。
- 想**更强物理**→ 调高 `metrics.weights.p1_physics`、调低 `physics.acceptance_severity`。
- 镜头多/长 → 调 `plan.n_shots`、`shot_duration`。
- 所有值都可被 `--config 你的.yaml` 覆盖，或代码里 `load_config(overrides=...)`。

---

## 4. 服务器上怎么跑

### A. 先验证 v0.1（CPU，几秒，确认环境）
```bash
cd Maestro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy pyyaml pytest
pytest -q                              # 应全绿

python scripts/run_pipeline.py \
  --prompt "a ball is thrown and bounces; a person runs through a city" \
  --music data/track.mp3 --image data/hero.png \
  --output outputs/demo.mp4
# 看 outputs/demo.report.json 与 outputs/demo.trajectory.jsonl 验证闭环
```
（此步 `demo.mp4` 仍是占位，目的是确认流程/闭环正确。）

### B. 接真实视频生成（v0.2，出真实像素）
最小改动路径——只换"出像素"的一环即可先看到真实视频：
1. 在 `src/maestro/models/video_gen.py` 新增一个 `BaseVideoGenClient` 子类（已有 `WaveSpeedClient` 完整实现 / `OmniWeavingClient` / `WanClient` 骨架），实现 `generate(prompt, duration, out_path, first_frame, reference_images, seed)`。纯文本 API 后端也可以——物理是验证出来的，不是注入的（v0.4）。
2. 在 `build_video_gen()` 工厂里按 `name` 分发到你的新类。
3. `configs/default.yaml` 把 `models.video_gen.name` 改成你的后端，并在 `.env` 填 key/endpoint。
4. ffmpeg 装上，`AssemblyTool` 会自动走真实拼接（不再降级 manifest）。

推荐先接的顺序（性价比）：先 `video_gen`（看到像素）→ 再 `mllm`（让 PhysicsCritic 用真实 VLM 真正定位物理错误）→ 再 `llm`（让规划更聪明）→ 最后真实 `physics simulator`。

### C. 需要配置的 API（提醒）
- v0.1：**不需要任何 key**。
- v0.2：按 `.env.example` 配 —— LLM（DeepSeek/GPT/Claude）、VLM（Qwen-VL）、视频生成（OmniWeaving/Wan 自部署 或 Veo/Sora API key）。物理仿真（MuJoCo/Newton）本地装、无需 key。
- GPU：真实视频生成/本地 VLM 需要 GPU；建议先用一张卡跑通单 shot，再放开 `n_candidates`/`max_revisions`。

---

## 5. 一句话总结
v0.1 已经把"规划→生成→多agent评审→关键帧局部精修→单调改进→经验沉淀→拼接"的**数据流和自改进闭环**完整跑通且可测；现在的 `.mp4` 是 mock 占位。要真实视频，只需在 `video_gen` 这一环替换成真实模型（接口已就绪），其余数据流不变。
