---
name: physics_measure
agent: PhysicsConsistencyCritic(非 AI 审阅者)
description: 基于实测的物理审查链 — 定位 → 追踪 → 认证 → 拟合运动定律 → 按实体、带帧区间的裁定(source 为 law_verifier)。无需参考、无需训练。
---

# 实测物理审查(非 AI)— 工具链及其契约

职责:只凭**像素**回答一个不带任何参数的问题——"每个实体观测到的运动,
是否存在**任何一种**物理上自洽的解释?"——而绝不是"它是否与某个仿真吻合"
(那会预设我们根本无从知晓的质量与尺度)。
这个审阅者是一次**测量**:在它的管辖领域(运动/存在/时序)内,
它的裁定优先级高于 VLM 的意见,并带有 `source="law_verifier"` 标签。

## 工具链(各阶段的工具,按顺序)

1. LOCATE(定位)— GroundingDINO(`models/detection_backends.py`)
   · 按名称对第 0 帧中每个已标注实体做零样本检测
   · 输出:归一化 bbox → 质心 = 追踪的种子点
   · 未检测到 → 用启发式种子点,且该裁定被标记为不可靠

2. TRACK(追踪)— CoTracker(`physics/track_extractor_backends.py`)
   · 种子点 [t=0, x_px, y_px] → 逐帧的点轨迹
   · 输出:归一化的屏幕空间轨迹 (x, y) ∈ [0,1],y 轴向下增长
     (重力表现为大小**未知**的 +y 方向加速度)

3. CERTIFY(认证)— 可靠性闸门(`physics/reliability.py`)
   · 追踪器在生成视频上会"说谎":轨迹震荡/抖动/过短
   · 被撤销认证的轨迹**绝不**产出实测裁定——该实体被**降级**到
     VLM 层,并以一次显式的"推迟裁定"(deferral)形式上报。
     这道闸门,是别人都没有的诚实环节。

4. FIT LAWS(拟合定律)— `physics/laws.py fit_best_law` + 异常检测器
   · 被动运动族:static / constant_velocity / constant_acceleration
     (重力向量**自由**拟合——不假设 9.81,也不做尺度标定)
   · violation = max(最佳拟合的 RMS 残差, 最严重异常的 severity) ∈ [0,1]
   · 局部化的异常检测器 → 类型化的失效模式:
     teleport → object_permanence · midair_reversal → gravity_inertia ·
     energy_gain → conservation · jerk_spike → collision

5. VERDICT(裁定)— `critics/physics_consistency.py`
   · violation ≥ 阈值(0.4 / strictness)→ PhysicsVerdict{mode, entity,
     frame_range, severity, source="law_verifier", suggested_intervention}
     + 一条镜像的失败 ChecklistItem(kind="physics")
   · 残差偏高但没有任何局部化异常 → mode=UNEXPLAINED(诚实的说法:我们知道
     没有定律拟合得上,但我们**不**知道是哪条定律被打破了)

## 规则

- 没有任何可核验内容时保持沉默(片段不可读/没有标注)——绝不在
  零证据的情况下给出言之凿凿的裁定。
- 每条裁定都写明**是谁**(entity)、**在哪**(frame_range)、**多严重**(severity):
  这正是 DefectReport 的缺陷定位与分段修复所消费的信息。
- 在任意已有视频上演示整条链(所有轨迹均被记录):
  `python scripts/test_physics_review.py --video x.mp4 --prompt "..."`
