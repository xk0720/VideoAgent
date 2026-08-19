# Skill 库索引

三类策略，Planner 按 `applies_to` 检索、按 `placement` 排期；prompt 全部写死，
只留少量插槽。卡片字段说明见 [`_SCHEMA.md`](_SCHEMA.md)。

## 1. asset_driven — 爆款片段驱动（开场）
| skill | 时长 | 驱动素材 | 实测 |
|---|---|---|---|
| [beat_reel_v02](asset_driven/beat_reel_v02.yaml) | 5s | v02 海报卡点 5s + 自带 BGM(123BPM) | ✅ pro/std 两轮全成 |

> prompt 为空——animate 不吃文本；一致性靠参考图，背景来自 hook 图。
> 待补：你后续提供的另外两条爆款片段。

## 2. templates — 写死的 prompt 模板（主体，按商品类目分）
### apparel（服装）
| skill | 时长 | 音频 | 动作自由度 | 字幕 | 实测 |
|---|---|---|---|---|---|
| [home_talking](templates/apparel/home_talking.yaml) | 10s | **口播**(音画同步) | 受说话节奏约束 | 模型自烧(会写错字) | ✅ 台词 9/9 准确 |
| [outdoor_narration](templates/apparel/outdoor_narration.yaml) | 10s | **旁白**(TTS) | **完全放开** | 后期烧(100%准确) | ✅ 三外景全成 |

> 选型口径：要"真人开口"的信任感 → `home_talking`；要动作花哨、字幕准确 → `outdoor_narration`。

## 3. closers — 收尾片段
| skill | 变体 | 时长 | 实测 |
|---|---|---|---|
| [sequential_reveal](closers/sequential_reveal.yaml) | 3人 / 2人 / 1人跳过 | 6s / 4s | ✅ 3人版(时序精准、未串脸)；2人版已写死未实跑 |

> 本卡已修正实测发现的"中间大两边小"问题：prompt 加了同尺寸/同景深/等距的硬约束。

## 跨 skill 的实测铁律（Planner/Act 必须遵守）
- **硬切做不到**：animate 会抹平驱动片段的硬切，seedance 无视 `hard cut` 改用推拉。要卡点硬切 → 后期剪。
- **整数铁律**：生成时长与时间戳只能整数秒，窗口最小 1s。
- **中文语速**：TTS 约 5 字/秒；口播 3 秒句配 10-13 字，旁白 10 秒配 26-30 字。
- **收尾一击要留衰减**：音乐多生成 2s、裁到 段长+0.5s，画面同步延长，否则最后那记重击在成片里消失。
- **安全动作**：走位/甩袖/摇肩/撩发/回眸/比心均一次成功；禁 `360` `spin` `rotate quickly`。
