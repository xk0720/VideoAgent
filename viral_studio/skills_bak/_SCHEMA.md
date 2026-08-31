# Skill 卡片结构

三类 skill 共用一套字段，Planner 靠 `applies_to` 检索，Act Agent 靠 `pipeline` +
`needs_background` 决定开几张工单，`prompt_template` 是写死的正文、`slots` 是仅有的插槽。

| 字段 | 作用 |
|---|---|
| `skill_id` | 唯一标识，分镜脚本里引用它 |
| `kind` | `asset_driven` / `template` / `closer` |
| `applies_to.categories` | 商品类目 → Planner 检索主键 |
| `applies_to.person_count` | 适用的 hook 人数（不匹配则该 skill 不可选） |
| `applies_to.placement` | `opening` / `body` / `ending` |
| `produces` | 时长、音频模式（`lipsync`/`voiceover`/`music_only`） |
| `needs_background` | **Act Agent 的分叉点**：true → 先 image_generation 再 video |
| `pipeline` | 声明式步骤链，每步给 `id` + `tool`，后续步骤用 `@id` 引用产物 |
| `prompt_template` | 写死的 prompt 正文，`{}` 处为插槽 |
| `slots` | 允许 Planner 填的空 + 填写约束（实测得出的字数/取值范围） |
| `measured` | 实测结论，供 Planner 判断可靠性；未实测的标 `verified: false` |

约定：
- prompt 一律英文，相机指令第一句，禁 `360/spin/rotate quickly`
- 时长与时间戳一律整数秒，窗口最小 1s
- 台词/旁白中文，长度按实测 TTS 语速 5 字/秒 配
