# viral_studio 记忆库索引

> Agent 每次创作前加载本索引；按需再读具体卡片。卡片三层：视频总卡(结构模板) /
> 段级资产卡(可检索可执行的工作单元) / 策略卡(可复用的编排套路+prompt模板)。
> `compat.animate_preflight` 含义: pass_verified=实测可作animate驱动;
> fail_no_human / fail_full_face=实测被拒(负例, 勿喂animate); untested=未实测。

## 视频总卡 (videos/)
- [v01](videos/v01_collage_reveal_haul.yaml) 三人拼贴依次亮相·白色系合集 — 54.4s/112BPM; hook=三人弹入
- [v02](videos/v02_cider_talk_haul.yaml) 海报卡点开场·讲解型合集 — 50.2s/123BPM; animate实测6成5拒
- [v03](videos/v03_transition_cta_haul.yaml) 贴纸hook·同景换装·快切CTA — 35.9s/112BPM

## 段级资产卡 (assets/cards/)
| 卡 | 类型 | animate兼容 | 一句话用途 |
|---|---|---|---|
| [v02_poster_beat_reel](assets/cards/v02_poster_beat_reel.yaml) | motion | ✅ 实测 | 卡点开场hook的驱动素材(5s, 已验证成功) |
| [v02_green_dress_talk](assets/cards/v02_green_dress_talk.yaml) | vo | ✅ 实测 | 讲解段动作+口型驱动源 |
| [v02_white_top_talk](assets/cards/v02_white_top_talk.yaml) | vo | ✅ 实测 | CTA口播动作源, 手势节奏强 |
| [v03_same_scene_outfit_swap](assets/cards/v03_same_scene_outfit_swap.yaml) | motion | ⚠️ 未测 | 同景换装种草结构(镜中人偏小有风险) |
| [v03_fastcut_cta_outro](assets/cards/v03_fastcut_cta_outro.yaml) | motion | ⚠️ 未测 | 快切CTA收尾(墨镜对FullFace未知) |
| [v01_opening_reveal](assets/cards/v01_opening_reveal.yaml) | structure | ❌ NoHuman | 三人依次亮相编排参考→走自创 |
| [v01_dress_detail](assets/cards/v01_dress_detail.yaml) | structure | ❌ NoHuman | 商品质感特写b-roll范本→改绘+i2v |
| [v03_sticker_pop_hook](assets/cards/v03_sticker_pop_hook.yaml) | fx | ⚠️ 特效类 | 贴纸弹出hook=剪辑期能力 |
| [v02_mirror_smallperson](assets/cards/v02_mirror_smallperson.yaml) | 负例 | ❌ NoHuman | 镜中人太小→别喂animate |
| [v02_mirror_phoneface](assets/cards/v02_mirror_phoneface.yaml) | 负例 | ❌ FullFace | 手机挡脸→别喂animate |

## 策略卡 (patterns/)
- [multi_person_reveal](patterns/multi_person_reveal.yaml) 多人依次亮相 — 多hook开场首选; 模板A轮流(稳)/模板B叠加(还原v01, 风险高)
- [beat_pose_swap](patterns/beat_pose_swap.yaml) 卡点换姿势 — 开场/收尾强节奏段; ⚠️实测两条路线都做不出硬切, 只得"平滑换姿势"
- [same_scene_outfit_swap](patterns/same_scene_outfit_swap.yaml) 同景同人换装 — 一人多SKU最高效结构; 商品图作@ImageN
- [talking_head_multishot](patterns/talking_head_multishot.yaml) **10s三动作口播** — ✅实测台词9/9准确、中文口型可用、身份稳定; 多hook可一人一段分色叙述

## 全局实测结论(跨 pattern, 规划时必须知道)
- **硬切做不到**: animate 抹平驱动视频的硬切(Δ108→Δ19); seedance 无视 hard cut 指令,
  改用推拉/景别变化。要卡点硬切 → 生成多段后期 ffmpeg 拼。
- **中文口播可用**: seedance `generate_audio` 出中文人声+口型, 台词逐字准确;
  长度按 3秒≈10-13字 配, 太短会留静默, 相邻句语义太近会被连读。
- **单图锁身份足够**: 一张 @Image1 即可稳住人物身份与服装配色, 跨段一致。

## 待办
- [ ] 音频理解补全: 口播文稿(ASR/omni)、BGM情绪标签 — 卡内标"待omni"处
- [ ] v03 三张motion卡的animate本地预检/实测
- [ ] Seedance时间控制实测结论回填 patterns/*.timing_note
