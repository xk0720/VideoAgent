# WaveSpeed 多图输入视频生成模型 — 调研报告(2026-07,Q1 裁决依据)

> 问题(用户 Q1):有没有视频生成模型能一次输入多张图片?有的话都实现。
> 方法:逐页抓取 wavespeed.ai 官方模型页(每条 schema 附出处 URL)。
> 结论:**有,而且不止一种**。已全部实现(见文末"落地"节)。

## 0. TL;DR — 多图路线总表(agent 完整报告版,18+ 官方页逐页核验)

| 模型 id | 图片字段 | 上限 | 备注 |
|---|---|---|---|
| bytedance/seedance-2.0/**text-to-video** | `reference_images` | **9** | @ImageN 提及;+`reference_videos`(≤3,15s)+`reference_audios`(≤3),总限 12 文件;**refs 仅在 t2v 端点验证** |
| bytedance/seedance-2.0/**image-to-video** | `image`+`last_image` | 2 | **schema 无 reference_images**(README 暗示 ≤4 但 schema 未列 = UNVERIFIED,不硬编码) |
| kwaivgi/**kling-video-o1**/reference-to-video | `images` | **7**(带 `video` 时降为 **4**) | 官方原文:"If the input reference parameters include a video, the number of reference images … will be reduced to 4";$0.112/s(带视频 $0.168/s) |
| kwaivgi/kling-v3.0-4k/image-to-video | `image`+`end_image`+`element_list`(≤3 元素) | 2+3 | **唯一"双硬锚+身份引用"同 schema** 的模型;element_list 存 element_id(两步流程) |
| kwaivgi/kling-elements-advanced | `frontal_image`+`refer_images`(2-4) | 5/元素 | 注册身份元素→element_id;$0.01/元素 |
| kwaivgi/kling-v1.6-multi-i2v-standard | `images` | 4 | legacy,被 video-o1 取代 |
| vidu/q3/reference-to-video | `images` | **4** | 多实体一致性;$0.35/5s@480p |
| vidu/reference-to-video-q2 | `images` | **7** | 廉价高数量 refs |
| vidu/q3/start-end-to-video | `image`+`last_image` | 2 | 双帧均必填 |
| google/veo3.1/reference-to-video | `images` | **3** | 固定 8s,原生音频,$3.20/run |
| google/veo3.1-lite/start-end-to-video | `image`+`last_image` | 2 | $0.40/720p |
| wavespeed-ai/wan-flf2v | `first_image`+`last_image` | 2 | 最便宜 FLF($0.30),无音频 |
| alibaba/wan-2.7/reference-to-video | `reference_images`(≤5)+`image` | 5 共享 | `videos`+`reference_images` 共享 1-5 预算 |
| pixverse/pixverse-c1/reference-to-video | `images`:[{image_url, type: subject\|background, ref_name}] | **7** | **最可控**:显式 subject/background 标签 + @ref_name 提及 |
| x-ai/grok-imagine-video/reference-to-video | `images` | **7** | @imageN;$0.05/s;≤720p |
| alibaba/wan-2.6/i2v · minimax/hailuo-2.3/i2v-pro | `image` | 1 | 排除项(单图) |

出处:seedance-2.0 参考通道 = wavespeed.ai/blog/posts/seedance-2-0-complete-guide-multimodal-video-creation/
("Up to 9 images / Up to 3 videos, max 15s / 12 files per generation";
@提及示例:"Reference @Image1 for the man's appearance in @Image2's elevator
setting. Fully replicate @Video1's camera movements");
kling multi-i2v = wavespeed.ai/models/kwaivgi/kling-v1.6-multi-i2v-standard;
kling elements = wavespeed.ai/models/kwaivgi/kling-elements-advanced;
veo3.1-lite = wavespeed.ai/models/google/veo3.1-lite/start-end-to-video;
wan-2.6 = wavespeed.ai/models/alibaba/wan-2.6/image-to-video;
首尾帧合集 = wavespeed.ai/collections/first-and-last-frame-video。

## 1. 对"上镜尾帧 + 本镜 keyframe"双图条件的推荐路线(按报告修正)

1. **硬双锚首选:seedance-2.0 i2v `image`+`last_image`**(= flf2v_bridge):
   像素级锁两端;4-15s、4K、原生音频——唯一同时满足这些的 FLF 模型。
2. **软双图:seedance-2.0 t2v + `reference_images`=[尾帧, keyframe]**
   (= ti2v_prev_plus_keyframe):@Image1 续接起点、@Image2 目标构图;
   构图级连续、不锁任何帧;可再叠上镜尾段当 `reference_videos`。
   ⚠ 修正:refs 仅在 **t2v** 端点验证;i2v+refs 是未验证 schema,不硬编码。
3. **多图融合:kling-video-o1 `images`(≤7,带 video ≤4)**
   (= multi_image_fusion):无指定首帧;可同请求带上镜尾段视频。
4. **双硬锚+身份**(未实现,登记):kling-v3.0-4k i2v `image`+`end_image`
   +`element_list`(先 kling-elements 注册身份)。

N 图身份一致性:seedance-2.0 t2v refs(≤9)一把梭;结构化可控选
pixverse-c1(subject/background 标签)——登记待做。

## 2. UNVERIFIED(报告原文确认的不定论项)

- **seedance-2.0 i2v 的 reference_images**:i2v 页 README 写"up to 4
  reference images",但 schema(模型页+docs-api 页)只列 image+last_image
  —— 视为未暴露,代码里 i2v+refs 主动丢弃并大声告警,refs 走 t2v;
- seedance first/last-frame 模式 × @references 同请求行为(官方无示例);
- kling v3.0-4k element_list 数组项的精确 JSON 形状;end_image+element_list
  同用只被 schema 暗示、无演示;
- wan-2.7 的 WaveSpeed 首尾帧模型 id(官方 guide 只给伪代码);
- minimax subject-reference 现状、vidu start-end 旧系列各 schema、
  每图大小上限(仅上传端点 ~300MB/7 天保存期有文档)。

## 3. 落地(全部已实现 + 测试)

- `WaveSpeedClient.generate(reference_images=…)`:**t2v 端点**接线(≤9,
  @ImageN);i2v+refs = 丢弃 + loud 告警(schema 未验证,不硬编码);
  legacy 模型 loud 报错。能力 `ref_images`。
- `WaveSpeedClient.multi_image_to_video(prompt, images, …, video=None)`:
  默认 **kling-video-o1/reference-to-video**(≤7 图;带 video 时 ≤4,按官方
  规则自动收缩);legacy kling-v1.6 schema 保留在 `multi_i2v_model` 后面。
  能力 `multi_i2v`。
- 窗口条件策略 +2(菜单门控 + 降级链 + 台账留痕):
  `ti2v_prev_plus_keyframe`(**t2v+refs 软锚**:@Image1 续接起点 + @Image2
  目标构图;要像素级续接用 ti2v_prev_last/flf2v_bridge——技能文件写明)、
  `multi_image_fusion`(video-o1 多图融合)。兜底优先级按硬锚>软锚重排。
- 未实现(登记在 TOOL_LIBRARY 缺口台账):kling-v3.0-4k 双硬锚+element_list
  路线、kling-elements 元素注册、pixverse-c1 标签化 refs、vidu r2v。
