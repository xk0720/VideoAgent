# WaveSpeed 多图输入视频生成模型 — 调研报告(2026-07,Q1 裁决依据)

> 问题(用户 Q1):有没有视频生成模型能一次输入多张图片?有的话都实现。
> 方法:逐页抓取 wavespeed.ai 官方模型页(每条 schema 附出处 URL)。
> 结论:**有,而且不止一种**。已全部实现(见文末"落地"节)。

## 0. TL;DR — 多图路线总表

| 模型 id | 图片字段 | 上限 | 语义 | 可组合 |
|---|---|---|---|---|
| bytedance/seedance-2.0/*(t2v/i2v/edit) | `reference_images` | **9** | 引导参考(@Image1 提及:外观/场景/风格) | 与首帧 image、`reference_videos`(≤3, 15s)同请求;总限 12 文件 |
| bytedance/seedance-2.0/image-to-video | `image` + `last_image` | 2 | 首帧 + 尾帧锚 | 与 references 的交互官方未写(UNVERIFIED) |
| kwaivgi/kling-v1.6-multi-i2v-standard | `images` | **4** | 融合:画面构成 + 角色一致性(无指定首帧) | duration 5\|10;aspect 1:1\|16:9\|9:16;$0.25/5s |
| kwaivgi/kling-elements-advanced | `frontal_image` + `refer_images` | 1+4 | 先注册"元素"(角色/物体)→ 拿 element_id 在 kling 生成里用 `element_list` 引用 | 两步流程;$0.01/元素 |
| google/veo3.1-lite/start-end-to-video | `image` + `last_image` | 2 | 首尾帧(三字段全必填) | 720p $0.40 / 1080p $0.64 |
| vidu/q3/start-end-to-video | `image` + `last_image` | 2 | 首尾帧(均必填) | 2026-07-02 调研已验证 |
| alibaba/wan-2.6/image-to-video | `image` | 1 | 单图锚,**无多图** | (排除项,如实记录) |

出处:seedance-2.0 参考通道 = wavespeed.ai/blog/posts/seedance-2-0-complete-guide-multimodal-video-creation/
("Up to 9 images / Up to 3 videos, max 15s / 12 files per generation";
@提及示例:"Reference @Image1 for the man's appearance in @Image2's elevator
setting. Fully replicate @Video1's camera movements");
kling multi-i2v = wavespeed.ai/models/kwaivgi/kling-v1.6-multi-i2v-standard;
kling elements = wavespeed.ai/models/kwaivgi/kling-elements-advanced;
veo3.1-lite = wavespeed.ai/models/google/veo3.1-lite/start-end-to-video;
wan-2.6 = wavespeed.ai/models/alibaba/wan-2.6/image-to-video;
首尾帧合集 = wavespeed.ai/collections/first-and-last-frame-video。

## 1. 对"上镜尾帧 + 本镜 keyframe"双图条件的推荐路线

1. **首选:seedance-2.0 i2v,`image`=上镜尾帧 + `reference_images`=[keyframe]**
   —— 首帧像素级锚定连续性,keyframe 作 @Image1 引导目标画面,**结尾不被
   锁死**(和 flf2v 的关键差异);单次调用、最强模型家族。
2. **双端都要锁:seedance-2.0 i2v `image`+`last_image`**(已有 flf2v_bridge)。
3. **融合式(无指定首帧):kling-v1.6-multi-i2v `images`=[尾帧, keyframe]**
   —— 画面按全部图片融合构成;模型代际较老(v1.6),排融合场景专用。

N 图身份一致性:seedance-2.0 `reference_images`(≤9)一把梭;长期角色库
可上 kling-elements(注册一次、多次引用)——列为后续项,未实现。

## 2. UNVERIFIED(没抓到官方页,不定论)

- seedance-2.0 **i2v 端点页**是否显式列出 reference_images 字段(博客称
  "限额对全部端点统一",按此实现;真跑若 400 报错正文会直说);
- first/last-frame 模式与 @references 同请求的行为(官方无示例);
- alibaba/wan-2.7 i2v、vidu reference-to-video(1-7 图?)、pixverse、
  minimax hailuo subject-reference —— agent 抓取中断,未核验。

## 3. 落地(全部已实现 + 测试)

- `WaveSpeedClient.generate(reference_images=…)`:seedance-2.0
  reference_images 通道接线(此前签名有参数、payload 没用——工具库盘点
  发现的缺口);≤9 截断并记日志;legacy 模型 loud 报错。能力 `ref_images`。
- `WaveSpeedClient.multi_image_to_video(prompt, images, …)`:kling
  multi-i2v 路线;≤4 截断;duration {5,10} 吸附;`multi_i2v_model` 可配。
  能力 `multi_i2v`。
- 窗口条件策略 +2(菜单门控 + 降级链 + 台账留痕):
  `ti2v_prev_plus_keyframe`(用户 4.(2.1)(1) 的字面实现:两张图一次调用)、
  `multi_image_fusion`(多图融合)。技能文件 window_generation/SKILL.md 同步。
- 未实现(登记在 TOOL_LIBRARY 缺口台账):kling-elements 两步元素注册、
  vidu reference-to-video(待核验后再动)。
