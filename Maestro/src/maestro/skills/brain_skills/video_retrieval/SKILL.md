---
name: video_retrieval
agent: RetrievalTool + AssetMemory(素材检索,窗口大循环 §B / 修复工具 retrieve_replace)
description: 从用户上传的素材库(图片/视频/身份锚)检索材料——keyframe 的两种来源(asset_image / video_extract)和修复工具 retrieve_replace 的底座。
---

# Video Retrieval — 素材检索技能

## 角色
用户上传的素材(AssetMemory:video_shots 源视频片段、identity_anchors
身份图、style_anchors 风格图)是"真实外观"的唯一来源。三个消费方:

1. §B keyframe 策略 `asset_image`:直接拿身份/风格锚的图片当本 shot 的
   keyframe(角色长相一致性的最强保证——真图赢过任何再生成)。
2. §B keyframe 策略 `video_extract`:按 shot 描述检索源视频片段
   (retrieve_source_shots),抽中间帧当 keyframe(中间帧比首帧更能代表
   片段内容)。
3. 修复工具 `retrieve_replace`(修复 brain 的菜单项):语义缺陷"缺某个真
   实元素"时,用源片段整段替换生成镜头。

## 检索规则
- retrieve_source_shots(query):query 用 shot 的完整描述,不要只给单词
  (匹配按 caption/标签重叠打分)。
- 优先级:identity 锚 > style 锚(身份是一致性的命门);源视频片段必须
  验证路径存在才算命中——检索命中但文件丢失 = 未命中,诚实降级。
- 门控:素材库为空时,依赖本技能的策略/工具从菜单里消失(brain 看不见
  不可执行的选项),绝不允许"假装检索到了"。

## 当前实现状态(诚实声明)
检索目前是确定性关键词/标签匹配(RetrievalTool);CLIP 向量检索是升级
方向,升级时本文件的规则(query 用全描述、路径必须存在、空库即隐藏)不变。
