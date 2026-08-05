# Skills——各工位的操作手册

一个工位一份 skill;一条法则只有一个家。每个 skill 文件夹里放一份
SKILL.md(frontmatter:name / agent / description),由 `loader.py`
整篇载入对应 agent 的提示词。每条法则只住在唯一一份 skill 里;代码在
每条法则背后都立着一道确定性的门禁来强制执行。

生产链(brain):
  brain_skills/screenplay          点子 → 剧本(用户自带剧本时跳过)
  brain_skills/character_extract   剧本 → 角色正典(canon)(用户给定角色时:图片说明本身就是正典)
  brain_skills/scene_write         剧本 → 分镜(镜头、结束状态、台词、背景预测、配乐规划)
  brain_skills/scene_image         每个 bg_id 一条「空景」(EMPTY)背景板提示词
  brain_skills/image_plan          角色/关键帧图片(人像提示词工艺;逐镜头需求)
  brain_skills/window_generation   视频提示词的正牌法典 + 条件策略语义
  brain_skills/prompt_enhancer     出厂前最后一道工序:接缝连续性 + 引用正确性
  brain_skills/orchestrator        修复决策(接受 / 转场 / 重新生成……)

审片链(VLM):
  reviewer_skills/semantic_critic  语义 + 条件遵循度(正典 = 图片说明)
  reviewer_skills/physics_critic   物理合理性(观点档)
  reviewer_skills/physics_measure  实测物理链(证据档)
  verifier_skills/verifier         每次修复后的盲测 A/B 收/退门禁

已退役(仅保留在 skills_backup/ 中):video_prompt_writing(已并入
window_generation——代码从未把它当作文件加载过)、review_summarizer
(verifier 自己负责自己的审阅)、video_retrieval(默认没有用户视频)。
