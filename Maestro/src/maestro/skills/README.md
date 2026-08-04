# Skills — the operating manuals

One station, one skill; one law, one home. Each skill folder holds a
SKILL.md (frontmatter: name / agent / description) loaded whole into its
agent's prompt by `loader.py`. Laws live in exactly ONE skill each; code
enforces every law with a deterministic gate behind it.

Production chain (brain):
  brain_skills/screenplay          idea → screenplay (skipped when the user provides one)
  brain_skills/character_extract   screenplay → cast canon (given characters: the image caption IS the canon)
  brain_skills/scene_write         screenplay → storyboard (shots, end states, dialogue, bg prediction, music plan)
  brain_skills/scene_image         one EMPTY background plate prompt per bg_id
  brain_skills/image_plan          character/keyframe images (portrait prompt craft; per-shot needs)
  brain_skills/window_generation   THE video-prompt law book + condition-strategy semantics
  brain_skills/prompt_enhancer     final outgoing pass: junction continuity + reference correctness
  brain_skills/orchestrator        repair decisions (accept / transition / regenerate …)

Review chain (VLM):
  reviewer_skills/semantic_critic  semantics + condition adherence (canon = the image captions)
  reviewer_skills/physics_critic   physical plausibility (opinion tier)
  reviewer_skills/physics_measure  measured physics chain (evidence tier)
  verifier_skills/verifier         blind A/B accept/reject gate after each repair

Retired (kept only in skills_backup/): video_prompt_writing (folded into
window_generation — the code never loaded it as a file), review_summarizer
(the verifier owns its own review), video_retrieval (no user videos by
default).
