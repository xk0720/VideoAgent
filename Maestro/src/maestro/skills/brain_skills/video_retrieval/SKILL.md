---
name: video_retrieval
agent: RetrievalTool + AssetMemory (asset retrieval; window stage B sources / repair tool retrieve_replace)
description: Retrieve material from the user's uploaded assets (images / videos / identity anchors) — the base of the asset_image / video_extract image sources and of the retrieve_replace repair tool.
---

# Video Retrieval — asset retrieval skill

## Role
User-uploaded assets (AssetMemory: video_shots source clips, identity_anchors
character images, style_anchors style images) are the ONLY source of "real
appearance". Three consumers:

1. Image-plan source `asset_image`: use a user image directly as a planned
   image (the strongest guarantee of character-look consistency — a real
   photo beats any regeneration).
2. Image-plan source `video_extract`: retrieve a source clip by the shot
   description (retrieve_source_shots) and extract its MIDDLE frame (more
   representative of the clip than the first frame).
3. Repair tool `retrieve_replace` (repair brain's menu): when a semantic
   defect is "a real element is missing", replace the generated shot with a
   source clip.

## Retrieval rules
- retrieve_source_shots(query): use the FULL shot description as the query,
  never a single word (matching scores by caption/label keyword overlap).
- Image assets are scored across the WHOLE catalog by keyword overlap between
  the query and each asset's label; the label priority chain is
  user-provided description > VLM caption (backfilled by
  ensure_asset_descriptions when a real VLM is available) > filename
  (honest degradation — retrieval quality is limited and logged).
- A hit whose file no longer exists is NOT a hit — degrade honestly.
- Gating: with an empty asset library, the strategies/tools that depend on
  this skill disappear from the menus (the brain never sees an inexecutable
  option); "pretending to retrieve" is forbidden.

## Current implementation status (honest note)
Retrieval is currently deterministic keyword/label matching; CLIP-embedding
retrieval is the registered upgrade path. The rules above (full-description
queries, existing-path requirement, empty-library gating) stay unchanged
when it lands.
