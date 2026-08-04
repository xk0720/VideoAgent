---
name: video_retrieval
agent: RetrievalTool + AssetMemory (asset retrieval; window stage B sources + @VideoN source-video references)
description: Retrieve material from the user's uploaded assets (images / videos / identity anchors) — the base of the asset_image / video_extract image sources, the scene_write media catalog, and the @VideoN source-video references on t2v routes.
---

# Video Retrieval — asset retrieval skill

## Role
User-uploaded assets (AssetMemory: video_shots source clips, identity_anchors
character images, style_anchors style images) are the ONLY source of "real
appearance". Consumers:

1. Image-plan source `asset_image`: use a user image directly as a planned
   image (the strongest guarantee of character-look consistency — a real
   photo beats any regeneration).
2. Image-plan source `video_extract`: retrieve a source clip by the shot
   description (retrieve_source_shots) and extract its MIDDLE frame
   (probed-duration midpoint) as a KEY IMAGE — for when a plan needs a
   specific object/moment as a frame. Labeling does NOT depend on this:
   ingest labels come from native video understanding (see below).
3. The scene_write MEDIA CATALOG: images AND videos with their semantic
   labels — the script brain sees what the user provided.
4. @VideoN source-video references: user videos ride natively on the t2v
   strategies (≤3, 15-second head clips, seedance-2.0 limits).
(retrieve_replace is RETIRED from the repair menu; only a
legacy execute handler remains.)

## Retrieval rules
- retrieve_source_shots(query): use the FULL shot description as the query,
  never a single word (ranking = cosine between a deterministic hashed
  bag-of-tokens embedding of the query and each shot's ingest embedding —
  full descriptions maximize token overlap).
- Every retrieval returns (path, ACTUAL semantic label), and the actual
  label — user description > VLM caption > filename — is what travels
  into downstream prompts, never the search query (describe
  what was actually retrieved, not what was searched for).
- Image assets are scored across the WHOLE catalog by keyword overlap between
  the query and each asset's label; the label priority chain is
  user-provided description > VLM caption (backfilled by
  ensure_asset_descriptions when a real VLM is available) > filename
  (honest degradation — retrieval quality is limited and logged).
- VIDEO assets are labeled by NATIVE VIDEO UNDERSTANDING:
  the VLM watches the WHOLE clip and writes identity words + setting + the
  main motion/camera — because a shot may directly continue the user's
  footage, the label must describe the clip's content and movement, not
  one frame. Degradation chain: native video caption > middle-frame
  caption (VLMs without a video channel, e.g. qwen-local) > filename
  (loud). Frame EXTRACTION itself is unchanged — video_extract still
  pulls a key image from the clip when a plan needs an exact frame.
- A hit whose file no longer exists is NOT a hit — degrade honestly.
- Gating: with an empty asset library, the strategies/tools that depend on
  this skill disappear from the menus (the brain never sees an inexecutable
  option); "pretending to retrieve" is forbidden.

## Current implementation status (honest note)
Retrieval is currently deterministic keyword/label matching; CLIP-embedding
retrieval is the registered upgrade path. The rules above (full-description
queries, existing-path requirement, empty-library gating) stay unchanged
when it lands.
