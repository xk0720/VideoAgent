# Survey — Localized Video Repair + Downstream Propagation (July 2026)

> Raw survey, collected 2026-07-02 via WebSearch + WebFetch (9 sources fetched
> in full, rest search-verified). Purpose: validate / improve Maestro's
> `pipeline/timeline.py` cascade re-anchor algorithm (repair ONE defective
> segment, re-anchor downstream segments i2v, early-stop on boundary
> similarity, ffmpeg splice) against prior art, and position it for the paper.
> "Comparison" entries are our own critical assessment.

---

## 0. TL;DR

- Prior work **supports** the design: bounded (first+last frame) generation is
  a validated way to regenerate a middle segment with zero downstream impact
  (NUWA-XL, Time Reversal Fusion, Wan-FLF2V, Kling start/end); "edit first
  frame → i2v" is an established propagation paradigm (AnyV2V, Mora).
- **Novelty**: no surveyed agentic framework (UniVA, MovieAgent, Anim-Director,
  Mora, VideoAgent, ViMax, VISTA) implements automated defect-segment repair
  with cascaded re-anchoring + similarity early-stop. Closest: CHIEF
  (arXiv 2606.18591) does clip-local keyframe repair but the **human creator
  manually gates** downstream propagation — no metric, no automatic cascade.
  Our automatic, metric-driven cascade with early-stop is the identifiable delta.
- **Weakest link in our design**: pixel-MSE 0.92 as the early-stop metric has
  no precedent and is brittle (luma shift / 1-frame offset). Field standards:
  CLIP-frame cosine, LPIPS, SSIM, optical-flow warp error. Only documented
  practical threshold anywhere: PySceneDetect HSV mean-diff 27/255 for "shot
  changed".
- **Single best API-adoptable upgrade**: prefer double-anchor FLF2V whenever
  both boundary frames survive review — it structurally *eliminates* the
  cascade instead of early-stopping it.
- Cascaded i2v drift is real and documented; the API-level mitigation is a
  **fixed global anchor/reference conditioning every hop** (StreamingT2V
  insight), not only the previous segment's new last frame.

---

## 1. Keyframe-edit propagation (edit one frame → whole clip)

### 1.1 TokenFlow — arXiv:2307.10373 (ICLR 2024)
- Edits sampled keyframes jointly (T2I editor), then propagates the edited
  diffusion FEATURES to all frames along inter-frame nearest-neighbor
  correspondences from the source video's own features.
- Training-free but needs diffusion internals — **not API-usable**.
- **Comparison:** appearance-only; preserves ORIGINAL motion. Cannot fix motion
  defects — different sub-problem from ours.

### 1.2 Pix2Video — arXiv:2303.04761 (ICCV 2023)
- Edit an anchor frame; progressively edit each next frame by injecting
  self-attention features from anchor+previous frame; guided latent update.
- Training-free, needs internals — **not API-usable**.
- **Comparison:** literally "fix one keyframe, propagate downstream" at
  per-frame granularity — validates anchor-frame-first repair.

### 1.3 CoDeF — arXiv:2308.07926 (CVPR 2024)
- Per-video canonical image + deformation field; edit the canonical image once,
  deterministic propagation to every frame. Needs per-video optimization.
- **Comparison:** strongest propagation guarantee but appearance-only.

### 1.4 VideoSwap — arXiv:2312.02087 (CVPR 2024)
- Semantic-point tracking drags a swapped subject through source motion;
  per-video optimization + user interaction. **Not API-usable.**

### 1.5 AnyV2V — arXiv:2403.14468 (TMLR 2024)
- (1) Edit FIRST frame with any image editor; (2) I2V from the edited frame,
  with DDIM inversion + feature injection so output keeps source motion.
- **Comparison:** the closest paradigm precedent for our i2v re-anchoring —
  our keyframe_edit_propagate is AnyV2V step (2) minus inversion. Price of
  API-only: downstream motion is re-sampled, which is exactly why we need the
  boundary-similarity early-stop.

### 1.6 I2VEdit — arXiv:2405.16537 (SIGGRAPH Asia 2024)
- First-frame-guided editing via I2V + per-clip motion LoRA. **Not
  training-free.**

### 1.7 GenProp — arXiv:2412.19761 (2024)
- Trained model: first-frame edit + learned whole-clip propagation. Hosted
  availability UNVERIFIED. Field converging on "first-frame edit + propagation".

## 2. Segment regeneration with boundary anchoring

### 2.1 Wan-FLF2V — Alibaba, Apr 2025, Apache-2.0
- Wan2.1 T2V + first/last frame conditioning, 720p.
- **Live on WaveSpeed**: https://wavespeed.ai/models/wavespeed-ai/wan-flf2v —
  the exact double-anchor primitive our frame_to_frame uses.

### 2.2 Kling v2.1 start/end frame — Kuaishou (WaveSpeed: kwaivgi/kling-v2.1-i2v-pro/start-end-frame)
- Start+end images → 5/10s transition. Docs WARN: anchors too dissimilar →
  model inserts a **lens switch (cut)** instead of blending.
- **Comparison:** documented failure mode directly relevant to us — pre-check
  anchor-pair similarity before calling flf2v; fall back to i2v+cascade.

### 2.3 Generative Inbetweening — arXiv:2408.15239 (2024)
- SVD fine-tuned to run backwards; dual-directional sampling fusing forward
  (from first key) + backward (from last key). Not training-free/hosted;
  commercial FLF2V APIs are the practical substitute.

### 2.4 Time Reversal Fusion — arXiv:2403.14611 (ECCV 2024)
- "**Bounded generation**": fuse forward denoising path (start frame) with
  backward path (end frame) in a stock I2V model — training-free (needs
  sampler access).
- **Comparison:** key conceptual support — constraining both ends yields a
  segment that CANNOT perturb its neighbors.

### 2.5 SEINE — arXiv:2310.20700 (ICLR 2024)
- Random-mask video diffusion = true temporal inpainting (given frames at
  arbitrary positions, inpaint the middle). Self-hostable, no major API.

### 2.6 NUWA-XL — arXiv:2303.12346 (ACL 2023)
- Coarse-to-fine: global diffusion → sparse keyframes; local diffusions infill
  between adjacent keyframes, in parallel.
- **Comparison:** architectural precedent that keyframe-bounded construction
  makes videos repairable segment-locally BY CONSTRUCTION.

### 2.7 Temporal inpainting — arXiv:2405.00251 (2024), AVID arXiv:2312.03816 (CVPR 2024)
- Masked-region conditional diffusion; research models, not API-usable. Our
  ffmpeg-splice + FLF2V achieves the effect with hosted tools.

## 3. Agentic video frameworks — what happens after review finds a defect?

| System | Repair granularity | Localized? | Propagation? |
|---|---|---|---|
| UniVA (2511.08521) | plan-level self-reflection, multi-round user-driven | no | no (ships flf2v + extend PRIMITIVES, never composes them into repair) |
| MovieAgent (2503.07314) | no post-gen review/repair loop found | — | — |
| Anim-Director (2408.09787) | self-reflect → best-of-N reroll | no | no |
| Mora (2403.13248) | human-in-loop whole-stage reroll | no | no (extends from last frame — same primitive) |
| VideoAgent (2410.10076) | whole-video refinement (robotics) | no | no |
| ViMax (2606.07649) | VLM best-of-k at KEYFRAME stage (prevention, pre-video) | no | no |
| VISTA (id UNVERIFIED) | whole-video reroll via prompt refinement | no | no |
| **CHIEF (2606.18591)** | **clip-local keyframe repair (closest)** | **yes** | **manual: human gates cascade, no metric, no early-stop** |

- **VideoRepair** — arXiv:2411.15115 (2024): training-free, MLLM-QA defect
  detection → localized regeneration, but localization is **SPATIAL** (Molmo +
  Semantic-SAM masks; preserved regions keep initial noise). Needs latent
  access. **Comparison:** the "repair only what's broken" philosophy applied to
  space; ours is its TEMPORAL, API-only counterpart. Cite together.

### Novelty verdict
No system combines (i) automated defect localization to a TIME segment,
(ii) double-anchor regeneration of only that segment, (iii) automatic cascaded
i2v re-anchoring downstream, (iv) similarity-metric early-stop. CHIEF validates
the problem and punts (iii)+(iv) to a human. Residual risk: MovieAgent full
text + unpublished industry systems not exhaustively audited.

## 4. Stopping criteria for propagation

- **No paper defines an early-stop criterion for repair propagation.** Nothing
  supports OR contradicts our 0.92; the metric CHOICE is what needs upgrading.
- Repurposable consistency metrics (EditBoard arXiv:2409.09668; TokenFlow /
  StableVideo evals): CLIP-F (frame-embedding cosine), LPIPS-P/T, SSIM,
  optical-flow warp error. **No published decision thresholds.**
- Only battle-tested threshold anywhere: PySceneDetect ContentDetector — HSV
  mean per-pixel diff > 27.0 (0–255) = shot cut.
  https://www.scenedetect.com/docs/latest/api/detectors.html
- Kling start/end docs: implicit similarity gate on anchor pairs (too different
  → cut inserted).

## 5. Cascaded-continuation failure modes + mitigations

- **Appearance forgetting / error accumulation** — StreamingT2V
  (arXiv:2403.14773, CVPR 2025): AR chunk generation forgets initial
  object/scene features; fix = Appearance Preservation Module conditioning
  EVERY chunk on a fixed anchor frame from chunk 1 (+ short-term attention over
  the previous chunk).
- **Temporal drift** — Rolling Forcing (arXiv:2509.25161): attention-sink
  anchoring of initial frames; Pathwise Test-Time Correction (arXiv:2602.05871):
  initial frame as stable reference calibrating intermediate sampling; Knot
  Forcing (arXiv:2512.21734): notes color/identity drift SURVIVES attention-sink
  fixes long-horizon.
- **Seams** — Gen-L-Video (arXiv:2305.18264), FreeNoise (arXiv:2310.15169,
  ICLR 2024): overlap-and-blend temporal co-denoising in LATENT space — not
  API-reachable; ffmpeg-level analogs below.
- **Practitioner consensus** (Kling drift guide): chain SHORT 3–5s shots, each
  re-initialized, cut between them ("resets error accumulation with each cut"),
  re-inject subject reference images every step — directly endorses our
  segment-wise re-anchor architecture.

## 6. Engineering actions for Maestro (API + ffmpeg only)

1. **Early-stop metric upgrade** (`timeline.frame_similarity`): replace raw
   pixel-MSE with a composite — SSIM on ~256px downscale (structure) + CLIP or
   DINO embedding cosine (identity/semantics) + PySceneDetect-style HSV
   mean-diff 27/255 as hard fail-fast. No published thresholds → calibrate
   per-video: measure natural adjacent-frame and old-vs-old boundary similarity
   in the same clip; stop when new-vs-old falls inside that natural band.
   Heuristic starting points (UNVERIFIED, ours): SSIM ≥ 0.85–0.90, CLIP cosine
   ≥ 0.97–0.98, LPIPS ≤ 0.15. Also sample 2–3 MID-segment frames, not just the
   boundary (drift peaks mid-segment).
2. **Prefer flf2v whenever both boundaries pass review** — provably local
   repair; pre-check anchor-pair similarity (Kling constraint) and fall back to
   i2v+cascade when anchors are too dissimilar.
3. **Anti-drift**: pass a CONSTANT global reference (same prompt + reference
   image via Kling Elements / runway references when available) into every
   downstream i2v hop — converts the chain from pure-Markov (drift compounds)
   to anchored (drift bounded). Re-anchor on the SHARPEST of the last N frames
   (max Laplacian variance), not blindly the last frame.
4. **Seams**: hard cut at the shared anchor frame (dedupe the doubled frame);
   xfade 2–4 frames ONLY when boundary motion is small; color-match the
   regenerated segment to the anchor frame (histogram match / haldclut LUT)
   before splicing; cap cascade depth and finish the chain with a terminal
   flf2v back onto original untouched footage.

## 7. Source list

Fetched in full: UniVA 2511.08521 · VideoRepair 2411.15115 · Anim-Director
2408.09787 · MovieAgent 2503.07314 · CHIEF 2606.18591 · ViMax 2606.07649 ·
VideoAgent 2410.10076 · Wan-FLF2V + Kling model pages (wavespeed.ai) ·
PySceneDetect docs. Search-verified: TokenFlow 2307.10373 · AnyV2V 2403.14468 ·
I2VEdit 2405.16537 · Pix2Video 2303.04761 · CoDeF 2308.07926 · VideoSwap
2312.02087 · GenProp 2412.19761 · Generative Inbetweening 2408.15239 · TRF
2403.14611 · SEINE 2310.20700 · NUWA-XL 2303.12346 · AVID 2312.03816 ·
2405.00251 · Mora 2403.13248 · StreamingT2V 2403.14773 · Rolling Forcing
2509.25161 · Pathwise TTC 2602.05871 · Knot Forcing 2512.21734 · FreeNoise
2310.15169 · Gen-L-Video 2305.18264 · EditBoard 2409.09668 · BAgger
(ryanpo.com/bagger, id UNVERIFIED) · Kling start/end quickstart + drift guide.
