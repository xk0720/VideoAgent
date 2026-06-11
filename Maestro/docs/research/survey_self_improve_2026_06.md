# Survey — Self-Improving Video Generation Agents & Test-Time Scaling (June 2026)

> Raw survey, collected 2026-06-11 via WebSearch + arXiv. "Key limitation" entries
> are our own critical assessment. Feeds the consolidated innovation doc.

---

## 1. Review → Replan → Regenerate Loops

### VISTA — arXiv:2510.15831 (Oct 2025, Google + NUS)
- Multi-agent test-time loop: structured temporal plan → batch generation → pairwise tournament selection with probing critiques → trio of critics (visual/audio/contextual) → Deep Thinking Prompting Agent rewrites prompts. Black-box over Veo-class models; 66.4% human preference.
- **Limitation:** Improvement confined to *prompt space* — cannot fix what the generator can't render (physics, identity drift); each iteration costs multiple full generations; MLLM tournaments noisy/self-preferencing; **no memory across sessions — every request restarts from zero.**
- **Opportunity:** couple prompt-space refinement with noise/latent search and persistent cross-task experience; cheaper verification than full-tournament generation.

### GenMAC — arXiv:2412.04440 (AAAI 2026)
- Design→Generation→Redesign; redesign decomposed into 4 sequential MLLM agents + self-routing to scenario-specialized correction agents; corrections update prompts, frame-wise layouts, guidance scales.
- **Limitation:** Bound to layout-controllable older backbones and *spatial* compositionality; handcrafted routing brittle; no temporal/narrative failure handling.

### VideoRepair — arXiv:2411.15115 (Nov 2024, UNC)
- Training-free, model-agnostic: MLLM QA localizes text-video misalignment → Region-Preserving Segmentation keeps correct regions → regenerates only misaligned regions.
- **Limitation:** Spatial inpainting only — no motion/event-order/physics repair; breaks under large motion; single round.
- **Opportunity:** *temporally* localized repair (fix seconds 3–5, keep the rest); iterate with a verifier until convergence.

### Mora — arXiv:2403.13248 (2024, baseline)
- Multi-agent pipeline composing open-source modules; feedback largely human-in-the-loop. Open-loop compounding stage errors.

### SciTalk — arXiv:2504.18805 (Apr 2025)
- Agents simulate viewer perspectives, feed critiques back into prompts; domain-narrow; simulated-viewer judgment unvalidated.

### MAViS — arXiv:2508.08487 (Aug 2025)
- End-to-end multi-agent long-sequence storytelling (script→shot design→keyframes→video→audio) with "3E" (Explore, Examine, Enhance) per-stage review.
- **Limitation:** Review is *per-stage and local* — **no global credit assignment** (a bad final video may stem from the script, but only the video stage gets re-examined); capped by frozen generators.
- **Opportunity:** cross-stage blame assignment and replanning (go back to the script when shot-level retries fail).

### GenAgent — arXiv:2409.01392 (Sep 2024)
- LLM constructs/debugs ComfyUI workflows-as-code, fixing failed executions.
- **Limitation:** Refines workflow *executability*, not output *quality*.
- **Opportunity:** quality-driven workflow search.

---

## 2. Test-Time Scaling / Best-of-N / Search for Video Diffusion

### Video-T1 — arXiv:2503.18942 (ICCV 2025)
- TTS as search from noise space; Tree-of-Frames: AR frame-level branch expansion with verifier-guided pruning at 3 stages.
- **Limitation:** Requires (pseudo-)AR generation — mismatches full-sequence DiTs; bounded by gameable VBench/MLLM verifiers; huge compute per quality delta.

### EvoSearch — arXiv:2505.17618 (May 2025)
- Evolutionary search along the denoising trajectory (selection on reward, SDE-tailored mutation, population diversity). 1.3B Wan beats 14B with the same verifier.
- **Limitation:** Population × denoising compute; rewards over-optimizable; no early-termination theory.
- **Opportunity:** sample-adaptive compute allocation; robust/ensembled verifiers.

### ScalingNoise — arXiv:2503.16400 (Mar 2025)
- Beam-like search for "golden" initial noises in long/streaming generation; one-step-denoised previews scored by a reward anchored on previously generated content.
- **Limitation:** One-step previews are crude value estimates; anchoring on past content **entrenches drift once errors enter history** (self-conditioning bias).
- **Opportunity:** multi-fidelity value estimation; recovery from (not just propagation of) accumulated errors.

### VideoReward / Flow-NRG — arXiv:2501.13918 (NeurIPS 2025, Kling)
- 182K-pair preference dataset; multi-dimensional VLM reward (Visual Quality / Motion Quality / Text Alignment); Flow-DPO, Flow-RWR, Flow-NRG (inference-time noise guidance with user weights).
- **Limitation:** Three coarse axes — no narrative, audio, fine physics; preference distribution-shift to new backbones; RM is the hackable bottleneck in test-time search.
- **Opportunity:** process rewards per-timestep/per-shot; physics- and story-aware reward dimensions; RM uncertainty to resist over-optimization.

### TTOM — arXiv:2510.07940 (ICLR 2026)
- Test-Time Optimization and Memorization: inference-time parameter insertion optimized against layout-attention alignment; **parametric memory caches optimization contexts across a prompt stream** (insert/read/update/delete).
- **Limitation:** White-box gradients required; spatial-compositionality objective; parametric memory with unclear interference/staleness behavior.
- **Opportunity:** the memorization seed of *cumulative test-time learning* — extending to semantic/episodic experience and temporal objectives is open.

### VChain — arXiv:2510.05094 (Oct 2025)
- GPT-4o infers sparse causal keyframes ("visual thoughts"); generator sparsely tuned on them at inference to inject state transitions.
- **Limitation:** Per-sample tuning costly; GPT-4o keyframes may be inconsistent with generator style/identity; sparse discrete changes only.

### LatSearch — arXiv:2603.14526 (Mar 2026)
- Latent reward model scoring *partially denoised latents* at arbitrary timesteps → reward-guided resampling/pruning without decoding. Beats Best-of-N and EvoSearch at matched compute.
- **Limitation:** Latent RM is backbone-specific; early-timestep latents weakly informative → confident-only pruning needed.
- **Opportunity:** backbone-agnostic process rewards; calibrated pruning.

### RAPO++ — arXiv:2510.20206 (Oct 2025)
- Cross-stage prompt optimization: retrieval-augmented refactor → test-time closed-loop per-sample prompt refinement with multi-source feedback → fine-tune the rewriter on optimized pairs (internalizing lessons).
- **Limitation:** Pure prompt space; pulls prompts toward training distribution (sacrifices intent/creativity); proxy metrics get Goodharted.
- **Opportunity:** Stage-3 distillation of *semantic failure lessons* (not just prompt pairs) is open.

### Inference-Time Scaling for Joint AV Generation — arXiv:2606.03183 (Jun 2026)
- First TTS for joint audio-video: multiple verifiers (video/audio quality, AV sync, text alignment) combined via Adaptive Reward Weighting guiding particle search.
- **Limitation:** Evaluated only on VGGSound/JavisBench-mini; heuristic weighting over immature AV-sync verifiers.

---

## 3. Self-Improvement via Experience

### VideoAgent (self-improving, embodied) — arXiv:2410.10076 (ICLR 2025)
- Refines generated video *plans* via self-conditioning consistency + VLM feedback; successfully executed plans become new training data — a real flywheel.
- **Limitation:** Depends on **ground-truth task success from an environment** — creative video has no such oracle; per-domain refinement training.
- **Opportunity:** an "environment signal" surrogate for creative video (human edits, retention, downstream task success).

### SAIL — arXiv:2506.06658 (Jun 2025)
- In-domain video model iteratively fine-tunes on self-generated, successfully-executed trajectories. Same oracle dependence; only works where success is binary and checkable.

### MemoGen — arXiv:2606.03243 (Jun 2026) — *T2I, the transferable blueprint*
- Training-free "agentic evolution layer" over frozen image generators: infer requirements → retrieve references → generate → evaluate → **write back experience memory** (task understanding, reference choices, successful strategies, failure lessons), retrieved on future similar tasks. Learning lives in external memory state, not parameters.
- **Limitation:** Image-only; no memory quality control (wrong lessons persist and get retrieved), no interference analysis, no long-horizon evaluation of self-evolution.
- **Opportunity:** exactly what's missing in video — a **validated** failure-lesson library with memory hygiene/decay and measured transfer.

**Gap finding:** Reflexion-style lesson/insight libraries applied to *video generation* returned essentially nothing as of June 2026 — VISTA is test-time-only, MemoGen is image-only, TTOM stores parametric (not verbal) memory. This sub-area is nearly empty.

---

## 4. Long-Video / Multi-Shot Story Generation

### MovieAgent — arXiv:2503.07314 — hierarchical multi-agent CoT planning; **fundamentally open-loop** (no critic/regeneration); character-bank consistency drifts.
### Anim-Director — arXiv:2408.09787 (SIGGRAPH Asia 2024) — GPT-4 director over Midjourney+Pika; reflection checks images, not motion.
### FilmAgent — arXiv:2501.12909 — critique-and-revise in fixed Unity 3D scenes; doesn't generate video; insights may not transfer to stochastic diffusion backends.
### LCT — arXiv:2503.10589 — scene-level attention (interleaved 3D RoPE, async noise) for multi-shot; quadratic → minute-scale ceiling; consistency learned, not guaranteed; no post-hoc correction.
### HoloCine — arXiv:2510.20822 (CVPR 2026 Highlight) — Window Cross-Attention + Sparse Inter-Shot Self-Attention; *emergent* persistent memory of characters/scenes.
- **Limitation:** Emergent memory is uncontrollable/unverifiable (cannot pin or fix a drifted identity); single-pass — an error in shot 2 of 12 requires regenerating everything.
- **Opportunity:** editable/addressable memory; partial-regeneration interfaces for agentic loops.
### Captain Cinema — arXiv:2507.18634 — top-down keyframe planning + interleaved conditioning; keyframe errors baked in.
- **Opportunity:** closed-loop keyframe revision before expensive synthesis — **cheap-to-verify keyframes vs expensive-to-verify videos is an under-exploited asymmetry.**
### VGoT — arXiv:2412.02259 — training-free modular multi-shot; consistency = face embeddings only; strictly feed-forward.
### StoryMem — arXiv:2512.19539 — keyframe memory bank injected via latent concat (LoRA).
- **Limitation:** **Memory ingests generated keyframes uncritically — a flawed shot contaminates all subsequent shots (no quality gate on memory writes);** appearance, not story state.
- **Opportunity:** verified memory writes (critic gates what enters memory) — the natural junction of self-refinement and memory literatures.
### VideoMemory — arXiv:2601.03655 — entity descriptor bank updated per shot; 54-case benchmark; updates can encode wrong changes with no rollback.
### ShotStream — arXiv:2603.25746 — next-shot prediction, causal streaming student via DMD, dual-cache memory; AR error accumulation acknowledged; the *user* is the critic.
- **Opportunity:** automated critic intervening between shots — the natural granularity for review→regenerate.
### Also: STAGE arXiv:2512.12372 (storyboard-anchored, feed-forward); SWIFT arXiv:2605.09442 (prompt-adaptive memory); AesopAgent arXiv:2403.07952 (early RAG "expertise library" ancestor).

---

## Synthesis — biggest open problems (mid-2026)

1. **The verifier is the bottleneck, and it's hackable.** Every TTS method and agentic loop is capped by coarse, miscalibrated, over-optimizable MLLM/VBench judges. Nobody has a calibrated, uncertainty-aware, story-level verifier — best-of-N against a weak verifier converges to reward hacking.
2. **Refinement is trapped in prompt/noise space; targeted regeneration is unsolved.** Black-box loops rewrite prompts; search reselects noise; white-box needs gradients. *Temporally* localized repair ("fix shot 4, keep the rest") barely exists; holistic multi-shot architectures make partial regen architecturally awkward. The field needs an "edit, don't regenerate" middle layer between critic and generator.
3. **No persistent learning: agents are amnesiac.** VISTA et al. reset per request; lessons from thousands of expensive failures are discarded. A Reflexion-style **validated** failure/lesson library for video generation — with memory hygiene, interference control, measured transfer — is essentially unclaimed, largely because there is no environment-grounded success signal to anchor the flywheel.
4. **Iteration economics don't close.** One critique-regenerate cycle = one or more full generations; tournaments/populations multiply it. Cheap-to-verify intermediate artifacts (storyboards, keyframes, partial latents, one-step previews) are exploited only in isolation. **An end-to-end system that verifies at the cheapest sufficient fidelity — script → storyboard → keyframe → latent → clip — with adaptive per-sample compute does not exist.**
5. **Consistency architectures and agentic loops haven't been married.** Memory banks ingest generated content uncritically (errors compound); agentic critics treat shots independently with no cross-stage credit assignment. The obvious synthesis — critic-gated memory writes, shot-level accept/reject/rollback, narrative-level blame assignment — is the clearest open system-building opportunity.
