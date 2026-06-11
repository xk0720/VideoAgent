# Survey — Physics in Video Generation: Evaluation, Enhancement, Verification (June 2026)

> Raw survey, collected 2026-06-11 via WebSearch + arXiv. Complements
> `../../PHYSICS_LITERATURE_REVIEW.md` (the C6 sketch-as-verifier repositioning).
> "Key limitation" entries are our own critical assessment.

---

## 1. Physics EVALUATION benchmarks / metrics

### 1.1 PISA — arXiv:2503.09595 (Mar 2025, NYU)
- Isolates free fall as a diagnostic; post-trains I2V on simulated drops with object-trajectory rewards; trajectory-L2 vs ballistic ground truth.
- **Limitation:** Single phenomenon; post-trained models fail to generalize beyond training distribution — measures memorization of one law. Reward needs clean single-object segmentation.
- **Opportunity:** trajectory-vs-analytic residuals for multi-object/multi-law; use as a *selection* signal rather than fine-tuning signal.

### 1.2 VideoPhy — arXiv:2406.03520; VideoPhy-2 — arXiv:2503.06800 (ICLR 2026)
- 200 actions / 3940 prompts; human + fine-tuned VLM AutoEval of semantic adherence, physical commonsense, rule grounding. Best model only 22% joint on hard subset; conservation laws worst.
- **Limitation:** Coarse 1–5 Likert by a VLM — no quantitative measurement, no per-frame violation localization; inherits VLM temporal-reasoning weakness; gameable by smooth-but-wrong motion.
- **Opportunity:** pair semantic judgments with measurement-level (trajectory/contact) verification; per-segment violation localization.

### 1.3 PhyGenBench / PhyGenEval — arXiv:2410.05363 (ICML 2025)
- 160 prompts over 27 laws; hierarchical VLM+LLM evaluation (key phenomenon, order, naturalness).
- **Limitation:** Frame-sampled QA misses continuous dynamics; hand-crafted QA templates per law — cannot evaluate free-form prompts; single-event prompts.
- **Opportunity:** automatic generation of verification programs per prompt; continuous-time metrics.

### 1.4 PhysBench — arXiv:2501.16411 (ICLR 2025)
- 10K entries benchmarking VLM physical *understanding*; PhysAgent (VLM + specialist vision tools) +18.4% for GPT-4o.
- **Limitation:** Understanding-side, but the critical reliability evidence: VLMs broadly fail at physical dynamics; MCQ format inflates apparent competence.
- **Opportunity:** PhysAgent's "VLM + vision tools" pattern is exactly the architecture for a physics critic of *generated* video — unexplored in that direction.

### 1.5 WorldModelBench — arXiv:2502.20694 (NeurIPS 2025)
- 350 conditions, 7 domains; instruction-following + commonsense + physics adherence; 67K human labels; 2B judger beats GPT-4o by 8.6% on violation prediction.
- **Limitation:** Binary "violation present" — no severity, no localization; 2B judge caps temporal reasoning.

### 1.6 Physics-IQ — arXiv:2501.09038 (WACV 2026, DeepMind)
- 396 real 4K videos, 66 scenarios; model predicts 5s continuation; Spatial/Spatiotemporal-IoU, MSE vs real continuation, normalized by two-take variance. Sora: 10%.
- **Limitation:** Pixel/IoU comparison vs a *single* ground-truth continuation penalizes valid alternative outcomes; conflates appearance error with dynamics error; I2V/V2V only.
- **Note:** ICCV 2025 Physics-IQ Challenge won by V-JEPA-2-reward entries — test-time selection demonstrably moves this metric.

### 1.7 Morpheus — arXiv:2504.02918 (Apr 2025)
- 80 real physical experiments; physics-informed metrics: extract dynamics from generated continuations, test energy/momentum conservation with PINNs + vision foundation models — no reference video needed.
- **Limitation:** Small; extraction chain (segmentation→tracking→PINN fit) on generated video is error-prone; conservation checks need calibration monocular video doesn't give.
- **Opportunity:** closest precedent for trajectory/conservation verification oracles — but used only for benchmarking, never for selection or regeneration.

### 1.8 T2VPhysBench — arXiv:2505.00337 (May 2025)
- Human-evaluated, 12 fundamental laws; all models < 0.60; **detailed physics prompt hints don't help and can hurt** (key evidence that prompt-space is a weak actuator).

### 1.9 LikePhys — arXiv:2510.11512 (Oct 2025)
- Training-free: sim-rendered valid/invalid video pairs; diffusion denoising loss as ELBO likelihood surrogate; Plausibility Preference Error measures whether the model prefers the valid clip.
- **Limitation:** Discriminative preference on synthetic pairs; sim-to-real gap; needs white-box diffusion loss access.
- **Opportunity:** diffusion-likelihood preference as a cheap *first-stage filter* in best-of-N (never done).

### 1.10 PhyDetEx — arXiv:2512.01843 (Dec 2025)
- Fine-tuned VLM detects implausibility + produces textual explanation of which law is violated; finds VLMs hold latent physics knowledge but suppress it when assuming the video is real.
- **Limitation:** Free-text explanations unverified against measurements; LLM-as-judge circularity.
- **Opportunity:** grounding explanations in extracted trajectories/contacts → actionable regen feedback (explicitly left open).

### 1.11 Honorable mentions
- **PhyWorldBench** arXiv:2507.13428 (incl. anti-physics prompts); **VideoScience-Bench** arXiv:2512.02942; **PDI / geometric-consistency world-model eval** arXiv:2605.15185 — segmentation + CoTracker3 lifted to 3D, projective-geometry residuals; validates the "tracker → 3D → residual" measurement chain (geometry only, no forces).

---

## 2. Inference-time enhancement WITHOUT retraining

### 2.1 WMReward — arXiv:2601.10553 (Jan 2026)
- Frozen latent world model (V-JEPA 2) scores candidate denoising trajectories; best-of-N / trajectory branching. Won ICCV 2025 Physics-IQ Challenge (62.64%).
- **Limitation:** Opaque learned surprise scalar — cannot say which law, where, by how much → no targeted regeneration, only rerolling; V-JEPA 2's blind spots become silent failure modes; heavy unquantified compute.
- **Opportunity:** combine with an interpretable measurement-based verifier; use reward to *localize* and steer, not just rank.

### 2.2 V-JEPA 2 reward for MAGI-1 — arXiv:2510.21840 (Oct 2025)
- 2-page report; embedding-prediction error reward ranks chunk-wise generation; ~6% Physics-IQ gain.
- **Limitation:** No ablations; modest gain; same opacity. Proves test-time physics selection headroom is real but small with embedding rewards alone.

### 2.3 PSIVG — arXiv:2603.06408 (CVPR 2026)
- First training-free simulator-in-the-loop T2V: template video → 4D scene + mesh reconstruction → engine simulates correct trajectories → re-guide diffusion with simulated motion cues.
- **Limitation:** **Open-loop** (simulate → inject; never verifies what the generator produced); rigid meshes only; 4D reconstruction from an artifact-laden template is fragile; physical parameters guessed.
- **Opportunity:** closing the loop — compare generated motion against simulation, intervene only where they disagree (exactly our verification-oracle framing).

### 2.4 PhyRPR — arXiv:2601.09255 (Jan 2026)
- PhyReason (MLLM infers initial physical states) → PhyPlan (analytic motion scaffold) → PhyRefine (noise-consistent latent fusion).
- **Limitation:** Open-loop; MLLM-inferred states unreliable; scaffold limits motion to ballistic/rigid single-object.
- **Opportunity:** scaffold *correction* from observed trajectories of a first draft.

### 2.5 SDG ("Reasoning the Implausibility") — arXiv:2509.24702 (Sep 2025)
- LLM constructs counterfactual prompts encoding likely violations; Synchronized Decoupled Guidance steers sampling away from the violating mode.
- **Limitation:** Text/CFG-space only — suppresses violations the LLM anticipates *a priori*; no perception of what the video actually does.
- **Opportunity:** instantiate negative guidance from *detected* violations rather than a-priori guesses.

### 2.6 PhyT2V — arXiv:2412.00596 (CVPR 2025)
- Training-free outer loop: generate → VLM captions → LLM CoT identifies physics mismatch → rewrites prompt → regenerate (3-4 rounds). 2.3× rule adherence.
- **Limitation:** Critic = a caption (lossy text bottleneck, no measurement); actuator = the prompt (weak lever per T2VPhysBench); no per-object control; no convergence guarantee.
- **Opportunity:** the closest published "agentic review loop" — weakest oracle and weakest actuator. Simulator+tracker oracle with targeted regen is strictly stronger on both axes.

### 2.7 Self-Refining Video Sampling — arXiv:2601.18577 (Jan 2026)
- Pretrained generators as denoising autoencoders enabling inner-loop refinement; uncertainty-aware self-consistency picks *which regions* to re-refine.
- **Limitation:** Self-consistency ≠ physical correctness — converges to the model's own possibly-wrong mode; no external grounding.
- **Opportunity:** their region-selective refinement is the right *actuator* for targeted regen; it lacks an oracle to aim it.

### 2.8 PhysGen — arXiv:2409.18964 (ECCV 2024); PhysGen3D — arXiv:2503.20746 (CVPR 2025)
- Simulation-AS-generation: image → segmented objects + estimated params → rigid/MPM simulation → diffusion rendering of the rollout.
- **Limitation:** Coverage bounded by simulator + single-image inverse problem; material presets; no open-ended text.
- **Opportunity:** establishes that single-image → calibrated mini-simulation is feasible — the "expected trajectory" half of a verification oracle can be built from the conditioning image alone.

---

## 3. Enhancement WITH training (context; we stay training-free)

- **PhysDreamer** arXiv:2404.13026 — distills video-prior motion into 3D Gaussian material fields (differentiable MPM); per-scene hours; circular when the prior is wrong.
- **Force Prompting** arXiv:2505.19386 — force vectors as conditioning, trained on Blender pairs; no mass/scale grounding; narrow archetypes. *Force-conditioning as a regen actuator is an idea worth borrowing.*
- **PhysCtrl** arXiv:2509.20358 — diffusion physics-trajectory generator (550K sim trajectories, 4 materials) drives I2V via point trajectories; **open-loop, video model deviates from conditioning with no verification.** Their trajectory prior could be a fast differentiable surrogate simulator.
- **VideoREPA** arXiv:2505.23656 (NeurIPS 2025) — Token Relation Distillation from video SSL encoder into DiT; +24.1% VideoPhy. VLM-judged proxy; full fine-tune required.
- **PhysMaster** arXiv:2510.13809 — PhysEncoder rep optimized via DPO on physics preference; optimizes *perceived* physics; PISA-style generalization concerns.
- **PhysRVG** arXiv:2601.11087 — unified RL with physics-grounded feedback (rigid-body/collision rewards), Mimicry–Discovery Cycle; self-built benchmark (circularity); reward hacking risk.
- **PhysVideoGenerator** arXiv:2601.03665 — recovers V-JEPA 2 physical reps *from diffusion latents*; feasibility study. *Latent physics probes as cheap early-exit verifiers.*
- **Phantom** arXiv:2604.08503 — joint visual + latent physical dynamics during training. **DDRL** arXiv:2512.04332 — measures/reduces *reward hacking* in video RL — the central failure mode for any learned physics reward.

---

## 4. Vision-based plausibility checking (the "observed trajectory" side)

### 4.1 CoTracker3 — arXiv:2410.11831 (Meta)
- SOTA point tracking, pseudo-labeled real video, 1000× less real data.
- **Limitation as a physics critic:** trained on real video — on generated videos with flicker/morphing, tracks silently latch onto morphing appearance → *plausible-looking trajectories of implausible objects*; 2D only (no metric scale → accelerations unknowable without calibration); **no published study quantifies its error distribution on generated content.**
- **Opportunity:** tracker-reliability characterization on generated video is an open, publishable gap.

### 4.2 TAPIR — arXiv:2306.08637
- Per-point independence — no joint rigidity reasoning, weak exactly where physics checking needs structure. *Physics-aware joint tracking (rigid/articulated-constrained) doesn't exist yet.*

### 4.3 SpatialTrackerV2 — arXiv:2507.12462 (Jul 2025)
- Feed-forward *3D* point tracking from monocular video: joint point tracking + monocular depth + camera pose → world-space trajectories decomposing scene/ego/object motion; 30% better than prior 3D trackers, ~50× faster than dynamic reconstruction.
- **Limitation:** Up-to-scale (absolute g unverifiable without scale anchor); untested on generated-video artifact statistics; weak on fast motion/blur (collisions!).
- **Opportunity:** the single most enabling tool for verification oracles — world-space observed trajectories with camera motion factored out, killing the "camera confound" objection to 2D checks.

### 4.4 TRAVL + ImplausiBench — arXiv:2510.07550 (Oct 2025)
- Fine-tuning recipe making VLMs better implausibility judges (balanced data + trajectory-aware attention); ImplausiBench removes linguistic shortcuts.
- **Limitation:** Even after TRAVL, absolute performance weak — fundamental VLM temporal/causal limits; binary verdicts, no measurement or localization.
- **Opportunity:** hybrid judges — VLM for semantics + explicit trajectory module for dynamics.

### 4.5 "Is Your Video Language Model a Reliable Judge?" — arXiv:2503.05977 (Mar 2025)
- Adding unreliable VLM judges *degrades* collective accuracy (noise dominates) — undermines ensemble-of-judges practice; calibration/weighting of physics judges is open.

### 4.6 Equation-discovery motion forecasting — arXiv:2507.06830 (Jul 2025)
- Symbolic regression extracts governing dynamics from observed trajectories → physically-grounded forecasts → conditions trajectory-guided I2V.
- **Limitation:** Brittle for multi-object contact-rich scenes; needs accurate trajectories first.
- **Opportunity:** discovered equations = *interpretable* expected-dynamics model — middle ground between full simulators and learned rewards.

### 4.7 Flow-based checks
- No 2025–26 paper survives as a dedicated optical-flow dynamics-law checker (integrating flow to acceleration amplifies noise) — the niche is effectively vacant.

---

## SYNTHESIS — strengthening the verification-oracle line (C6)

**The novelty claim that holds:** *Nobody closes the loop with measurements.*
- PSIVG / PhyRPR / PhysCtrl inject simulation **open-loop** — never check the output.
- WMReward / V-JEPA-2-reward close the loop with an **opaque scalar** — cannot localize or explain.
- PhyT2V / PhyDetEx close the loop through a **lossy VLM-text bottleneck**.
- Morpheus / PISA / PDI do trajectory-residual **measurement** — but only for benchmarking, never for selection/regen.

The intersection — **measured, interpretable, per-object residuals driving selection AND targeted regeneration, training-free** — is unoccupied.

### Five strengthening moves (each kills a named reviewer attack)

- **S1 · Probabilistic verification with system identification** (kills "your simulator is wrong, not the video"). Infer a *posterior* over physical parameters (short system-ID from early observed frames, PhysGen3D-style material estimation, class priors); roll out an *ensemble* of simulations; score candidates by whether observed trajectories fall inside the ensemble's tolerance band (Mahalanobis in trajectory space; conservation residuals are parameter-light — ratios, not absolute masses). Report "outside the 95% band of physically consistent futures," not "differs from my simulation."
- **S2 · Verifier-reliability gating** (kills "your critic fails on generated video"). Self-diagnosis layer: forward–backward tracking consistency, cross-tracker agreement (CoTracker3 vs SpatialTrackerV2 vs TAPIR), rigid-body consistency of tracked point sets, per-trajectory confidence. Emit verdicts only where tracking is certified; otherwise fall back to S3 tiers. **Bonus:** tracker disagreement is itself a plausibility cue (real rigid motion → consistent tracks; morphing objects → divergent ones). Standalone publishable: "how reliable are point trackers as critics of generative video?"
- **S3 · Hybrid neural–symbolic oracle with a verifiability router** (kills "only ballistic rigid bodies"). Route per-prompt/per-object: (a) reconstructable rigid/ballistic → simulator+tracker residual; (b) parametric clean motion → equation-discovery residual; (c) fluids/deformables/biological → learned surprise (V-JEPA 2 reward / LikePhys PPE); (d) semantic violations (teleports, count changes) → VLM judge with TRAVL-style trajectory-aware prompting. **Report which tier verified what** — partial-verification transparency converts the coverage weakness into an explicit confidence taxonomy.
- **S4 · Residual-targeted regeneration instead of blind reroll** (kills "best-of-N is just expensive rejection sampling"). Use the oracle's unique asset — a localized, signed residual (which object, which frames, which direction) — as an actuator: (i) spatio-temporal inpainting of only the violating object's tube (Self-Refining Video Sampling's machinery, aimed by an external oracle); (ii) inject the *corrected* trajectory as conditioning for the regen pass (PhysCtrl/PhyRPR-style latent fusion, Force-Prompting-style impulses); (iii) counterfactual negative guidance (SDG) instantiated from the *detected* violation. Turns N-sample search into gradient-like correction.
- **S5 · Early-exit cascaded verification + anti-circularity protocol** (kills "compute cost" and "you graded your own homework"). Cascade: (1) cheap latent/likelihood probes on partial denoising prune most of N early; (2) tracker checks on decoded keyframe windows; (3) full simulator verification only for finalists — speculative-decoding-style amortization. Anti-circularity: evaluate on independent axes (VideoPhy-2 AutoEval, Physics-IQ protocol, human study); hack-detection regularizers in the oracle (penalize degenerate slow motion, dynamism floor à la VideoPhy-2); ablate interpretable-measured reward vs WMReward-style opaque reward **at matched compute** — itself a novel experiment nobody has run.

### Residual risks to design around
- Monocular scale ambiguity for absolute-g checks → S1 conservation residuals + scale anchoring from known-size objects.
- Gains upper-bounded by base-generator sample diversity (if no candidate in N is physical, selection can't help) → S4's corrective regen is the answer; ablate explicitly.
- Runtime → report wall-clock vs quality Pareto against WMReward and PhyT2V (both conspicuously omit it).
