# Survey — Skill Libraries in Agents + Audio-Video Joint Generation (June 2026)

> Raw survey, collected 2026-06-11 via WebSearch + arXiv. Feeds the consolidated
> innovation doc. Critical "Key limitation" entries are our own assessment.

---

## TOPIC A1 — Skill Acquisition / Skill Libraries in Agents

### 1. Voyager — arXiv:2305.16291 (May 2023)
- **Method:** First LLM-powered lifelong-learning agent in Minecraft. Maintains an ever-growing **skill library of executable JS code**, indexed by embedding of the skill description; new skills are written by GPT-4, verified by execution feedback, and composed into more complex skills via an automatic curriculum.
- **Key limitation:** Skills are code against a *programmatic, deterministic* API (Mineflayer). Verification = "did the code run / did inventory change" — a binary, cheap oracle. This does not transfer to domains where outcomes are perceptual/aesthetic (e.g., "did this edit look good"), where no cheap verifier exists.
- **Opportunity:** Skill libraries with *perceptual* (VLM-judged) verification instead of execution-success verification.

### 2. Agent Workflow Memory (AWM) — arXiv:2409.07429 (Sep 2024)
- **Method:** Induces reusable **workflows** (common sub-routine action sequences in natural language/abstracted steps) from past agent trajectories, stores them in memory, and injects them into context for future web tasks. Works online (from own successes) or offline. +24.6% on Mind2Web, +51.1% on WebArena.
- **Key limitation:** Workflows are *textual action traces*, not parameterized callable procedures — no type signature, no compositional contract, no failure handling; induction assumes task distributions repeat (websites), and quality degrades when self-judged successes are wrong (no external verifier).
- **Opportunity:** Typed, parameterized workflow extraction with explicit pre/post-conditions; cross-domain workflow transfer.

### 3. SkillWeaver — arXiv:2504.07079 (Apr 2025)
- **Method:** Web agents **self-improve by discovering and honing skills**: autonomously explore a website, propose candidate skills, practice them, and distill into robust, plug-and-play **APIs (code)** with tests. +31.8% relative on WebArena; skills transfer from strong→weak agents (+54.3%).
- **Key limitation:** Exploration-driven discovery presumes a *bounded, revisitable* environment (one website) where repeated practice is cheap and idempotent. In generation-tool domains (video/audio models) every "practice" call costs minutes of GPU and outputs are stochastic — the practice-until-robust loop is economically infeasible as-is.
- **Opportunity:** Skill honing under expensive, stochastic, non-resettable tool calls (few-shot skill validation; caching exemplar outcomes).

### 4. Gödel Agent — arXiv:2410.04444 (Oct 2024, ACL 2025)
- **Method:** Self-referential agent that recursively **modifies its own code/logic** (including the modification logic itself), guided only by high-level objectives, with monkey-patching of its runtime. Outperforms hand-designed agents on coding/science/math tasks.
- **Key limitation:** Self-modification is unconstrained and unsafe; improvements are noisy, hard to attribute, and evaluated only on cheap-to-verify benchmarks; no persistent, transferable artifact like a skill library emerges — gains are entangled with the specific run.
- **Opportunity:** Restricting self-evolution to a *structured skill layer* (audit-able, versioned) rather than arbitrary code self-rewrites.

### 5. A Survey of Self-Evolving Agents — arXiv:2507.21046 (Jul 2025)
- **Method:** First systematic survey organizing self-evolution along **what/when/how to evolve** — covering parametric (training) and non-parametric (prompt, memory, toolset) evolution, single- vs multi-agent, and domain cases.
- **Key limitation (of the field, per survey):** Evaluation of self-evolution is ad hoc; almost all works evolve *either* memory *or* tools *or* prompts, rarely a unified skill abstraction; safety/regression control of evolved components is open.
- **Opportunity:** A unified "skill" abstraction spanning memory + tool + workflow, with regression testing.

### 6. MemSkill — arXiv:2602.02474 (Feb 2026)
- **Method:** Reframes static memory operations as **learnable, evolvable "memory skills"** — structured routines for extracting/consolidating/pruning information. A controller selects skills, an LLM executor applies them, and a "designer" periodically evolves underperforming skills. Gains on LoCoMo, LongMemEval, HotpotQA, ALFWorld.
- **Key limitation:** Skills operate on *text memory management*, not on external tool orchestration; evolution signal comes from QA-style benchmarks with ground truth — again absent in creative domains.
- **Opportunity:** Apply the controller/executor/designer skill-evolution loop to multimodal tool pipelines.

### 7. AutoSkill — arXiv:2603.01145 (Mar 2026)
- **Method:** Model-agnostic **plugin layer for lifelong skill self-evolution**: derives skills from dialogue/interaction traces, maintains them through continual refinement, and dynamically injects them into future requests without retraining; skills are shareable across agents/users/tasks.
- **Key limitation:** Skills are abstractions of *conversational/preference* experience (largely declarative), not validated executable procedures; no mechanism reported for verifying a skill is actually correct before reuse (risk of consolidating bad habits).
- **Opportunity:** Verified skill admission ("skill CI") before library insertion.

### 8. Surveys: Adaptation of Agentic AI — arXiv:2512.16301 (Dec 2025); Externalization in LLM Agents — arXiv:2604.08224 (Apr 2026)
- **Method:** Both taxonomize agent adaptation, explicitly separating **skills** (reusable procedural artifacts, e.g., SKILL.md / code skills) from memory and post-training; 2604.08224 frames skills/memory/protocols as "externalization" of capability out of weights.
- **Key limitation noted:** Skill ecosystems are dominated by *text/web/coding* agents; essentially **no work instantiates skill libraries for multimodal generation pipelines**; skill granularity and naming/retrieval remain folklore.
- **Opportunity (explicit gap):** Skill acquisition for perceptual/creative tool-use agents.

---

## TOPIC A2 — Video-Creation/Editing Agents and Their Tool Libraries

### 9. UniVA — arXiv:2511.08521 (Nov 2025)
- **Method:** Open-source omni-capable video generalist: **Plan-and-Act dual-agent** + modular **MCP tool servers** (understanding/segmentation/editing/generation), plus 3-level memory (global tool stats, task artifacts, user prefs). Introduces UniVA-Bench.
- **Audio handling:** post-hoc tools via MCP — `music_gen`, `video_foley` (ThinkSound V2A), `speech_gen`/`voice_clone`. Fire-and-forget: no audio-visual consistency check after generation.
- **Skills vs tools:** **Tools are entirely fixed/hand-registered.** Memory provides continuity but does *not* acquire new skills or workflows.
- **Key limitation:** No learning loop — every task re-plans from scratch; identity drift in reference-based generation; audio tools invoked open-loop.
- **Opportunity:** Workflow/skill induction over its own MCP traces; closed-loop AV review.

### 10. CutClaw — arXiv:2603.29664 (Mar 2026)
- **Method:** Autonomous multi-agent framework editing **hours-long raw footage into music-synchronized montages**. Structured captions (Qwen3-VL, PySceneDetect, Whisper); **madmom** hierarchical music keypoints; Playwriter anchors narrative to musical structure, Editor grounds clips, Reviewer enforces aesthetic/continuity constraints. SOTA on Visual Quality / Instruction Following / AV Harmony.
- **Audio handling:** **Music is user-provided/retrieved, never generated**; music is the invariant temporal anchor — video is cut to fit audio.
- **Skills vs tools:** All capabilities are **hardcoded tools + fixed agent roles**; editing knowledge lives in prompts, re-derived per run.
- **Key limitation:** Cannot generate music/SFX; no generative visual effects; MLLM context limits on dense hour-long footage; Reviewer checks visuals against the plan, not AV *semantic* fit.
- **Opportunity:** Inverting the direction (music conditioned on edited video); distilling prompt-borne editing heuristics into reusable skills.

### 11. FilmAgent — arXiv:2501.12909 (Jan 2025)
- **Method:** Multi-agent film automation **inside 3D virtual spaces**: director/screenwriter/actor/cinematographer collaborate via Critique-Correct-Verify and debate. Human eval 3.98/5.
- **Key limitation:** Confined to pre-built Unity 3D scenes with finite action/camera vocabulary — no real footage, no generative video, audio limited to TTS; "skills" are role prompts, zero persistence.
- **Opportunity:** Porting multi-role critique loops to real generative/asset-based pipelines.

### 12. MovieAgent — arXiv:2503.07314 (Mar 2025)
- **Method:** Long-form movie generation via **multi-agent CoT planning** over script + character bank; hierarchical scene/shot decomposition, multi-shot character consistency, subtitles, audio.
- **Key limitation:** Audio = TTS narration stitched post-hoc; no music score, no foley, no AV alignment evaluation; fixed cascade — no reflection on output quality, no skill memory.
- **Opportunity:** Quality-feedback loops and a real soundtrack stage.

### 13. Crayotter — arXiv:2606.07636 (May 2026)
- **Method:** Traceable multi-agent long-form video editing: coverage-aware material prep → artifact-based "editing research" → **tool-grounded timeline execution** with 21 registered tools (incl. loudness normalize, `duck_background_audio`); externalizes all state into inspectable artifacts.
- **Key limitation:** Audio = mixing-level only; tools fixed; no ablations isolating module contributions.
- **Opportunity:** Its artifact/trace externalization is exactly the substrate AWM-style skill induction could mine — unexploited.

**Pattern across A2:** every video agent (UniVA, CutClaw, FilmAgent, MovieAgent, Crayotter) treats capabilities as **static hand-registered tools + role prompts**. None has Voyager/SkillWeaver-style skill acquisition; procedural editing knowledge is trapped in prompts and is re-derived every run.

---

## TOPIC B3 — Joint Audio-Video Generation & V2A / V2Music

### 14. Veo 3 / 3.1 — no arXiv (Google DeepMind, May/Oct 2025)
- **Method:** First production model with **native joint audio**: latent DiT attending over unified visual+audio token sequence at every denoising step → lip-synced dialogue, foley, ambience, music.
- **Key limitation:** Closed, ~8s clips, expensive; cannot add audio to *existing/edited* footage; no tech report.
- **Opportunity:** Agents orchestrating Veo-3-like clips still need cross-clip music continuity — per-clip native audio ≠ coherent long-form score.

### 15. Movie Gen (Audio) — arXiv:2410.13720 (Oct 2024)
- **Method:** 30B video model + **13B Movie Gen Audio**: flow-matching V2A/T2A generating 48kHz synced SFX + music, with audio extension to multi-minute coherence.
- **Key limitation:** Cascaded (audio can't influence visuals); unreleased; long-form narrative scoring unevaluated; no dialogue.
- **Opportunity:** Open replication of long-form audio extension; agentic control of T2A conditioning.

### 16. MM-Diffusion — arXiv:2212.09478 (Dec 2022, lineage anchor)
- Coupled U-Nets, random-shift cross-attention, joint AV denoising. Toy domains, no text control.

### 17. AV-DiT — arXiv:2406.07686 (Jun 2024)
- Frozen image DiT + lightweight AV adapters; audio as spectrogram "images." Toy benchmarks, no text conditioning.

### 18. UniForm — arXiv:2502.03897 (Feb 2025)
- **Single unified DiT** denoising audio+video in one shared latent, task tokens for T2AV/A2V/V2A.
- **Key limitation:** Underperforms specialists; shared latent forces compromises.
- **Opportunity:** Evidence one-model-for-all is feasible but not winning — argues for agentic orchestration of specialists.

### 19. JavisDiT — arXiv:2503.23377 (Mar 2025; v2 Feb 2026)
- Joint AV DiT with **Hierarchical Spatio-Temporal Synchronized Prior**; introduces **JavisBench** (10,140 videos) + new sync metric.
- **Key limitation:** Auxiliary prior model; trails cascaded SOTA on complex scenes; seconds-long clips.
- **Opportunity:** Benchmark/metric directly reusable for evaluating *agent-assembled* AV content.

### 20. Ovi — arXiv:2510.01284 (Oct 2025)
- **Twin-DiT blockwise fusion**: audio tower identical to video DiT, scaled-RoPE timing exchange + bidirectional cross-attention → 5s synced AV without post-hoc alignment.
- **Key limitation:** 5s; doubles parameters; weak music; can't score existing footage.
- **Opportunity:** Strongest *open* joint-AV checkpoint — candidate "talking/SFX shot" tool, useless for long-form scoring.

### 21. Seeing and Hearing — arXiv:2402.17723 (CVPR'24)
- Training-free bridging of off-the-shelf video and audio generators via **ImageBind gradient guidance**; V2A, A2V, joint.
- **Key limitation:** Clip-level semantic alignment only — no fine temporal sync; slow; degrades quality.
- **Opportunity:** Closest ancestor of an *agentic* AV consistency loop (replace gradients with VLM critique + re-prompt).

### 22. FoleyCrafter — arXiv:2407.01494 (Jul 2024)
- Frozen T2A + semantic adapter + temporal controller (onset detection). Coarse sync; superseded by MMAudio/ThinkSound.

### 23. MMAudio — arXiv:2412.15322 (Dec 2024; CVPR 2025)
- **Multimodal joint training** (video-audio + large text-audio), frame-level sync module. 157M params, 8s audio in 1.23s, SOTA V2A.
- **Key limitation:** 8s window — chunking has no cross-chunk acoustic continuity (ambience/music jumps at seams); foley-oriented, weak music.
- **Opportunity:** Default open V2A tool; *chunk-seam continuity* is an open agent-level problem.

### 24. ThinkSound — arXiv:2506.21448 (NeurIPS 2025)
- **Chain-of-Thought V2A**: MLLM reasoning (AudioCoT) conditions flow-matching audio gen across three stages — overall foley → object-centric refinement → instruction-based editing. SOTA on VGGSound + MovieGen Audio bench.
- **Key limitation:** CoT bounded by MLLM video understanding (misses off-screen sounds); refinement presumes human-in-loop; clip-level.
- **Opportunity:** Its **editing-by-instruction interface is the natural actuator for an agent's audio-review loop** (critic finds flaw → instructs ThinkSound edit). UniVA wraps it, but open-loop.

### 25. HunyuanVideo-Foley — arXiv:2508.16930 (Aug 2025)
- REPA-style representation alignment, 100k-hour corpus, high-fidelity foley. Foley only (no music/speech), per-clip.

### 26. VidMuse — arXiv:2406.04321 (CVPR 2025)
- Video-to-**music** with Long-Short-Term visual modeling, 360K video-music pairs.
- **Key limitation:** Aligns to global mood/coarse rhythm, not narrative beats or cut points; AR drift over minutes; no user style constraints.
- **Opportunity:** Beat/cut-aware conditioning and agent-supplied textual music briefs.

### 27. Video2Music — arXiv:2311.00968; M2UGen — arXiv:2311.11255 (Nov 2023)
- Symbolic MIDI chord/emotion matching; LLaMA-bridge to MusicGen. Both show *semantic* video→music mapping; neither solves *temporal* mapping.

### 28. V2M-Zero — arXiv:2603.11042 (Mar 2026)
- **Zero-pair, time-aligned video-to-music**: per-modality "event curves" (when/how much change); fine-tune T2M on music-event curves, swap in video-event curves at inference. +21-52% temporal sync, +28% beat alignment.
- **Key limitation:** Event curves capture change *magnitude*, not semantics — needs a text brief from elsewhere, i.e. *assumes an upstream director*.
- **Opportunity:** Perfect downstream tool for an agent that writes the music brief (semantics) while V2M-Zero handles timing — a claimable division of labor.

---

## TOPIC B4 — How Agents Handle Audio Today + AV Evaluation

**Current agent practice (all post-hoc, all open-loop):**
- **UniVA:** post-hoc V2A (ThinkSound) + T2M + TTS via MCP; **no verification** that audio matches video; music and foley independent, no mixing logic.
- **CutClaw:** **music retrieval only**; solves sync by *cutting video to the music* (the inverse problem); Reviewer checks visual continuity, not AV semantics.
- **Crayotter:** mixing-level only (loudness, ducking).
- **MovieAgent / FilmAgent:** TTS dialogue/narration only.
- **WavJourney** — arXiv:2307.14335 (Jul 2023): LLM writes a structured **audio script** (speech/music/SFX layers with timing) compiled into TTS/T2M/T2A calls — earliest "audio director agent," but text-driven only (never *looks* at video). **Audio-Agent** (arXiv:2410.03335) similar, no closed-loop quality check.
- **Conclusion:** No published agent closes the loop: *generate audio → perceive the muxed result → critique AV semantic/temporal fit → re-brief the audio tool*.

**Evaluation instruments:**
- **AV-Align** (TempoTokens, arXiv:2309.16429, AAAI'24): IoU between audio energy-onset peaks and optical-flow motion peaks — cheap, training-free temporal sync metric. *Limitation:* motion-onset premise fails for music (beats ≠ motion) and ambient audio; gameable.
- **ImageBind score** (since 2402.17723): clip-level AV semantic cosine. No temporal resolution; saturates.
- **JavisBench metric** (2503.23377): finer-grained sync for diverse scenes. **VidAudio-Bench** (arXiv:2604.10542, Apr 2026): V2A/VT2A across four audio categories — finds MMAudio best temporal sync + text consistency, ThinkSound/AudioX best fidelity → **no single V2A model dominates**, arguing for agentic model routing.

---

## SYNTHESIS

### (a) What's missing in how video agents define/learn SKILLS

1. **Total absence of skill acquisition in the video-agent literature.** The skill-library line (Voyager → AWM → SkillWeaver → AutoSkill/MemSkill) and the video-agent line (UniVA, CutClaw, Crayotter, FilmAgent, MovieAgent) have **zero intersection** as of June 2026. Every video agent ships static hand-registered tools and role prompts; procedural knowledge ("cut on downbeat, hold 2s after dialogue, duck music −12dB under speech") is re-derived per run inside prompts and discarded.
2. **The skill abstraction itself doesn't fit:** existing skills are code with execution-success verifiers (Voyager/SkillWeaver) or text workflows from repeated cheap trajectories (AWM). Video tool calls are expensive, stochastic, and judged perceptually — no published mechanism for (i) verifying a candidate video-editing skill (needs VLM-as-judge admission tests), (ii) honing skills when each practice call costs GPU-minutes, or (iii) parameterizing skills over *assets* (footage, character banks, music tracks) rather than over web/API arguments.
3. **Concrete open contribution:** a video agent that mines its own externalized traces into a **versioned, VLM-verified skill library** — e.g., "montage-with-beat-sync(footage, track, mood)" distilled once and reused — with regression checks against AV-Align/VLM scores. Surveys (2512.16301, 2604.08224) explicitly flag multimodal-pipeline skills as unoccupied territory.

### (b) Most defensible audio strategy for a training-free long-video agent

**Architecture: post-hoc, multi-model, agent-routed — joint generation is not viable for agents.** Joint models (Veo 3, Ovi, JavisDiT, UniForm) only produce audio for video *they themselves generate*, at 5-8s scale, and can't score existing/edited/asset-based footage. VidAudio-Bench shows no single V2A model wins everywhere, so *routing is itself the contribution*.

**Best open stack:**
- **Foley/SFX:** MMAudio (speed+sync) or HunyuanVideo-Foley (fidelity); **ThinkSound specifically because its instruction-editing stage gives the agent an actuator for revisions.**
- **Music:** (i) agent writes a semantic **music brief** → text-to-music; (ii) **V2M-Zero** for temporal alignment via event curves, agent supplies the semantics it lacks; or CutClaw-style retrieval + cut-to-beat when user supplies music.
- **Speech:** TTS/voice-clone. **Mixing:** deterministic ducking/loudness tools.

**Novel, defensible agent-level contributions (none published):**
1. **Audio-visual consistency review loop:** mux → VLM/audio-LLM critic scores semantic fit per shot + AV-Align/JavisBench temporal check → structured critique → re-invoke ThinkSound editing or re-brief the music model. Agentic analogue of Seeing-and-Hearing's guidance, training-free at orchestration level.
2. **Cross-shot acoustic continuity management:** all V2A models ≤8-10s; plan ambience beds, music sections, transition points *across* chunk seams — a limitation every clip-level model shares and no agent handles.
3. **Sound design as a planning stage:** LLM "sound director" producing a timestamped audio script (dialogue/foley/ambience/music layers with levels and sync anchors) compiled to routed model calls — WavJourney's idea, but video-grounded and critic-verified.
4. **Evaluation:** AV-Align (temporal), ImageBind/JavisBench (semantic), VidAudio-Bench categories — noting AV-Align is invalid for music (use beat-alignment metrics from V2M-Zero/VidMuse).
