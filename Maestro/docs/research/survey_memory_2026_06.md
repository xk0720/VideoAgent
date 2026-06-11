# Survey — Agent Memory for Multimodal / Long-Video Generation & Understanding (June 2026)

> Raw survey, collected 2026-06-11 via WebSearch + arXiv. "Key limitation" entries
> are our own critical assessment. Feeds the consolidated innovation doc.

---

## 1. General Agent Memory Architectures

### 1.1 A-MEM — arXiv:2502.12110 (Feb 2025)
- **Method:** Zettelkasten-inspired memory: each note gets LLM-generated keywords/tags/context, dynamically linked to related notes; old notes' attributes are "evolved" when new related memories arrive.
- **Key limitation:** All structure produced by unverified LLM judgment on *text* notes — link noise compounds; no forgetting budget; no concept of visual/perceptual content or task-grounded write policy.
- **Opportunity:** Verified, modality-aware note linking; memory evolution driven by downstream task error.

### 1.2 Mem0 — arXiv:2504.19413 (Apr 2025)
- **Method:** Extracts salient facts from conversation, consolidates (ADD/UPDATE/DELETE/NOOP), retrieves at query time; Mem0ᵍ stores a relation graph. 26% LLM-judge gain over OpenAI memory on LOCOMO.
- **Key limitation:** Extraction is aggressively lossy and text-only — perceptual detail, temporal ordering, anything not a short declarative fact is discarded at write time.
- **Opportunity:** Extend extract-consolidate loops to visual/latent payloads (reference frames, embeddings) with the same CRUD discipline.

### 1.3 HippoRAG 2 — arXiv:2502.14802 (Feb 2025, ICML 2025)
- **Method:** OpenIE KG over corpus, Personalized PageRank retrieval, online LLM recognition memory. +7% associative over embedding RAG.
- **Key limitation:** Read-mostly memory over quasi-static text corpus — no principled write policy for streaming experience, no episodic timestamps, no visual identity/spatial state representation.
- **Opportunity:** PPR-style associative retrieval over a multimodal entity graph (faces, objects, locations) with online writes.

### 1.4 MemGPT → Letta + sleep-time compute — arXiv:2310.08560; arXiv:2504.13171
- **Method:** OS-inspired virtual context paging between in-context memory and external storage; sleep-time compute reorganizes/consolidates memory offline between sessions.
- **Key limitation:** Paging is brittle, token-expensive, text-blob-based; no compression of visual streams or sub-symbolic memory.
- **Opportunity:** "Sleep-time" consolidation for visual memory (re-clustering entity references, distilling episodes to schemas offline) — unexplored.

### 1.5 MemoryBank — arXiv:2305.10250 (May 2023)
- **Method:** Long-term memory with Ebbinghaus forgetting-curve retention + evolving user portrait.
- **Key limitation:** Decay is a fixed psychological prior, not learned from task utility; text-only; portrait drift.
- **Opportunity:** Utility-learned retention applied to expensive visual memory where storage actually matters.

### 1.6 MemoryOS — arXiv:2506.06326 (Jun 2025)
- **Method:** Three-tier store (short/mid/long) with OS-style management: FIFO promotion, "heat"-based segmented pages; four modules (storage, updating, retrieval, generation).
- **Key limitation:** Promotion/eviction are hand-coded heuristics tuned for chitchat; nothing handles non-text pages or task-conditioned retention.
- **Opportunity:** Same OS metaphor with learned schedulers and multimodal pages.

### 1.7 MemOS — arXiv:2507.03724 (Jul 2025)
- **Method:** Memory as schedulable system resource via **MemCube** unifying plaintext, activation (KV/hidden states), and parametric (weights/LoRA) memory, with provenance, versioning, migration, fusion.
- **Key limitation:** Systems blueprint; transitions between memory forms underspecified; MemCubes defined for language only — no visual/latent cube, no diffusion-facing interface.
- **Opportunity:** A "MemCube" equivalent for generative visual state (entity LoRAs, keyframe latents, 3D caches) scheduled across shots.

### 1.8 MIRIX — arXiv:2507.07957 (Jul 2025)
- **Method:** Six typed stores — Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault — each managed by a dedicated memory-manager agent + meta-router. Handles screenshot streams (+35% on ScreenshotVQA, 99.9% storage reduction).
- **Key limitation:** "Multimodal" = captioning screenshots into text at ingestion — pixel evidence gone, cannot support visual re-grounding ("regenerate this exact character"); fixed taxonomy; expensive routing.
- **Opportunity:** Keep typed stores but make Resource Memory hold *usable visual assets* (embeddings, crops, latents), not descriptions.

### 1.9 Memory in the Age of AI Agents: A Survey — arXiv:2512.13564 (Dec 2025)
- **Method:** forms/functions/dynamics taxonomy: factual / experiential / working memory.
- **Key limitation (field-level):** Explicitly flags multimodal memory, RL-integrated memory, trustworthiness as *open frontiers* — surveyed systems are nearly all text-conversational.
- **Opportunity:** Direct citation support that multimodal + generation-facing memory is the under-served quadrant.

*(Also: EverMemOS arXiv:2601.02163; MemR³ arXiv:2512.20237 — both text-only.)*

---

## 2. Multimodal Memory & Long-Video Understanding

### 2.1 VideoAgent (memory-augmented) — arXiv:2403.11481 (Mar 2024)
- **Method:** One offline pass builds *temporal memory* (segment event captions) + *object memory* (tracked/re-identified objects in SQL); LLM answers via retrieval tools.
- **Key limitation:** Memory constructed once, per video; no incremental update, no cross-video persistence; tracking errors baked in permanently.
- **Opportunity:** Online, self-correcting object memory; memory shared across the agent's lifetime.

### 2.2 Glance-Focus — arXiv:2401.01529 (NeurIPS 2023)
- **Method:** Glancing stage generates compact dynamic event memories that prompt the focusing stage for multi-event VideoQA.
- **Key limitation:** Small fixed-capacity latent set, per-video, non-interpretable, trained per-benchmark.
- **Opportunity:** Capacity-elastic, queryable event memory with explicit schema.

### 2.3 MA-LMM — arXiv:2404.05726 (Apr 2024)
- **Method:** Online frame processing into a long-term memory bank with similarity-based token compression.
- **Key limitation:** Averaging-style merging irreversibly destroys fine appearance detail and temporal precision; intra-inference only.
- **Opportunity:** Hierarchical compression with recoverable detail (pointers back to raw frames).

### 2.4 WorldMM — arXiv:2512.02425 (Dec 2025)
- **Method:** Three complementary memories over hours-long video — multi-scale episodic, continuously updated semantic, raw visual — with an adaptive retrieval agent. +8.4% avg over SOTA on 5 long-video QA benchmarks.
- **Key limitation:** Expensive offline per-video pass; read-only during reasoning; serves QA only — nothing reusable as generation conditioning.
- **Opportunity:** Make the episodic/semantic/visual triad write-capable and consumable by a generator.

### 2.5 VideoMem — arXiv:2512.04540 (Dec 2025)
- **Method:** Ultra-long video understanding as sequential generation with a global memory buffer; retention policy trained with Progressive GRPO — the model *learns* what to keep vs evict.
- **Key limitation:** Retention optimized purely for QA reward — task-myopic, uninterpretable, not transferable to generation consistency.
- **Opportunity:** RL-learned retention retargeted at generation-consistency metrics.

### 2.6 MM-Mem ("From Verbatim to Gist") — arXiv:2603.01455 (Mar 2026)
- **Method:** Pyramidal memory — sensory buffer → episodic stream → symbolic schema — trained with a Semantic Information Bottleneck; entropy-driven retrieval descends to fine layers only when uncertain.
- **Key limitation:** "Relevance" defined by QA supervision — gist keeps what answers questions, not what reconstructs appearance; abstraction is one-way (schema cannot decode back to pixels/conditioning).
- **Opportunity:** An *invertible* gist — compression constrained by entity-reconstruction fidelity, exactly what generation needs.

### 2.7 FlexMem — arXiv:2603.29252 (Mar 2026)
- **Method:** Training-free: MLLM visual KV caches as memory, dual-pathway compression, task-adaptive reading; 1k+ frames on one RTX 3090.
- **Key limitation:** Welded to one model's KV cache — non-persistent, non-shareable, non-symbolic, unauditable.
- **Opportunity:** Bridge KV-level memory to an external persistent store.

*(Also: StreamMeCo arXiv:2604.09000 — streaming memory compression, same QA-centric caveat.)*

---

## 3. Memory for Generation Consistency

### 3.1 WORLDMEM — arXiv:2504.12369 (Apr 2025)
- **Method:** World simulator with external memory bank of past frames + states (pose, timestamp); memory attention retrieves per generation step; reconstructs previously seen scenes across large gaps.
- **Key limitation:** Retrieval keys on explicit pose/state (cheap in Minecraft, expensive in the wild); memory units are raw frames — appearance only, no entity/narrative semantics; bank grows linearly, no consolidation.
- **Opportunity:** Pose-free semantic keys; consolidated (not raw-frame) memory units.

### 3.2 Context-as-Memory — arXiv:2506.03141 (SIGGRAPH Asia 2025)
- **Method:** All historical frames as memory via frame concatenation; Memory Retrieval prunes by camera-FOV overlap.
- **Key limitation:** FOV retrieval presumes static scene + known poses — wrong the moment objects move or characters act.
- **Opportunity:** Retrieval keyed on dynamic entity identity + scene semantics, not camera geometry.

### 3.3 Long-term Spatial Memory world models — arXiv:2506.05284 (Jun 2025)
- **Method:** Geometry-grounded 3D point-map store of the generated world with store/retrieve feeding the video world model.
- **Key limitation:** Captures *static geometry* only — dynamic agents, state changes, non-rigid content fall outside the ontology; bounded by online 3D reconstruction error.
- **Opportunity:** Layered memory: static 3D scene + dynamic entity tracks + event log.

### 3.4 HyDRA ("Out of Sight but Not Out of Mind") — arXiv:2603.25716 (Mar 2026)
- **Method:** Hybrid memory for dynamic video world models: compressed memory tokens + spatiotemporal relevance-driven retrieval so subjects exiting frame retain identity/motion on re-entry; HM-World benchmark (59K clips).
- **Key limitation:** Short horizons, subject-level appearance/motion only; no narrative state, no multi-shot/cut handling.
- **Opportunity:** Extend exit-entry identity persistence to multi-shot, cut-heavy narrative video.

### 3.5 StoryMem — arXiv:2512.19539 (Dec 2025)
- **Method:** Multi-shot storytelling as iterative shot synthesis conditioned on a compact, dynamically updated keyframe memory bank, injected into a single-shot diffusion model via latent concat + negative RoPE shifts (LoRA fine-tune); semantic keyframe selection.
- **Key limitation:** Memory = a handful of keyframes — purely appearance-level, heuristic selection; cannot represent *state change* (wet, wounded, changed clothes) vs *identity* — either over-freezes appearance or drifts; minute-scale.
- **Opportunity:** Separate canonical-identity memory from current-state memory; learned keyframe write policy.

### 3.6 VideoMemory — arXiv:2601.03655 (Jan 2026)
- **Method:** Entity-centric multi-agent framework: Dynamic Memory Bank of explicit visual+semantic descriptors per character/prop/background, retrieved to condition keyframe + video synthesis, *updated after each shot*.
- **Key limitation:** 54-case self-built benchmark; the update step is LLM/VLM-described, **unverified against generated pixels** — errors written into memory propagate forward.
- **Opportunity:** Verification-gated memory writes (only commit updates a critic confirms in the rendered output).

### 3.7 EntityBench / EntityMem — arXiv:2605.15199 (May 2026)
- **Method:** 140 episodes from real narrative media with per-shot entity schedules (recurrence gaps up to 48 shots); EntityMem generates and *verifies* isolated per-entity visual+textual references in a persistent bank *before* generation, then reuses them.
- **Key limitation:** References frozen pre-generation — cannot model story-driven appearance evolution, cannot ingest new entities mid-generation; verification once, not continuous.
- **Opportunity:** Combine EntityMem's verified canonical refs with VideoMemory-style per-shot state updates (the static-vs-evolving gap is explicitly open).

### 3.8 Genie 3 — DeepMind blog, Aug 2025 (no arXiv)
- **Method:** Real-time interactive world model; consistency emerges with *implicit* visual memory reaching back ~1 minute.
- **Key limitation:** Memory is emergent AR context — fixed-horizon, non-inspectable, non-editable, unscalable to hour-long narratives.
- **Opportunity:** Strong evidence implicit memory saturates; explicit external memory is the scaling path.

### 3.9 ConsisID — arXiv:2411.17440 (CVPR 2025 Highlight)
- **Method:** Tuning-free identity-preserving T2V on DiT via frequency-decomposed facial identity signals injected at different depths.
- **Key limitation:** Single human-face identity within a single shot — conditioning technique, not memory: nothing persists or updates.
- **Opportunity:** Frequency-decomposed ID features as the *storage format* inside an entity memory bank (compact, pose-invariant keys).

---

## 4. Cross-Session Personalization Memory for Creative Agents

### 4.1 Learning User Preferences for Image Generation — arXiv:2508.08220 (Aug 2025)
- MLLM preference capture, contrastive preference loss, learnable preference tokens. **Limitation:** simulated users, static profiles, image-only — no temporal/cinematic preference dimensions (pacing, shot grammar).

### 4.2 MultiSessionCollab — arXiv:2601.02702 (Jan 2026)
- Per-session reflection writes preference info to persistent memory injected next session. **Limitation:** text tasks; memory capped at prompt size; simulator-derived signals.

### 4.3 Premier — arXiv:2603.20725 (Mar 2026)
- Per-user learnable embedding modulating a T2I generator. **Limitation:** per-user gradient training (cold start, drift), non-agentic, image-only.

### 4.4 Learning Personalized Agents from Human Feedback — arXiv:2602.16173 (Feb 2026)
- Clarify-before-acting → ground in retrieved preferences → post-action feedback integration with drift detection. **Limitation:** text memory; relies on explicit feedback, not implicit signals (regenerations, abandons) that dominate creative workflows.
- **Opportunity:** The clarify→ground→update loop is the right control flow for a creative director-agent; nobody has instantiated it for video generation.

*(Also: Discrete Preference Learning for Personalized Multimodal Generation, arXiv:2604.20434.)*

---

## 5. Gap Map

| | Text/agent memory | Visual memory |
|---|---|---|
| **Understanding** | Mature (A-MEM, Mem0, MIRIX, MemOS…) | Emerging (WorldMM, MM-Mem, VideoMem) — read-only, per-video, QA-rewarded |
| **Generation** | Essentially absent | Nascent (StoryMem, EntityMem, WorldMem, HyDRA) — appearance-level, heuristic writes, single-project |

Structural gaps no paper covers:
1. **No closed read–write loop between perception and generation.** Generation memories store what was *intended*, never what was *verified in the output*; understanding memories never condition a generator.
2. **No identity-vs-state separation.** EntityMem freezes references (can't evolve); VideoMemory/StoryMem evolve freely (can drift). Nobody factorizes canonical identity (immutable) from mutable state with a transition log.
3. **All retention policies in generation memory are heuristic.** VideoMem proved retention can be RL-learned — but only for QA reward.
4. **All generation memory dies with the project.** Zero cross-video persistence; the personalization literature has never been connected to a video generator.
5. **Compression is one-way.** Gist abstraction can't be decoded back to conditioning signals; generation needs *invertible* memory.

## Candidate Innovation Angles

- **Angle 1 — Verification-gated memory writes ("commit only what rendered").** VLM critic compares each generated shot against memory; entity-state updates committed only when confirmed in pixels; discrepancies trigger regen or an explicit memory correction entry. Attacks the dominant failure mode (drift accumulation); measurable on EntityBench long-gap splits.
- **Angle 2 — Dual-register entity memory: canonical identity ⊕ evolving state with a typed transition log.** Immutable identity register (ConsisID-style frequency features + verified reference crops) + mutable state register (costume, damage, emotion, location, lighting) updated via discrete logged transitions authored by the director agent. Gives continuity *auditability*.
- **Angle 3 — Generation-rewarded learned retention.** RL write/evict policy where reward = cross-shot consistency metrics (entity similarity on re-entry, scene-revisit fidelity), not QA. Clean unclaimed objective; compute-heavy.
- **Angle 4 — Cross-project creative memory.** MIRIX-style typed store at *user* level: recurring characters, style schemas, preference memory updated from implicit signals (kept vs regenerated takes) via clarify→ground→update. Strongest product moat, lowest architectural risk.
- **Angle 5 — Layered hybrid world memory** (3D scene register + dynamic entity tracks + symbolic event log) with **bidirectional grounding** — every symbolic node keeps pointers to retrievable latents/frames so plot-level memory decodes back into conditioning. Highest novelty, highest integration risk.

**Recommended wedge:** Angles 1+2 ("self-verifying dual-register entity memory"), evaluable on EntityBench + StoryMem's setting, with Angle 4 as the system-level differentiator.
