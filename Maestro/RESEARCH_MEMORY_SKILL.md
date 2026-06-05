# Research note — agentic memory and skill paradigms, and the Maestro-specific gap

> Long task: extend Maestro's innovation surface beyond C1-C6 by importing the
> agentic **memory** and **skill** research of the last 18 months, *then*
> identifying what is genuinely new when these patterns meet our four hard
> constraints (training-free, multimodal, physics-grounded, self-improving).
>
> Deliverable: §1-§3 are landscape survey; §4-§5 propose two Maestro-specific
> innovations (**C7 PhysicsTyped Skill Library** and **C8 Multi-Layer Memory**);
> §6 is an implementation plan that lands them in v0.3 without disturbing
> C1-C6 or breaking the 72-passing test suite.

---

## 1. Memory landscape (2024 → 2026)

### 1.1 Canonical 4-tier taxonomy

Drawn from cognitive science and now the de-facto reference frame:

| Tier | What it holds | LLM-agent realisation |
|---|---|---|
| **Working** | live context window | conversation buffer, current tool returns |
| **Episodic** | "what happened, when" | session logs, full trajectories, decision traces |
| **Semantic** | facts, distilled patterns | "customers with X get Y"; rule libraries |
| **Procedural** | how to do tasks (skills) | skill libraries, fine-tuned policy heads |

Refs:
- [Designing Agentic Memory in 2026](https://thenuancedperspective.substack.com/p/designing-agentic-memory-in-2026)
- [Atlan — Types of AI Agent Memory](https://atlan.com/know/types-of-ai-agent-memory/)
- [Memory for Autonomous LLM Agents survey, arXiv:2603.07670](https://arxiv.org/html/2603.07670v1)
- [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

The survey consensus: **most production systems implement two tiers well; the
four-tier integration is the aspiration**. That is the gap Maestro can fill in
a *task-specific* way — see §4.

### 1.2 Recent landmarks (data-structure innovations)

- **A-MEM** ([arXiv:2502.12110, NeurIPS 2025](https://arxiv.org/abs/2502.12110)) —
  **dynamic Zettelkasten**. Each memory is a *structured note* (description +
  keywords + tags). New writes *evolve* neighboring notes' attributes and
  links — the network self-organises rather than relying on fixed insertion
  rules. Beats SOTA on six foundation models.
- **HippoRAG / HippoRAG 2** ([OpenReview](https://openreview.net/forum?id=hkujvAPVsg))
  — **KG + Personalized PageRank** mimicking hippocampus/neocortex. Improves
  associative-QA F1 by +7 over embedding retrievers; non-parametric continual
  memory.
- **MemoryBank** — applies **Ebbinghaus forgetting curve** to active recall and
  decay; autonomous summon + update.
- **G-Memory** ([arXiv:2506.07398](https://arxiv.org/pdf/2506.07398)) —
  three-tier graph hierarchy (**insight / query / interaction**) for *multi-agent*
  systems, inspired by organisational memory theory.
- **MemOS** ([memtensor MemOS_0707.pdf](https://statics.memtensor.com.cn/files/MemOS_0707.pdf))
  — a "memory OS": **slicing, tagging, hierarchical mapping, context binding**,
  per-role memory views, auto-archive on task completion.
- **MIRIX** ([arXiv:2507.07957](https://arxiv.org/pdf/2507.07957)) — multi-agent
  memory system with explicit message routing.
- **HiAgent** ([ACL 2025](https://aclanthology.org/2025.acl-long.1575.pdf)) —
  **working-memory chunking by subgoals**, summarisation on subgoal completion.
- **Me-Agent** ([arXiv:2601.20162](https://arxiv.org/html/2601.20162)) —
  **two-level habit learning + hierarchical preference memory**, +39.7 % task
  completion over baseline.
- **Mem0** ([state-of-the-art blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
  — production-grade open memory service; "meaningful step toward AI agents
  that maintain long-term context".

### 1.3 Video-specific memory (the part nobody else surveys together)

- **VideoMemory** ([arXiv:2601.03655](https://arxiv.org/abs/2601.03655)) —
  multi-agent narrative decomposition + **Dynamic Memory Bank** of explicit
  visual+semantic descriptors for **characters / props / backgrounds**;
  updated after each shot to reflect story-driven changes while preserving
  identity. Eval on a 54-case multi-shot consistency benchmark.
- **MemoryPack + Direct Forcing** ([arXiv:2510.01784](https://arxiv.org/pdf/2510.01784))
  — packs textual + image global guidance, single-step approximation curbs
  error propagation, **minute-level temporal consistency**, linear complexity.
- **PSIVG** ([arXiv:2603.06408](https://arxiv.org/html/2603.06408)) — physical
  simulator **in-the-loop** of video diffusion; 4D scene + foreground meshes
  initialised in a real simulator. (Adjacent to Maestro's C1 sketch but with
  online sim, not pre-computed control signal.)
- **Long Video Agent (Novi AI)** — production "director" with global scene
  memory, ~5-minute videos with lighting/weather/background stability.

---

## 2. Skill paradigm landscape

### 2.1 The Voyager line — "skills as compositional executable code"

- **Voyager** ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) — three
  pillars: **automatic curriculum**, **ever-growing skill library** (executable
  code), **iterative prompting** with env feedback. 3.3× more items, 15.3×
  faster milestones, generalises to a fresh Minecraft world by reusing the
  library. Skills are **temporally extended, interpretable, compositional**.
- **SkillWeaver** ([arXiv:2504.07079](https://arxiv.org/pdf/2504.07079)) —
  web agents. **Three-stage pipeline**: (1) explore env to *propose* novel
  skills, (2) practise → convert to reusable APIs, (3) test/debug for
  reliability. +25→38 % WebArena success rate, +39.8 % on real sites.
- **SkillFoundry** ([arXiv:2604.03964](https://arxiv.org/html/2604.03964)) —
  self-evolving skill libraries from *heterogeneous scientific resources*.
- **SkillAdaptor** ([arXiv:2606.01311](https://arxiv.org/html/2606.01311)) —
  self-adapting skills from past trajectories (no env interaction needed).
- **SkillOps** ([arXiv:2605.13716](https://arxiv.org/html/2605.13716)) —
  **lifecycle management**: skill libraries as self-maintaining software
  ecosystems — health, deprecation, dependency tracking.
- **RL for self-improving skill library** ([arXiv:2512.17102](https://arxiv.org/pdf/2512.17102))
  — RL-based skill acquisition layered on top of Voyager's library paradigm.

### 2.2 Physics-shaped skill embeddings (the other line)

- **ASE — Adversarial Skill Embeddings** ([SIGGRAPH 2022](https://dl.acm.org/doi/abs/10.1145/3528223.3530110))
  — large-scale **reusable** skill embeddings for physically simulated
  characters; combines adversarial imitation + unsupervised RL; controllable
  representation for downstream tasks.

This is the *closest precedent for what Maestro needs*: physically-grounded,
reusable skill abstractions. ASE is animation/control oriented — not paired
with a self-improving neural video generator. The bridge we'll build in §4
plugs that gap.

### 2.3 What none of them have

A common shape across Voyager / SkillWeaver / SkillFoundry / SkillOps:

```
skill = (name, code, description, retrieval_embedding, success_history)
discovered = "the agent rehearsed it and it worked"
retrieved = "text similarity at task time"
```

**No skill paradigm in the literature is:**
1. *Physics-typed* — keyed on which physical-failure modes the skill resolves.
2. *Verifier-monotonic-distilled* — discovered because a monotonic
   improvement Verifier (M3) accepted at the cheapest tier with non-trivial
   initial defects. (Voyager uses environment reward; SkillWeaver uses
   rehearsal repeatability; neither has a Verifier of physical-failure modes.)
3. *Coupled to a per-mode failure lesson library* — a Maestro skill **carries
   pointers** to LessonLibrary entries that occurred while developing it, so
   retrieving the skill auto-injects its known failure-mode lessons.
4. *Composable along the time axis of a single shot* — skills concatenate to
   form a longer clip with sketch-level handoff frames (closer to ASE's
   embedded animation skills than to Voyager's discrete code blocks).

These four properties are the **task-specific** opportunity §4 develops.

---

## 3. Maestro's current memory & skill surface — and what's missing

### 3.1 Current state (v0.2.2)

| Subsystem | What it does today | Memory tier |
|---|---|---|
| `AssetMemory` (`pipeline/understand.py`) | per-run shots / identities / styles / music | working (transient) |
| `LessonLibrary` (`memory/lesson_library.py`) | JSONL persistent lesson; bag-of-tokens cosine retrieval; C4 distillation = first resolved mode | semantic (partial) |
| `TrajectoryLogger` (`trajectory.py`) | full JSONL of every agent action | episodic (written but **never replayed**) |
| Conversation buffer in pipeline | implicit, transient | working |
| **— nothing —** | — | **procedural / skills** |

### 3.2 Specific gaps against the literature

| Gap | Reference work that exposes it | Why it matters for Maestro |
|---|---|---|
| Episodic memory is written, never queried | A-MEM / HippoRAG / Mem0 | We're throwing away the most valuable signal: which interventions worked vs. didn't, *per-task* |
| Lesson library is flat — no link between lessons, no evolution | A-MEM Zettelkasten | New lessons can't refine related-but-not-identical existing ones |
| Cross-run **entity** persistence is zero | VideoMemory Dynamic Memory Bank | The same hero regenerated on Day 2 starts from scratch; no face/style anchor carries over |
| No skill objects at all | Voyager / SkillWeaver / ASE | HSI tiers are hardcoded; recipes that worked once are not consolidated; identical recipes are re-derived next run |
| No hierarchical scoping (global / task / user) | G-Memory / MemOS / Me-Agent | One global LessonLibrary pollutes across users / projects |
| No user-preference memory | Me-Agent | Maestro keeps no record of "this user always likes static-shot slow-motion physics" |
| No forgetting / lifecycle | SkillOps / MemoryBank | Stale lessons accumulate; no way to prune deprecated patterns |

---

## 4. Two task-specific innovations

The line we draw is deliberate: **borrow the structural patterns** (4-tier
taxonomy, dynamic linking, lifecycle), **innovate on the dimensions where
video generation + physics grounding + monotonic Verifier give us levers no
prior work has**.

### 4.1 — C7 · PhysicsTyped Skill Library ("compiled shot recipes")

A **Skill** in Maestro is *not* executable code (Voyager) and *not* a web API
(SkillWeaver). It is a **structured plan template** carrying everything needed
to reconstruct a shot that the Verifier already accepted under non-trivial
physics:

```python
@dataclass
class Skill:
    skill_id: str
    name: str                                         # human-readable
    physical_signature: set[PhysFailureMode]          # KEY for typed retrieval
    triggers: list[str]                               # prompt keywords/cues
    sketch_template: PhysicsSketchTemplate            # parametric entities/interactions
    cinematography_preset: CinematographyTags
    checklist_template: list[ChecklistItem]           # per-skill success criteria
    acceptance_thresholds: dict[str, float]           # min p1/p2/m1 to claim success
    coupled_lessons: list[str]                        # LessonLibrary IDs auto-injected
    perf_score: float = 0.0                           # rolling avg weighted_total
    uses: int = 0
    last_used_ts: float = 0.0
    embedding: Optional[NDArray] = None               # for fallback text retrieval
```

#### Four ways Maestro skills differ from Voyager / SkillWeaver / SkillFoundry

**(a) Physics-typed retrieval, not text-similarity-only.**
At plan time, `detect_expected_modes(prompt)` returns a set of
`PhysFailureMode`s. Skill retrieval is `argmax over (signature_overlap × text_similarity)`,
not pure embedding cosine. This matches the planner's *intent type* directly
— no other framework retrieves skills by cognitively-typed key. (Voyager uses
text similarity on free-form descriptions; SkillWeaver uses webpage features.)

**(b) Verifier-monotonic distillation, not env reward or rehearsal.**
A skill is *born* exactly when this is true at the end of `generate_shot`:

```
res.escalations == 0           # HSI never had to leave Tier 0
AND initial_verdicts_severity_max >= 0.5
                              # the recipe handled a non-trivial physics issue
AND res.converged              # all checklist items resolved (no escape hatch)
```

That triple says: "the cheapest tier handled a real problem and the Verifier
accepted." The (sketch params + cinematography + thresholds + lessons used)
freezes into a Skill. **No other framework uses a Verifier's monotonic
acceptance as the distillation signal.**

**(c) Lesson coupling — skills *carry* their failures.**
Every skill stores pointers into LessonLibrary. When the skill is retrieved
at plan time, those lessons auto-inject as preconditions into the ShotSpec
(piggyback on the v0.2.1 `injected_lessons` channel). Compare to SkillOps,
which tracks health metrics; we track *concrete past failure-and-fix
pairs* per skill.

**(d) Time-axis composition.**
Voyager skills are discrete code blocks; SkillWeaver skills are individual
API calls. A Maestro skill is a *sub-shot template* with explicit start/end
markers. The Director can plan a longer shot as a chain
`[skill_A → skill_B → skill_C]` with sketch-level **handoff frames** (a
skill's last frame becomes the next skill's first_frame anchor). This makes
the C2 keyframe-edit pathway natively composable.

#### Skill library lifecycle (borrowed from SkillOps, instrumented for video)

- **Promotion**: each successful skill use lifts `perf_score` (EMA of
  weighted_total at acceptance).
- **Aging**: skills not used for N tasks get `perf_score *= 0.95` per epoch.
- **Eviction**: drop skills with `perf_score < 0.4` AND `uses > 5` (proven
  bad), or `last_used_ts > 90 days`.
- **Versioning**: when a skill's checklist needs to be modified, write a new
  `skill_id` with a `parent_id` link. (A-MEM-style memory evolution applied
  to procedural memory.)

### 4.2 — C8 · Multi-Layer Memory (MLM) — 4 canonical + 2 video-specific tiers

```
┌────────────────────────────────────────────────────────────────────┐
│ Tier 0  Working      pipeline state, current shot's review history │
│ Tier 1  Episodic     trajectory store + REPLAY (NEW)               │
│ Tier 2  Semantic     LessonLibrary (extended with A-MEM linking)   │
│ Tier 3  Procedural   SkillLibrary (NEW — C7)                       │
│ Tier 4  Entity       cross-run identity / asset DB  (NEW)          │
│ Tier 5  Preference   user cinema / style preferences (NEW)         │
└────────────────────────────────────────────────────────────────────┘
        │
        └─►  A-MEM-style cross-tier links + HippoRAG-style associative
             retrieval. One query lights up entities, skills, lessons,
             and episodes simultaneously via PPR over the cross-tier graph.
```

#### Tier 1 — Episodic with **replay**

Trajectory JSONL already exists. The new step: at the start of a new task,
look up the K most-similar *past tasks* (embedding cosine on user_prompt),
load their trajectories' *acceptance events*, and feed them to the Director
as "comparable precedents." This is **A-MEM's evolution applied to action
streams**.

#### Tier 2 — Semantic with **evolution**

Adopt A-MEM's note schema for `Lesson`:

```python
@dataclass
class Lesson:
    trigger: str                # unchanged
    fix: str
    failure_mode: Optional[PhysFailureMode]
    # NEW (A-MEM-inspired):
    keywords: list[str]
    tags: list[str]
    linked_lesson_ids: list[str]    # bidirectional
    revised_by: list[str]           # when a more general lesson replaces it
    confidence: float = 1.0         # decays with disuse, grows with re-confirmation
```

On `add(lesson)`: an LLM (or in v0.2.2 mock: a keyword-overlap heuristic)
proposes which existing lessons should be **linked** or **revised**. This
matches A-MEM's "memory evolution" property — the network self-organises.

#### Tier 3 — Procedural — see §4.1.

#### Tier 4 — Entity memory (new, borrowing VideoMemory)

```python
@dataclass
class PersistentEntity:
    entity_id: str
    canonical_name: str
    embedding: NDArray              # face / object embedding, cross-run stable
    style_descriptors: dict[str, str]   # "hair: long red, jacket: blue"
    appearance_log: list[dict]      # (task_id, bbox, prompt_context)
    physics_profile: dict[str, float]   # learned "this character runs at 1.5x"
```

VideoMemory ships a *per-run* Dynamic Memory Bank; ours is **cross-run** —
two Maestro tasks the same week can reuse the same hero with stable face,
voice, gait, even per-character physics priors.

#### Tier 5 — Preference (new, borrowing Me-Agent)

```python
@dataclass
class UserPreference:
    user_id: str
    cinematic_priors: dict          # shot_scale_distribution, movement_distribution
    style_priors: list[str]         # "kodachrome", "anamorphic"
    physics_strictness: float       # how aggressively to weight p1 in MetricTool
    recent_lessons_endorsed: list[str]
```

#### Cross-tier link graph + HippoRAG-style retrieval

The 6 tiers are nodes in a knowledge graph. Edges:

- `Lesson → PhysFailureMode` (categorical)
- `Skill → Lesson` (coupled lessons)
- `Skill → Entity` (skills built around specific characters)
- `Entity → Lesson` (failures we learned about this character)
- `EpisodicTrace → Lesson, Skill` (where they were born / used)
- `UserPreference → Skill, Entity` (which skills/characters this user likes)

At plan time the user prompt is encoded → seeds for the cross-tier graph
walk → top-K nodes returned together. **HippoRAG's neurobiological insight
applied to the multi-modal memory of a video agent.**

---

## 5. Why these two innovations are genuinely new

Two-axis claim:

| Axis | UniVA | CutClaw | VISTA | M3 | Voyager | SkillWeaver | A-MEM | VideoMemory | Me-Agent | **Maestro v0.3** |
|---|---|---|---|---|---|---|---|---|---|---|
| Memory tiers covered | 2 (semantic + user pref) | 0 | 0 | 0 | 1 (procedural) | 1 (procedural) | 1 (semantic w/ evol) | 1 (entity, single run) | 2 (semantic + pref) | **6** (all 4 canonical + entity + preference) |
| Skill discovery signal | n/a | n/a | n/a | n/a | env reward | rehearsal repeatability | n/a | n/a | n/a | **Verifier-confirmed Tier-0 monotonic convergence on non-trivial physics** |
| Skill retrieval key | n/a | n/a | n/a | n/a | text similarity | webpage features | n/a | n/a | n/a | **physical-mode signature × text similarity** |
| Skill ↔ Lesson coupling | n/a | n/a | n/a | n/a | none | none | n/a | n/a | n/a | **skills carry pointers to known lessons** |
| Cross-run entity persistence | weak | n/a | n/a | n/a | n/a | n/a | n/a | per-run only | n/a | **cross-run, with learned physics profile** |
| Memory evolution | none | n/a | n/a | n/a | n/a | none | yes (Zettelkasten) | n/a | none | **yes, applied across all 6 tiers** |

Single-sentence claim:

> **C7+C8 = Voyager's skill library + A-MEM's evolving Zettelkasten +
> VideoMemory's entity bank + Me-Agent's preference scoping, all wired through
> a HippoRAG-style cross-tier associative graph, and distilled from the
> Maestro-specific signal that no prior work has — a monotonic Verifier's
> acceptance at the cheapest HSI tier.**

---

## 6. Implementation plan (v0.3)

Stage by stage, each step must keep v0.2.2's 72 tests green.

### 6.1 — Foundations (touch types.py + memory/)

- Add `Skill`, `PersistentEntity`, `UserPreference`, `EpisodicTrace` dataclasses
  to `types.py`.
- Refactor `memory/` into a package:
  - `memory/lesson_library.py` (existing, **extend** with A-MEM fields)
  - `memory/skill_library.py` (NEW)
  - `memory/entity_store.py` (NEW)
  - `memory/preference_store.py` (NEW)
  - `memory/episodic_store.py` (NEW)
  - `memory/multi_layer.py` — façade that wraps all of the above + the
    cross-tier graph + HippoRAG-style retrieval.

### 6.2 — Hook C7 into `generate_loop.py`

- In `_distill_lesson(...)` (already exists), add a *parallel* distillation:

  ```python
  if res.escalations == 0 and initial_severity_max >= 0.5 and res.converged:
      skill_library.distill(spec, best, initial_modes, lessons_used)
  ```
- In `pipeline/plan.py:plan_shots`, BEFORE PhysicsPlanner runs, call
  `skill_library.retrieve(spec)` and attach the matched Skill to the spec
  as `spec.matched_skill`.
- Generator reads `spec.matched_skill.cinematography_preset /
  acceptance_thresholds` to pre-warm the loop.

### 6.3 — Hook C8 into `pipeline/run.py`

- Build `MultiLayerMemory` per run, persist between runs.
- `understand.py:build_asset_memory` queries Tier 4 (Entity) before generating
  new identity anchors — reuses existing entity if a strong embedding match
  exists.
- Director reads Tier 5 (Preference) to bias cinematography choices.
- Configure via `configs/default.yaml`:
  ```yaml
  memory:
    enable_skills: true
    skill_distill_severity_threshold: 0.5
    enable_entity_persistence: true
    preference_user_id: "default"
    forgetting:
      lesson_decay_per_epoch: 0.97
      skill_decay_per_epoch: 0.95
  ```

### 6.4 — Tests (≥ 6 new)

1. Skill distilled when HSI converges at Tier 0 with severity ≥ 0.5.
2. Skill **not** distilled when severity < 0.5 (avoid noise).
3. Skill retrieval ranks physical-signature match above text-only match.
4. Lessons coupled to a retrieved skill are auto-injected into the ShotSpec.
5. EntityStore cross-run: same character embedding from run 2 finds run 1's
   record.
6. A-MEM evolution: adding a new lesson links it to a related existing lesson
   (heuristic in v0.2.2; LLM-driven in v0.3).

### 6.5 — Update docs

- New `docs/memory.md` with the 6-tier architecture diagram.
- README v0.3 changelog section listing C7 + C8.
- `COMPARISON.md` gets two columns (C7, C8) and a row for the comparison
  matrix.

---

## 7. References

**Memory architectures**
- A-MEM ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110), NeurIPS 2025; [code](https://github.com/WujiangXu/A-mem))
- HippoRAG ([OpenReview](https://openreview.net/forum?id=hkujvAPVsg))
- MemoryBank ([Medium overview](https://dr-arsanjani.medium.com/introducing-memory-bank-building-stateful-personalized-ai-agents-with-long-term-memory-f714629ab601))
- G-Memory ([arXiv:2506.07398](https://arxiv.org/pdf/2506.07398))
- MemOS ([memtensor pdf](https://statics.memtensor.com.cn/files/MemOS_0707.pdf))
- HiAgent ([ACL 2025](https://aclanthology.org/2025.acl-long.1575.pdf))
- MIRIX ([arXiv:2507.07957](https://arxiv.org/pdf/2507.07957))
- Me-Agent ([arXiv:2601.20162](https://arxiv.org/html/2601.20162))
- Mem0 ([blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
- Survey: Memory for Autonomous LLM Agents ([arXiv:2603.07670](https://arxiv.org/html/2603.07670v1))
- Survey: LLM Agent Memory ([preprints.org](https://www.preprints.org/manuscript/202603.0359/v1))

**Video-specific memory**
- VideoMemory ([arXiv:2601.03655](https://arxiv.org/abs/2601.03655))
- MemoryPack + Direct Forcing ([arXiv:2510.01784](https://arxiv.org/pdf/2510.01784))
- PSIVG ([arXiv:2603.06408](https://arxiv.org/html/2603.06408))

**Skill paradigms**
- Voyager ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291); [project](https://voyager.minedojo.org/); [code](https://github.com/MineDojo/Voyager))
- SkillWeaver ([arXiv:2504.07079](https://arxiv.org/pdf/2504.07079))
- SkillFoundry ([arXiv:2604.03964](https://arxiv.org/html/2604.03964))
- SkillOps ([arXiv:2605.13716](https://arxiv.org/html/2605.13716))
- SkillAdaptor ([arXiv:2606.01311](https://arxiv.org/html/2606.01311))
- RL for self-improving skill library ([arXiv:2512.17102](https://arxiv.org/pdf/2512.17102))
- ASE — Adversarial Skill Embeddings ([SIGGRAPH 2022](https://dl.acm.org/doi/abs/10.1145/3528223.3530110))
