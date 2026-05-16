# Design Decisions (running log)

Each entry has a one-line summary, the reason, and the alternatives considered.
Style intentionally terse — this is a working doc, not a paper.

---

## 2026-05-16 — initial v0.1 scaffold

### D-001 · Python state-machine fallback for Stage 2 instead of LangGraph
**Why:** `langgraph` pulls `langchain-core` and friends (~30 MB). v0.1 must run
on a CPU laptop with `pip install -e .` and no GPU; tests should not require the
extra. The fallback's node names/signatures mirror LangGraph so the swap is one wrapper.
**Alternatives:** vendored langgraph; or write our own DAG library.
**Owner:** v0.2 can flip `plan.use_langgraph: true` when langgraph is installed.

### D-002 · MockLLMClient is the default everywhere when `mocks.llm == true`
**Why:** Hermetic tests / hermetic onboarding. The mock can read a per-alias
JSON fixture (`tests/fixtures/mock_llm/<alias>.json`) so prompt-template work
is testable without burning API tokens.
**Alternatives:** require API keys to be set even for tests (rejected — onboarding pain).

### D-003 · Tools always validate "last candidate" immediately after retrieve/generate
**Why:** Saves one LLM step per segment; the EditorAgent's `validate` action
is then only used when it wants to re-judge the whole candidate pool. This
trims the typical step count from ~6 to ~3.
**Alternatives:** strict ReAct (think → act → think → validate). DIRECT's
loop is roughly the same shape.

### D-004 · numpy fallback for vector search if FAISS is not installed
**Why:** Same hermetic-install reason. `MemoryStore.search_by_embedding`
auto-detects faiss and L2-normalises identically in both branches so results
are unaffected.
**Alternatives:** hard-depend on faiss-cpu.

### D-005 · Real ffmpeg assembly, even in v0.1
**Why:** Design-doc §11 mandates a real .mp4 output. ffmpeg-python plus the
system binary is small and ubiquitous; mocking would defeat the smoke test.
**Alternatives:** moviepy (heavier and slower).

### D-006 · MetricTool sits next to the agents (not inside RetrievalTool)
**Why:** Per design-doc §7.4 EditorAgent must be able to *query* metrics
directly during ReAct ("why did this candidate score 4.2 on m4?"). Sharing
the metric functions across both call sites is a single-file convention.

### D-007 · Single SQLite file for memory metadata
**Why:** stdlib, zero-deps, copy-and-go cache. FAISS index is a separate
file so a missing FAISS install doesn't corrupt the metadata.

### D-008 · TextEncoder type alias used by MemoryRetriever
**Why:** Decouples retrieval from CLIP/transformers; tests inject the
deterministic mock encoder so retrieval scoring is reproducible. Swapping to
the real open_clip text head is one line in `build_video_gen_from_config`'s
sibling factory.

### D-009 · Prompts live as .txt under `src/longvideoagent/prompts/`
**Why:** Design-doc §15 forbids hardcoded prompts. Bundling them in
`package_data` lets `lva-run` work after `pip install` outside the repo.

### D-010 · Trajectory log redacts large ndarrays by default
**Why:** Avoids 100-MB JSONL files. Replaced by `{"__ndarray__": true,
"shape": ..., "dtype": ...}` so RAGEN can replay the trajectory deterministically
(it cares about action/observation pairs, not raw pixels).

### D-011 · video_gen wrappers raise NotImplementedError instead of silently mocking
**Why:** If a user explicitly picks `backend: omniweaving` with `mocks.video_gen=false`,
they should fail loudly. Mock mode flips through `MockVideoGenClient` which writes
a deterministic colour clip.

### D-012 · `config.py` uses dataclasses, not `pydantic.BaseModel`
**Why:** Pydantic v2 ships a 2-MB binary wheel (`pydantic_core`) that does not always
install cleanly on locked-down hosts. The config tree is fully typed; dataclasses plus
`typing.get_type_hints` and a 30-line `_from_dict` give us the same validate-on-load
shape with zero binary deps. Flipping back to pydantic is mechanical.

### D-013 · Soft imports for loguru / rich / scipy / ffmpeg-python
**Why:** Same hermetic-install goal as D-004 (FAISS) and D-012 (pydantic).
Every "optional" wrapper degrades gracefully: `logger` → stdlib `logging` when loguru
is missing, `rich` table → plain-text trajectory dump, `scipy.stats.wasserstein_distance`
/ `spearmanr` → numpy fallbacks (m4/m6 keep working), `ffmpeg-python` → direct
`subprocess.run(["ffmpeg", ...])` (system binary still required).
**Trade-off:** the docstring at the top of each affected file still names the canonical
OSS library — those names are the v0.2 install targets, not optional fluff.

### D-014 · `shutil.move` instead of `Path.replace` for final mp4 emission
**Why:** AssemblyTool writes into `tempfile.TemporaryDirectory()` which often lives on
a different filesystem than the user-chosen output path, so `Path.replace` raises
``OSError: Invalid cross-device link``. `shutil.move` falls back to copy+unlink.

---

## 2026-05-16 (later) — v0.2 self-loop evolution decisions

### D-015 · 6 agents, not 5 — CriticAgent is post-hoc and isolated
**Why:** Self-grading bias (an agent praising its own outputs) is a known failure
mode of Self-Refine-style loops. We keep the executor agents (Editor / Validator)
focused on the current run and give a **separate** CriticAgent the trajectory
scan + lesson extraction job. This mirrors Self-Discover's separation of meta-agent
from base agents (Zhou et al., 2024) and Reflexion's actor/evaluator split.
**Trade-off:** one extra agent in the system; but it adds zero LLM-call cost in v0.1
(rule-based scanners) and is cheap to upgrade to an LLM-driven critic in v0.2.

### D-016 · LessonBook stores **across runs**, not just **across episodes**
**Why:** Reflexion's original setup keeps verbal memory within one task attempt.
For evolving a long-video editing system we want lessons that persist across the
many runs a user / pipeline does — that's where the data compounding lives.
JSONL append-only schema makes the file simultaneously: (a) a Reflexion memory,
(b) a future RL training set, (c) a Trace-style (Microsoft 2024) signal source.
**Trade-off:** schema must stay stable; we document it in `memory/lessons.py` docstring.

### D-017 · EnsembleRewardModel uses ≥3 mock judges with different weight vectors
**Why:** Even in fully-mock mode we want a non-trivial disagreement signal so that
downstream consumers (`EnsembleResult.is_active_learning_candidate(...)`) and
tests can exercise the active-learning path. Three judges with weights drawn from
heuristic presets (balanced / semantic-priority / motion-priority) is the minimal
configuration that produces a meaningful variance.
**Trade-off:** 3× the per-candidate scoring cost; but mock cost is negligible and
real (MLLM) judges are still optional.

### D-018 · PreferenceLogger schema mirrors HuggingFace `trl.DPOTrainer`
**Why:** v0.3 RM training will be done with `trl.DPOTrainer` (or its KTO/SimPO
variants). Matching its expected schema (`prompt` / `chosen` / `rejected`) means
no transformation code at training time — `jq` can project our records directly.
We also include `judge` (which RM signed off on the preference) so future
research can study judge-induced bias.
**Trade-off:** schema is slightly more verbose than necessary for one consumer;
worth it because GRPO/RLOO/KTO can all be derived from the same records.

### D-019 · Self-Consistency on Screenwriter only, not every agent
**Why:** Self-consistency (Wang et al., ICLR 2023) multiplies LLM cost by K. We
apply it only to the **highest-leverage** decision point (Screenwriter's global
plan controls every downstream agent's input). Lower-leverage decisions (Director's
per-section query, Editor's per-step action) don't pay back the 3× cost.
**Trade-off:** non-Screenwriter agents remain single-sample; v0.3 can flip a per-agent
flag if benchmark results justify it.

### D-020 · Co-cite older + newer work, don't replace
**Why:** Some users want canonical references (DIRECT, Reflexion, DPO); others
want the latest (HunyuanVideo, Tülu-3, GRPO, rStar, Trace). Every code docstring
and SYSTEM_GUIDE entry now lists **both** when applicable, so reviewers from either
camp find their footing without us having to take sides. Following the user's
"引用尽量要最新" requirement, we prioritise 2024–2025 work in `configs/` defaults
and in the dependency map, but historical references stay for traceability.
