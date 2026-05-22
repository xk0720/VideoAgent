# `rl-integration` Branch — Detailed Summary

> Final summary of Round 13 — created on `git checkout -b rl-integration`.
> Implements the v0.3-v0.5 plan from [`AGENTIC_RL_PROPOSAL.md`](./AGENTIC_RL_PROPOSAL.md)
> as a working, mock-first **`training/`** sub-tree with passing tests.

---

## 1. What this branch adds

| Sub-tree | Purpose | Files |
|---|---|---|
| `training/env/`        | Gymnasium-style RL env wrapping `EditorAgent`'s per-segment ReAct loop. ORS-compatible tool surface. | `base.py` · `editor_env.py` · `context_manager.py` |
| `training/policy/`     | OpenRLHF `AgentInstanceBase`-style policy interface; ships an EditorAgent adapter. | `base.py` · `editor_policy.py` |
| `training/data/`       | TRL-compatible loaders for `trajectory.jsonl` / `preferences.jsonl` / `lessons.jsonl`. | `trajectory_dataset.py` · `preference_dataset.py` · `sft_dataset.py` · `lesson_dataset.py` |
| `training/rewards/`    | Composite reward (α·RM + β·m1..m6 + γ·beat) + Bradley-Terry RM trainer + HCAPO-style hindsight refiner. | `composite.py` · `editing_quality_rm.py` · `hindsight.py` |
| `training/stages/`     | Three-stage post-training: SFT → KTO → GRPO. Stub + TRL/verl/ProRL backends. | `sft.py` · `kto.py` · `grpo.py` · `_stub.py` |
| `training/runners/`    | Synchronous rollout runner + end-to-end `PipelineRunner`. | `rollout.py` · `pipeline.py` |
| `training/scripts/`    | Console entry points: `lva-train-rm` / `-sft` / `-kto` / `-grpo`. | `cli.py` |
| `tests/training/`      | 22 unit + 1 integration test. | `test_env.py` · `test_data.py` · `test_rewards.py` · `test_stages.py` · `test_runners.py` |

Plus: pyproject.toml updates (script entries, `pythonpath`, packages.find), a root `conftest.py`, and `tests/__init__.py` to make pytest's import-mode play nicely with the second top-level package.

---

## 2. The 6-framework research that shaped the design

Eight WebSearches in Round 13 surfaced 30+ relevant 2025-2026 frameworks. The architectural decisions below come directly from them:

| Decision | Source it came from |
|---|---|
| 3-component split (Env Manager / Context Manager / Agent Proxy) | **RAGEN** — modular refactor in the 2025 update with verl as submodule. ([github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN)) |
| Server/client separation for inference engine vs agent | **verl 0.7 AgentLoop** — server-client + asyncio for non-blocking tool calls. ([verl.readthedocs.io/start/agentic_rl](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)) |
| `AgentPolicyBase` class with sync `reset` + `act` | **OpenRLHF `AgentInstanceBase`** — same signature shape, async stripped because env is sync. ([openrlhf docs](https://openrlhf.readthedocs.io/en/latest/async_rl.html)) |
| SFT → KTO → GRPO three-stage | **EVA — Efficient Video Agent** (arXiv 2603.22918, 2026). Exact recipe is reproduced in `training/stages/`. |
| ORS-style tool calls as action space | **Open Reward Standard** ([openrewardstandard.io](https://openrewardstandard.io)) — agent⇄env interaction via tool calls only. EditorEnv.tools() returns OpenAI-style function schemas. |
| TRL v1.0 unified trainer surface | **HuggingFace TRL v1.0** (Apr 2026) — same config shape across SFT / DPO / KTO / GRPO. ([TRL docs](https://huggingface.co/docs/trl)) |
| Turn-PPO macro-action (segment = episode) | **Turn-PPO (EACL 2026)** — eliminates token-level credit-assignment variance. |
| HCAPO hindsight refiner | **HCAPO** (arXiv 2603.08754, 2026) — +7.7% WebShop / +13.8% ALFWorld over GRPO. |
| Bradley-Terry RM stub for `EditingQualityRM` | **Tülu-3-RM** (Allen AI, Nov 2024) + **Skywork-Reward-Gemma-2-27B** (Oct 2024) recipe. |
| Composite reward with disentangled α/β/γ | **ODIN** (arXiv 2402.07319) — disentangled reward mitigates hacking. Anti-hacking signal comes from `EnsembleResult.disagreement`. |
| GRPO leave-one-out advantage | **GRPO** (DeepSeek-Math 2024); **DGPO** (ICLR 2026) is the v0.4 swap. |
| Stub backend with deterministic loss | **"SFT Memorizes, RL Generalizes" (ICLR 2026)** — recommends *closest-to-base SFT ckpt* not highest-eval; recorded as a config knob `selection_strategy`. |

All references are cited in the relevant source file's top docstring, not just here.

---

## 3. How the data flows (v0.2 outputs → v0.3 training)

```
   v0.2 (main branch) produces           v0.3 (rl-integration) consumes
   ───────────────────────────────       ───────────────────────────────
   outputs/run_N_trajectory.jsonl  ───►  TrajectoryDataset       ───┐
                                            ├─ steps() (BC SFT)     │
                                            ├─ segment_summaries()  │
                                            └─ filter_high_reward() │
                                                                    ▼
   outputs/run_N_preferences.jsonl ───►  PreferenceDataset    ───►  SFTDataset
                                            ├─ .to_hf_dataset()         │
                                            └─ .kto_view()              ▼
                                                                    SFTStage (Stage A)
   .cache/lessons.jsonl            ───►  LessonDataset                  │
                                            ├─ .retrieve_relevant()     ▼
                                                                    KTOStage (Stage B)
                                                                        │
                                                                        ▼
                                                                    GRPOStage (Stage C)
                                                                    via EditorEnv + EditorAgentPolicy
                                                                        │
                                                                        ▼
                                                                    PipelineReport (JSON)
                                                                    + ckpts under .cache/v04/...
```

---

## 4. Verification

```
pyflakes (src + tests + benchmark + scripts + training): 0 warnings
pytest tests/ -q                                          : 124 passed in ~20 s
   - main repo:        102 tests (unchanged from main branch)
   - tests/training/:   22 tests (new)
```

Each `training/` sub-module has its own focused tests:

| Test file | Coverage |
|---|---|
| `tests/training/test_env.py` (5)        | ContextManager bounded history; EditorEnv reset / step / unknown-action penalty / termination / ORS tool surface |
| `tests/training/test_data.py` (4)       | TrajectoryDataset views + reward filter; SFTDataset format; PreferenceDataset triples + KTO view |
| `tests/training/test_rewards.py` (7)    | CompositeReward breakdown; EditingQualityRMTrainer fit + save/load; drop-in BaseRewardModel; Hindsight γ=0/γ=1/critic-downweight |
| `tests/training/test_stages.py` (4)     | SFT stub; KTO with default + DPO loss; GRPO stub rollouts |
| `tests/training/test_runners.py` (2)    | RolloutRunner transitions + dump; **end-to-end PipelineRunner integration** (RM → SFT → KTO → GRPO orchestrator) |

---

## 5. What the branch deliberately leaves stubbed (and the v0.4 swap-ins)

| Stub | One-flag activation | Real-world cost |
|---|---|---|
| SFTStage real backend | `SFTConfig(backend="trl")` + `pip install trl transformers datasets` | LoRA on Qwen3-VL-8B fits 1 × H100; days |
| KTOStage real backend | `KTOConfig(backend="trl", loss_type="kto"\|"dpo"\|"ipo"\|"simpo")` | Same |
| GRPOStage real backend | `GRPOConfig(backend="verl")` + install verl 0.7 + fill in `_fit_verl()` (it's a ~50-line glue) | Multi-GPU; depends on rollout batch size |
| EditingQualityRMTrainer deep model | Replace `editing_quality_rm.py` internals with `trl.RewardTrainer` on Qwen3-VL-8B | Standard RM training; days on 1 × H100 |
| HCAPO LLM critic | Fill in `HindsightCriticRefiner.refine_with_llm` per arXiv 2603.08754 | Adds 1 LLM call per episode |
| Async rollout (verl AgentLoop) | Wrap RolloutRunner.run in asyncio + use vLLM | Throughput win, no behaviour change |

Every stub raises `NotImplementedError` with a pointer to the right 2026 paper / repo for the implementor.

---

## 6. Reproducibility check (do this on the branch)

```bash
# 1. switch to branch
git checkout rl-integration

# 2. install base deps (no torch/TRL/vLLM needed for stub backend)
pip install -e .

# 3. verify full test suite stays green
python -m pytest tests/ -q
# Expected: 124 passed

# 4. dry-run the four CLIs
python -m training.scripts.cli train_rm    --preferences <path> --output <path>
python -m training.scripts.cli train_sft   --trajectory <path>   --output <path>
python -m training.scripts.cli train_kto   --preferences <path>  --output <path>
python -m training.scripts.cli train_grpo  --cache-dir <path>    --output <path>
# Each writes metrics.json + ckpt artifact to the given output dir.
```

---

## 7. What I'd do next (Round 14 candidates, not yet executed)

1. **Wire one real `backend="trl"` end-to-end on a small model** (Qwen3-1.5B or similar) to validate the stub→real swap is in fact one flag.
2. **Build a `train_pipeline` CLI** that calls `PipelineRunner` from the command line, so a single command produces RM + SFT + KTO + GRPO artifacts.
3. **Add a `training/eval/` sub-package** with `mashup_bench_runner.py` + `reward_hacking_probes.py` — the success metrics in proposal §7 are not yet executable harnesses.
4. **DGPO loss option for KTOStage** (ICLR 2026 paper) — diffusion-RL focused but applicable.
5. **Multi-agent MARTI swap for v0.5** — once v0.4 Editor RL is real, MARTI 0.4 lets us co-train Director + Editor with shared advantage estimation.

---

## 8. Tracking against the original ask

The original request was:
> 在原来项目中单独建一个 RL 的分支，搜索 Agentic RL 如何应用到目前或者类似的框架中（一定要完整），然后进行多轮的代码编写，检查，测试等环节。最后再写一个文档详细总结。

| Sub-task | Status |
|---|---|
| 建分支 `rl-integration` | ✅ done; main 仍干净 |
| Agentic RL 在类似框架的实现调研（完整） | ✅ 8 WebSearches，30+ 真实框架/论文 |
| 多轮代码编写 | ✅ 8 sub-packages, 22 source files |
| 检查 | ✅ pyflakes 0 warning |
| 测试 | ✅ 22 unit + 1 integration tests (124 total green) |
| 详细总结文档 | ✅ this file + `training/README.md` + `AGENTIC_RL_PROPOSAL.md §9.5 实现状态` |
