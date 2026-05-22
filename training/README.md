# training/ — Agentic RL Sub-tree

> v0.3 → v0.5 scaffolding implementing the plan in
> [`../docs/AGENTIC_RL_PROPOSAL.md`](../docs/AGENTIC_RL_PROPOSAL.md).
>
> All code is **mock-first**: the test suite passes on CPU with no torch /
> TRL / vLLM installed; flipping ``backend="trl"`` / ``"verl"`` / ``"prorl"``
> activates the real training paths (still uses the same data + env code).

---

## What's in here

| Sub-package | What it is | Maps to which 2026 framework |
|---|---|---|
| `env/`       | Gymnasium-style RL envs that wrap agents as ORS-style tool callers. | RAGEN Environment Manager + verl 0.7 AgentLoop + Open Reward Standard |
| `policy/`    | Synchronous policy ABC + EditorAgent adapter. | OpenRLHF `AgentInstanceBase` (sync version) |
| `data/`      | Loaders for our v0.2-emitted `trajectory.jsonl`, `preferences.jsonl`, `lessons.jsonl`. | TRL v1.0 dataset schemas (prompt / chosen / rejected) |
| `rewards/`   | Composite reward + EditingQualityRM trainer + Hindsight critic refiner. | Tülu-3-RM, Skywork-Reward, AgentPRM, HCAPO |
| `stages/`    | Four-stage post-training: `SFTStage` → **`OPDStage` (A.5, optional)** → `KTOStage` → `GRPOStage`. | EVA (arXiv 2603.22918) + GKD/OPD (arXiv 2306.13649) + TRL v1.0 trainers |
| `runners/`   | Rollout runner + end-to-end `PipelineRunner`. | verl/ProRL rollout-as-a-service |
| `scripts/`   | Console-script entry points (`lva-train-rm` / `-sft` / `-kto` / `-grpo`). | — |

---

## How to use it (mock backend — runs on CPU, no GPU)

### 1. Generate some data

The v0.2 pipeline already produces the inputs the training subtree needs:

```bash
# This run writes outputs/run1_trajectory.jsonl + outputs/run1_preferences.jsonl
PYTHONPATH=src python scripts/run_pipeline.py \
    --source tests/fixtures/tiny_clip.mp4 \
    --cache-dir .cache/run1 \
    --user-prompt "Make a highlight reel" \
    --output outputs/run1.mp4 \
    --trajectory-log outputs/run1_trajectory.jsonl \
    --preference-log outputs/run1_preferences.jsonl
```

Repeat with different prompts / sources to accumulate dataset diversity.

### 2. Train an EditingQualityRM on the accumulated preferences

```bash
PYTHONPATH=src:. python -m training.scripts.cli train_rm \
    --preferences outputs/run1_preferences.jsonl \
    --output .cache/rm/editing_quality_rm.json \
    --epochs 5
```

Output: a JSON file holding the 6 Bradley-Terry weights + bias + in-train
pairwise accuracy. Load with `EditingQualityRM.load(path)`; it satisfies
`BaseRewardModel`, so it drops straight into the `EnsembleRewardModel`.

### 3. SFT cold start (Stage A)

```bash
PYTHONPATH=src:. python -m training.scripts.cli train_sft \
    --trajectory outputs/run1_trajectory.jsonl outputs/run2_trajectory.jsonl \
    --output .cache/sft \
    --reward-threshold 7.0 \
    --backend stub
```

Stub backend writes `metrics.json` + `ckpt.json` (deterministic stand-in).
Flip `--backend trl` once you have GPUs + TRL installed.

### 4. KTO preference alignment (Stage B)

```bash
PYTHONPATH=src:. python -m training.scripts.cli train_kto \
    --preferences outputs/run1_preferences.jsonl \
    --output .cache/kto \
    --loss-type kto
```

Switch `--loss-type` to `dpo`, `ipo`, or `simpo` to try different losses
(same data — TRL v1.0's unified config makes the swap one flag).

### 5. GRPO RL (Stage C — stub backend)

```bash
PYTHONPATH=src:. python -m training.scripts.cli train_grpo \
    --cache-dir .cache/run1 \
    --output .cache/grpo \
    --n-rollouts 4 --n-steps 4
```

Stub backend rolls episodes through `EditorEnv` + `EditorAgentPolicy` and
writes `grpo_metrics.json` (mean reward, std, GRPO leave-one-out
advantages, hindsight-refined episode totals). For real GRPO, set
`--backend verl` or `--backend prorl` (NotImplementedError until you wire
the gradient updates — see the docstrings).

### 6. Or run the whole pipeline at once

```python
from pathlib import Path
from training.runners.pipeline import PipelineConfig, PipelineRunner

# (See tests/training/test_runners.py for a complete usage example.)
runner = PipelineRunner(PipelineConfig())
report = runner.run(
    trajectory_path=Path("outputs/run1_trajectory.jsonl"),
    preferences_path=Path("outputs/run1_preferences.jsonl"),
    env_factory=my_env_factory,   # takes a EditingQualityRM, returns AgentEnvBase
    policy=my_policy,
    output_dir=Path(".cache/v04_pipeline"),
)
print(report)
```

---

## How the 2026 references map into source files

| Reference | File |
|---|---|
| Open Reward Standard (ORS) — agents interact with env only via tool calls | `env/base.py::AgentEnvBase.tools()` |
| RAGEN Environment / Context / Agent split | `env/editor_env.py`, `env/context_manager.py`, `policy/editor_policy.py` |
| verl AgentLoop server/client decoupling | `runners/rollout.py` runs sync; verl wraps it in asyncio at scale |
| OpenRLHF `AgentInstanceBase` | `policy/base.py::AgentPolicyBase` |
| Turn-PPO macro-action (EACL 2026) | `env/editor_env.py` treats each segment as one episode |
| HCAPO hindsight credit (arXiv 2603.08754, 2026) | `rewards/hindsight.py::HindsightCriticRefiner` |
| EVA SFT→KTO→GRPO (arXiv 2603.22918, 2026) | `stages/{sft,kto,grpo}.py` |
| TRL v1.0 unified trainers (HF, Apr 2026) | `stages/sft.py`, `stages/kto.py` lazy-import |
| GRPO (DeepSeek-Math, 2024) / DGPO (ICLR 2026) | `stages/grpo.py::GRPOStage._fit_stub` leave-one-out baseline |
| Tülu-3-RM / Skywork-Reward / AgentPRM RM training | `rewards/editing_quality_rm.py` (Bradley-Terry stub; TRL `RewardTrainer` is the v0.4 swap) |
| ODIN-style disentangled reward (anti reward-hacking) | `rewards/composite.py` α/β/γ knobs — set to 0 to mask a term |
| "GRPO secretly DPO" (arXiv 2510.00977, 2025) | preferences.jsonl ↔ GRPO rollouts; same data feeds either |

---

## What's intentionally stubbed (and why)

* **Real GRPO / PPO gradient updates** — handled by verl 0.7 or ProRL Agent
  externally. Our `stages/grpo.py::_fit_stub` walks the rollout loop, computes
  the same advantages a real GRPO would, and dumps them to disk. When you
  install verl + flip `backend="verl"`, the rollout/advantage/update loop is
  the verl `AgentLoop`'s job; we just hand it `env_factory` + `policy`.
* **EditingQualityRM as a deep model** — v0.3 stub is a 6-weight Bradley-Terry
  fit. The deep model slot is a v0.4 swap with `transformers` + `trl.RewardTrainer`.
* **LLM-driven HCAPO** — `HindsightCriticRefiner.refine_with_llm` raises
  `NotImplementedError`; v0.4 work.
* **Async rollout** — `RolloutRunner.run` is sync. OpenRLHF 0.8.0 / verl 0.7
  give you async; we don't reproduce the asyncio machinery here.

---

## Running the tests

```bash
python -m pytest tests/training/ -q
```

22 unit + 1 integration test cover env, data, rewards, stages, runners.
`pytest tests/ -q` runs the full repo suite (124 tests).

---

## CI

The same `.github/workflows/test.yml` job that runs the v0.2 tests also
runs the training/ ones — no separate workflow needed.
