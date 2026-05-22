"""Console-script entry points for `lva-train-*`.

Each function follows the same pattern as ``src/longvideoagent/scripts_impl.py``:
    parse args → build minimal objects → call runner → print summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ─── lva-train-rm ────────────────────────────────────────────────────


def train_rm_main() -> int:
    from ..data.preference_dataset import PreferenceDataset
    from ..rewards.editing_quality_rm import EditingQualityRMTrainer

    p = argparse.ArgumentParser(prog="lva-train-rm",
                                description="Train an EditingQualityRM on a preferences.jsonl")
    p.add_argument("--preferences", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Path for the trained RM JSON (will be created)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()

    prefs = PreferenceDataset.from_file(args.preferences)
    trainer = EditingQualityRMTrainer(lr=args.lr, n_epochs=args.epochs)
    trainer.fit(prefs)
    trainer.save(args.output)
    acc = trainer.stats.n_correct / max(1, trainer.stats.n_seen)
    print(json.dumps({"n_pairs": trainer.stats.n_seen,
                      "in_train_pairwise_acc": acc,
                      "weights": trainer.stats.weights,
                      "ckpt": str(args.output)}, indent=2))
    return 0


# ─── lva-train-sft ───────────────────────────────────────────────────


def train_sft_main() -> int:
    from ..data.trajectory_dataset import TrajectoryDataset
    from ..data.sft_dataset import SFTDataset
    from ..stages.sft import SFTConfig, SFTStage

    p = argparse.ArgumentParser(prog="lva-train-sft",
                                description="Stage A: cold-start SFT on filtered trajectories")
    p.add_argument("--trajectory", type=Path, required=True, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reward-threshold", type=float, default=7.0)
    p.add_argument("--backend", choices=["stub", "trl"], default="stub")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    traj = TrajectoryDataset.from_files(args.trajectory)
    sft_data = SFTDataset.from_trajectory(traj, reward_threshold=args.reward_threshold)
    stage = SFTStage(SFTConfig(backend=args.backend, n_epochs=args.epochs))
    m = stage.fit(sft_data, args.output)
    print(json.dumps({"n_records": m.n_records, "final_loss": m.final_loss,
                      "elapsed_s": m.elapsed_s, "output": str(args.output)}, indent=2))
    return 0


# ─── lva-train-kto ───────────────────────────────────────────────────


def train_kto_main() -> int:
    from ..data.preference_dataset import PreferenceDataset
    from ..stages.kto import KTOConfig, KTOStage

    p = argparse.ArgumentParser(prog="lva-train-kto",
                                description="Stage B: preference alignment (KTO/DPO/IPO/SimPO)")
    p.add_argument("--preferences", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--loss-type", choices=["kto", "dpo", "ipo", "simpo"], default="kto")
    p.add_argument("--backend", choices=["stub", "trl"], default="stub")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    prefs = PreferenceDataset.from_file(args.preferences)
    stage = KTOStage(KTOConfig(backend=args.backend, loss_type=args.loss_type,
                                n_epochs=args.epochs))
    m = stage.fit(prefs, args.output)
    print(json.dumps({"n_records": m.n_records, "final_loss": m.final_loss,
                      "loss_type": args.loss_type,
                      "output": str(args.output)}, indent=2))
    return 0


# ─── lva-train-grpo ──────────────────────────────────────────────────


def train_grpo_main() -> int:
    from dataclasses import asdict
    from longvideoagent.config import load_config
    from ..env.editor_env import EditorEnv
    from ..policy.editor_policy import EditorAgentPolicy
    from ..rewards.editing_quality_rm import EditingQualityRM
    from ..stages.grpo import GRPOConfig, GRPOStage
    from longvideoagent.models.llm import MockLLMClient
    from longvideoagent.models.reward.base import MockRewardModel
    from longvideoagent.models.video_gen import MockVideoGenClient
    from longvideoagent.tools import GenerationTool, RetrievalTool
    from longvideoagent.memory.retriever import MemoryRetriever
    from longvideoagent.memory.store import MemoryStore

    p = argparse.ArgumentParser(prog="lva-train-grpo",
                                description="Stage C: GRPO RL (stub backend by default)")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Cache dir created by an earlier `lva-preprocess` run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rm-path", type=Path, default=None,
                   help="Optional EditingQualityRM JSON; falls back to MockRewardModel")
    p.add_argument("--n-rollouts", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=2)
    p.add_argument("--backend", choices=["stub", "verl", "prorl"], default="stub")
    args = p.parse_args()

    cfg = load_config()
    store = MemoryStore(args.cache_dir)
    memory = store.load_full_memory(load_features=True)
    retriever = MemoryRetriever(store, embed_dim=cfg.preprocess.feature_extractor.embed_dim)

    rm = EditingQualityRM.load(args.rm_path) if args.rm_path else MockRewardModel()

    # Reuse pipeline.plan to get realistic guidances.
    from longvideoagent.pipeline.plan import plan as plan_fn
    guidances = plan_fn(memory, "Train GRPO smoke prompt", cfg, memory_store=store)

    def env_factory():
        return EditorEnv(
            memory=memory, guidances=guidances,
            retrieval_tool=RetrievalTool(retriever,
                                          beam_width=cfg.compose.retrieval.beam_width,
                                          top_k_pool=cfg.compose.retrieval.top_k_pool),
            generation_tool=GenerationTool(MockVideoGenClient(),
                                            default_duration_s=cfg.compose.generation.duration_default),
            reward_model=rm,
            max_steps=cfg.compose.max_editor_steps,
        )

    policy = EditorAgentPolicy(MockLLMClient(alias="editor"))
    stage = GRPOStage(GRPOConfig(backend=args.backend,
                                  n_rollouts_per_step=args.n_rollouts,
                                  n_steps=args.n_steps))
    m = stage.fit(env_factory, policy, args.output)
    print(json.dumps(asdict(m), indent=2))
    store.close()
    return 0
