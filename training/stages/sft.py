"""SFT cold-start stage.

References:
    • DeepSeek-R1 SFT cold start
    • EVA Stage 1 (arXiv 2603.22918)
    • "SFT Memorizes, RL Generalizes" (ICLR 2026) — warning: select ckpt
      *closest to base model*, not highest eval score
    • TRL v1.0 ``SFTTrainer`` (https://huggingface.co/docs/trl)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..data.sft_dataset import SFTDataset
from ._stub import StubMetrics, stub_train


@dataclass
class SFTConfig:
    backend: str = "stub"                              # "stub" | "trl"
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"      # only used when backend='trl'
    lora_r: int = 16
    lora_alpha: int = 32
    lr: float = 2e-5
    n_epochs: int = 1
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    max_seq_len: int = 4096
    selection_strategy: str = "closest_to_base"        # ICLR 2026 advice; alt: "highest_eval"


class SFTStage:
    """Stage A: behavioural-cloning SFT on reward-filtered trajectories."""

    name = "sft"

    def __init__(self, config: Optional[SFTConfig] = None) -> None:
        self.config = config or SFTConfig()

    def fit(self, dataset: SFTDataset, output_dir: Path | str) -> StubMetrics:
        output_dir = Path(output_dir)
        if self.config.backend == "stub":
            return stub_train(
                self.name, dataset, n_epochs=self.config.n_epochs,
                output_dir=output_dir,
                extras={"model_name": self.config.model_name,
                        "selection_strategy": self.config.selection_strategy},
            )
        if self.config.backend == "trl":                # pragma: no cover
            return self._fit_trl(dataset, output_dir)
        raise ValueError(f"Unknown SFTStage backend {self.config.backend!r}")

    def _fit_trl(self, dataset: SFTDataset, output_dir: Path) -> StubMetrics:  # pragma: no cover
        # Lazy import — only when the caller flips backend='trl'.
        try:
            from trl import SFTTrainer, SFTConfig as TRLSFTConfig    # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            from datasets import Dataset                             # type: ignore
        except ImportError as e:
            raise ImportError(
                "SFTStage(backend='trl') needs `pip install -e .[llm]` "
                "and `pip install trl transformers datasets`."
            ) from e
        ds = Dataset.from_list(list(dataset))
        cfg = TRLSFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.grad_accum_steps,
            learning_rate=self.config.lr,
            num_train_epochs=self.config.n_epochs,
            max_seq_length=self.config.max_seq_len,
        )
        tok = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
        trainer.train()
        trainer.save_model(str(output_dir))
        return StubMetrics(stage=self.name, backend="trl", n_records=len(ds),
                           n_epochs=self.config.n_epochs, final_loss=float("nan"),
                           elapsed_s=0.0,
                           extras={"output_dir": str(output_dir)})


__all__ = ["SFTStage", "SFTConfig"]
