"""KTO preference-alignment stage.

References:
    • KTO — Kahneman-Tversky Optimization (Ethayarajh et al., 2024)
    • EVA Stage 2 (arXiv 2603.22918) — KTO works well on noisy / partial labels
    • TRL v1.0 ``KTOTrainer`` (https://huggingface.co/docs/trl/kto_trainer)
    • SimPO / DPO are drop-in alternates — flip ``loss_type``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..data.preference_dataset import PreferenceDataset
from ._stub import StubMetrics, stub_train


@dataclass
class KTOConfig:
    backend: str = "stub"                              # "stub" | "trl"
    loss_type: str = "kto"                             # "kto" | "dpo" | "ipo" | "simpo"
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    lr: float = 5e-6
    n_epochs: int = 1
    beta: float = 0.1
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8


class KTOStage:
    """Stage B: preference-pair fine-tune."""

    name = "kto"

    def __init__(self, config: Optional[KTOConfig] = None) -> None:
        self.config = config or KTOConfig()
        self.name = self.config.loss_type

    def fit(self, dataset: PreferenceDataset, output_dir: Path | str) -> StubMetrics:
        output_dir = Path(output_dir)
        # Choose which dataset view to feed.
        if self.config.loss_type == "kto":
            data = dataset.kto_view()
        else:
            data = list(dataset)
        if self.config.backend == "stub":
            return stub_train(
                self.name, data, n_epochs=self.config.n_epochs,
                output_dir=output_dir,
                extras={"loss_type": self.config.loss_type,
                        "model_name": self.config.model_name,
                        "beta": self.config.beta},
            )
        if self.config.backend == "trl":                # pragma: no cover
            return self._fit_trl(data, output_dir)
        raise ValueError(f"Unknown KTOStage backend {self.config.backend!r}")

    def _fit_trl(self, data, output_dir: Path) -> StubMetrics:  # pragma: no cover
        try:
            from datasets import Dataset                     # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError("KTOStage(backend='trl') needs trl+transformers+datasets.") from e
        ds = Dataset.from_list(list(data))
        tok = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        if self.config.loss_type == "kto":
            from trl import KTOConfig as TRLKTOConfig, KTOTrainer       # type: ignore
            cfg = TRLKTOConfig(output_dir=str(output_dir),
                                per_device_train_batch_size=self.config.per_device_batch_size,
                                gradient_accumulation_steps=self.config.grad_accum_steps,
                                learning_rate=self.config.lr,
                                num_train_epochs=self.config.n_epochs,
                                beta=self.config.beta)
            trainer = KTOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
        elif self.config.loss_type in ("dpo", "ipo", "simpo"):
            from trl import DPOConfig as TRLDPOConfig, DPOTrainer       # type: ignore
            cfg = TRLDPOConfig(output_dir=str(output_dir),
                                per_device_train_batch_size=self.config.per_device_batch_size,
                                gradient_accumulation_steps=self.config.grad_accum_steps,
                                learning_rate=self.config.lr,
                                num_train_epochs=self.config.n_epochs,
                                beta=self.config.beta,
                                loss_type=self.config.loss_type)
            trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
        else:
            raise ValueError(f"Unknown loss_type {self.config.loss_type!r}")
        trainer.train()
        trainer.save_model(str(output_dir))
        return StubMetrics(stage=self.name, backend="trl",
                           n_records=len(ds), n_epochs=self.config.n_epochs,
                           final_loss=float("nan"), elapsed_s=0.0,
                           extras={"output_dir": str(output_dir)})


__all__ = ["KTOStage", "KTOConfig"]
