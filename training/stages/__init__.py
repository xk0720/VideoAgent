"""Three-stage post-training: SFT → KTO → GRPO.

Convention (EVA arXiv 2603.22918 + TRL v1.0):
    Stage A: SFT cold start    → ``sft.py::SFTStage``
    Stage B: preference align  → ``kto.py::KTOStage`` (default), DPOStage / SimPOStage available
    Stage C: RL fine-tune      → ``grpo.py::GRPOStage``

Each Stage exposes the same surface:

    stage = SomeStage(config)
    metrics = stage.fit(dataset, output_dir=Path(...))

Real torch / TRL training is hidden behind a ``backend="trl"`` flag; the
default is ``backend="stub"`` which deterministically simulates a few
training epochs over the data structure — useful for CPU-only tests +
sanity checking the data pipeline before lighting up GPUs.
"""
from .sft import SFTStage, SFTConfig
from .kto import KTOStage, KTOConfig
from .grpo import GRPOStage, GRPOConfig
from .distill import OPDStage, OPDConfig, OPDMetrics

__all__ = ["SFTStage", "SFTConfig", "KTOStage", "KTOConfig",
           "GRPOStage", "GRPOConfig",
           "OPDStage", "OPDConfig", "OPDMetrics"]
