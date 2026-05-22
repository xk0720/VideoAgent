"""Rewards used by RL training.

Three primitives, composable:
    • CompositeReward       — α·RM + β·m1..m6 + γ·beat (env step reward)
    • EditingQualityRMTrainer — fine-tunes a Bradley-Terry RM on preferences
    • HindsightCriticRefiner  — HCAPO-style post-hoc Q-value smoothing

All are mock-first: trainers run in "stub" mode without torch.
"""
from .composite import CompositeReward
from .editing_quality_rm import EditingQualityRMTrainer, EditingQualityRM
from .hindsight import HindsightCriticRefiner

__all__ = ["CompositeReward", "EditingQualityRMTrainer", "EditingQualityRM",
           "HindsightCriticRefiner"]
