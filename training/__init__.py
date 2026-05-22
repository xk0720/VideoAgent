"""LongVideoEditAgent — training (RL / SFT / preference) subtree.

This directory implements the v0.3 → v0.5 roadmap from
[`docs/AGENTIC_RL_PROPOSAL.md`](../docs/AGENTIC_RL_PROPOSAL.md).

Architecture conventions (cross-referenced against real 2026 frameworks):
    • RAGEN's three-component split  — Environment Manager / Context Manager / Agent Proxy
        https://github.com/RAGEN-AI/RAGEN  (StarPO)
    • verl 0.7 AgentLoop              — server / client separation, asyncio rollouts
        https://verl.readthedocs.io/en/latest/start/agentic_rl.html
    • OpenRLHF AgentInstanceBase      — simple Python class interface
        https://openrlhf.readthedocs.io/en/latest/async_rl.html
    • EVA (arXiv 2603.22918)          — three-stage SFT → KTO → GRPO for video agents
    • Open Reward Standard (ORS)      — HTTP-based, tool-based agent⇄env protocol
        https://openrewardstandard.io
    • TRL v1.0 (HuggingFace, Apr 2026) — unified SFT/DPO/KTO/GRPO trainer surface
        https://huggingface.co/docs/trl

Everything in this subtree is **mock-first**: real torch / TRL / vLLM are
optional. The CPU-only test suite must pass without them.
"""
