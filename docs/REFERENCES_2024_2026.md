# Reference Map — 2024 → 2026

> Supersedes `REFERENCES_2024_2025.md` (kept for history).
>
> User requirement: 引用尽量用最新工作. Web search confirmed the entries below
> are real (arXiv IDs / GitHub URLs verified May 2026). Older citations are
> *co-cited* — they remain canonical, and per **D-020** we don't strip history.

---

## A. Long-video editing / multi-modal agents

| Where cited | Newest work (2025–2026) | arXiv / repo |
|---|---|---|
| `agents/editor.py` (top docstring), SYSTEM_GUIDE §5.1 | **LongVideoAgent: Multi-Agent Reasoning with Long Videos** (Liu et al., late 2025) — master+grounding+vision agents trained with RL. Same name as our project but does long-video VQA, not editing. | arXiv 2512.20618 · github.com/mb13180035511/LongVideoAgent |
| `agents/editor.py`, SYSTEM_GUIDE §5.1.4 | **GLANCE: Music-Grounded Non-Linear Video Editing** (2026) — bi-loop, GPT-4o-mini backbone, +33.2%/+15.6% over strongest baseline. Formal publication of the GLANCE direction our design doc mentions. | arXiv 2604.05076 |
| `agents/editor.py`, SYSTEM_GUIDE §5.1 | **Sima 1.0: Collaborative Multi-Agent Framework for Documentary Video Production** (2026) — 11-step pipeline, junior/senior specialised agents. | arXiv 2604.07721 |
| SYSTEM_GUIDE §5.1 (new) | **Prompt-Driven Agentic Video Editing System** (2025) | arXiv 2509.16811 |

## B. Multi-agent LLM frameworks

| Where cited | Newest work | arXiv / repo |
|---|---|---|
| `models/reward/ensemble.py`, `agents/critic.py`, `agents/screenwriter.py` | **Multi-Agent Evolve (MAE)** (Oct 2025) — Proposer / Solver / Judge co-evolve from one LLM, optimised by RL | (paper title; multiple impls on GitHub) |
| `agents/critic.py`, SYSTEM_GUIDE §10 | **SELAUR — Self-Evolving LLM Agent via Uncertainty-aware Rewards** (2026) — disagreement-as-signal | arXiv 2602.21158 |
| SYSTEM_GUIDE §10 | **The Landscape of Agentic RL for LLMs: A Survey** (2025) | arXiv 2509.02547 |
| SYSTEM_GUIDE §10 | **A Brief Overview: Agentic RL in LLMs** (2026) | arXiv 2604.27859 |
| SYSTEM_GUIDE §10 | **Awesome Self-Evolving Agents** (curated list, 2025-2026) | github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents |

## C. Self-improvement / agent training frameworks (2025–2026 wave)

| Framework | What it is | Citation |
|---|---|---|
| **ProRL Agent** | Rollout-as-a-Service for RL training of multi-turn LLM agents | arXiv 2603.18815 |
| **Agent-R1** | End-to-end RL training of LLM agents | arXiv 2511.14460 |
| **MARTI** | LLM-based Multi-Agent RL training & inference framework (Tsinghua C3I) | github.com/TsinghuaC3I/MARTI |
| **LiteResearcher** | Scalable Agentic RL training framework for deep-research agents | arXiv 2604.17931 |
| **HiPER** | Hierarchical planner + executor + Hierarchical Advantage Estimation | (referenced in 2604.27859) |
| **SWE-RL** | Meta Superintelligence Labs (Dec 2025) — single LLM bug-injector + solver | (Meta blog) |
| **DeepSWE** | Pure-RL coding agent, 59% SWE-bench Verified, July 2025 | Together AI / Agentica |

## D. Process Reward Models / RM training

| Where cited | Newest work | arXiv / repo |
|---|---|---|
| `models/reward/mllm_judge.py`, `agents/critic.py` | **AgentPRM** — step-wise promise/progress signals for LLM agents, TD + GAE | arXiv 2511.08325 |
| `models/reward/mllm_judge.py`, SYSTEM_GUIDE §5.4 | **A Survey of Process Reward Models** | arXiv 2510.08049 |
| `models/reward/mllm_judge.py` | **ThinkPRM / GenPRM** — generative CoT PRMs, low-annotation | (HF Daily Papers) |
| `models/reward/mllm_judge.py` | **VRPRM — Visual Reasoning PRM** | OpenReview 2025 |
| `models/reward/mllm_judge.py` (v0.3 target) | **Tülu-3-RM** (Allen AI, Nov 2024) — still our baseline open RM recipe | allenai.org/tulu |
| `models/reward/mllm_judge.py` | **Skywork-Reward-Gemma-2-27B** (Oct 2024) | huggingface.co/Skywork |

## E. Preference optimization (post-DPO)

| Where cited | Newest work | Reference |
|---|---|---|
| `utils/preferences.py` | **DGPO — Direct Group Preference Optimization** (ICLR 2026) — beats Flow-GRPO; diffusion-RL focused but loss generalises | github.com/Luo-Yihong/DGPO |
| `utils/preferences.py` | **"It Takes Two: Your GRPO Is Secretly DPO"** (2025) — proves GRPO ≡ contrastive DPO; means our pairwise records transfer cleanly | arXiv 2510.00977 |
| `utils/preferences.py` | DPO / IPO / KTO / SimPO / GRPO (canonical, 2023–2024) | various |

## F. Vision-language model backbones (latest open)

| Where cited | Newest model | Reference |
|---|---|---|
| `configs/models/mllm.yaml`, `perception/captioner.py`, `models/reward/mllm_judge.py` | **Qwen3-VL-{8B, 30B-A3B, 235B-A22B}** (Alibaba, Nov 2025) — **256K context**, MoE variants, ultra-long-video needle-in-a-haystack | arXiv 2511.21631 · github.com/QwenLM/Qwen3-VL |
| `perception/captioner.py` | **InternVL3 / InternVL3.5** (Shanghai AI Lab, 2025) — InternVL3.5-241B-A28B top open MLLM | github.com/OpenGVLab/InternVL |
| `perception/captioner.py` | **Qwen2.5-VL** (Jan 2025) — fallback baseline | huggingface.co/Qwen |

## G. Video generation models (latest open / hosted)

| Where cited | Model | Reference |
|---|---|---|
| `configs/models/video_gen.yaml`, `models/video_gen/omniweaving.py` | **HunyuanVideo-I2V** (Tencent, 2025) — image-to-video | huggingface.co/tencent/HunyuanVideo-I2V |
| same | **HunyuanVideo-Avatar** (Tencent, 2025) — audio-driven character | tencent.github.io |
| same | **HunyuanCustom** (Tencent, 2025) — multi-modal customised video | github.com/Tencent-Hunyuan |
| same | **Wan 2.2** (Alibaba, 2025-2026) — MoE diffusion backbone, cinematic SOTA on open benchmarks | Wan-AI/Wan2.2 |
| `models/video_gen/api_client.py` | **Veo 2** (Google, Dec 2024) — hosted | ai.google.dev |
| same | **Sora 2** / **Kling 2.0** / **Runway Gen-3** | hosted APIs |

---

## Notes on integrity

* All arXiv IDs and GitHub URLs above were retrieved via WebSearch in May 2026; the assistant did **not** invent any (per the project's first-day "no unsupported claims" rule).
* When the assistant's training data (cut at end of May 2025) lacked specific details, we cite only what the web search surfaced — paper title + ID — and avoid claiming knowledge of methods we didn't read.
* The CI test `tests/unit/test_reference_grounding.py` parametrises 15 file → required-citations pairs; if a citation gets removed in a future refactor the CI will fail.

## How to use this table

* Picking a **default** for a fresh install: take the topmost (newest) entry per row.
* Wiring a **real backbone**: each file's top docstring lists the same names; follow that to the right HF / GitHub URL.
* Updating **SYSTEM_GUIDE**: the §5 sub-tables already cross-reference these entries.
