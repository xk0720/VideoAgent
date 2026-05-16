# Reference Freshness Audit (2024-2025)

> User requirement: 引用尽量用最新工作。
>
> This table audits every cited work and notes the 2024-2025 state-of-the-art alternative we should cite (or have already added). When the older citation remains canonical (Reflexion, ReAct, DPO, etc.), we keep it but co-cite the newer descendant.

---

## A. Video generation backbones

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| OmniWeaving (2024) | **HunyuanVideo** (Tencent, Dec 2024) — open-weights 13B T2V, same Tencent lab → strongest "drop-in" upgrade. | `configs/models/video_gen.yaml`, `models/video_gen/omniweaving.py` docstring, SYSTEM_GUIDE |
| Wan2.6 | **Wan2.1 / Wan2.6** (Alibaba, 2024) + **CogVideoX-5B** (Zhipu, Aug 2024) + **Mochi-1** (Genmo, Oct 2024) + **LTX-Video** (Lightricks, Dec 2024) | Same files |
| Veo (Google) | **Veo 2** (Google, Dec 2024); via google-genai SDK. | `models/video_gen/api_client.py` |
| (none in design doc) | **Open-Sora 2.0** (HPCAITech, 2025-Q1) — open Sora replication, training-recipe available | SYSTEM_GUIDE §5.6 |

## B. MLLM / Caption backbones

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| Qwen2-VL-7B | **Qwen2.5-VL-7B / 72B** (Alibaba, Jan 2025) — stronger long-context video understanding | `configs/models/mllm.yaml`, `perception/captioner.py` docstring |
| GPT-4o | GPT-4o is still current; co-cite **Claude Sonnet 4.5** (current best on multimodal reasoning) and **Gemini 2.5 Pro** | `configs/models/mllm.yaml` |
| ShotVL/ShotBench | **ShotBench** (Vchitect, 2024) stays canonical; no 2025 successor yet | unchanged |

## C. Reward Modeling

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| Constitutional AI (Anthropic, 2022) | **Tülu-3-RM** (Allen AI, Nov 2024) — full open RM training recipe with preference data; **Skywork-Reward-Gemma-2-27B** (Skywork, Oct 2024) — top RewardBench score | SYSTEM_GUIDE §5.4; `models/reward/mllm_judge.py` |
| RLAIF (Lee et al., 2023) | Same — RLAIF remains the term-of-art; co-cite Tülu-3 as the modern open recipe | SYSTEM_GUIDE §5.4 |
| Process Reward Models (Lightman et al., 2023) | **PRM800K** + **rStar-Math** (Microsoft, Jan 2025) — stronger PRMs for math but methodology transfers | SYSTEM_GUIDE §5.4 |
| G-Eval (Liu et al., 2023) | **MJ-Bench** (2024), **JudgeLM** (2024) — purpose-trained judge models | SYSTEM_GUIDE §5.4 |

## D. RL training frameworks

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| PPO / verl | **GRPO** (Shao et al., DeepSeek-Math, 2024; used in DeepSeek-V3 / R1) — simpler than PPO, no separate value head | SYSTEM_GUIDE §5.3, roadmap v0.4 |
| RAGEN / StarPO | **RAGEN** still current; co-cite **OpenRLHF** (Hu et al., 2024) — easier setup for LLM-as-agent | SYSTEM_GUIDE §5.3 |
| (none) | **RLOO** (Ahmadian et al., 2024) — even simpler than GRPO, useful as baseline | SYSTEM_GUIDE §5.3 |

## E. Self-Improvement / Agentic Evolution

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| STaR (Zelikman, NeurIPS 2022) | **rStar** (Microsoft, 2024) + **rStar-Math** (Microsoft, Jan 2025) — self-play search-augmented reasoning | SYSTEM_GUIDE §5.3 |
| Reflexion (Shinn et al., 2023) | **Trace** (Microsoft, 2024) — gradient-style optimization over agent traces; **AFlow** (Zhang et al., 2024) — automatic agent workflow generation | SYSTEM_GUIDE §5.3, `memory/lessons.py` docstring |
| Voyager (2023) | **AgentFly** (Mou et al., 2024) + **AutoAgent** (2024) — modern open-ended agent training | SYSTEM_GUIDE §5.3 |
| Self-Refine (Madaan, 2023) | Still canonical; co-cite **Self-Discover** (Zhou et al., 2024 — already cited) | already in SYSTEM_GUIDE |

## F. Multi-agent frameworks

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| Multi-Agent Debate (Du et al., 2023) | **DyLAN** (Liu et al., 2024 — already cited) + **MAD-Bench** (2024 — for evaluation) | SYSTEM_GUIDE §5.2 |
| AutoGen (Microsoft, 2023) | **AutoGen 0.4** (Microsoft, late 2024) — event-driven rewrite; **MetaGPT** (already cited) | SYSTEM_GUIDE §5.2 |
| (none) | **AFlow** (Zhang et al., 2024) — meta-agent that designs workflows for sub-agents | SYSTEM_GUIDE §5.2 |
| (none) | **AgentBoard** (Ma et al., 2024) — multi-task agent benchmark | SYSTEM_GUIDE §11 |

## G. Preference learning (for our PreferenceLogger consumers)

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| DPO (Rafailov, 2023) | **IPO** (Azar, 2023 — already cited); **KTO** (Ethayarajh et al., 2024); **SimPO** (Meng et al., 2024) — newer preference-optimization losses | `utils/preferences.py` docstring, SYSTEM_GUIDE §11.1 |

## H. Video / multimodal agent works (newer than the original design doc)

| Older citation | 2024–2025 update | Where to add |
|---|---|---|
| (design doc lists DIRECT, FilmAgent, MovieAgent, GLANCE, CineAgents) | **VideoAgent** (Wang et al., 2024 — same-name paper for long-video QA); **Video-of-Thought** (Fei et al., 2024); **VideoTree** (Wang et al., 2024); **MMAgent** (2024) | SYSTEM_GUIDE §5.1 — add a §5.1.6 multi-modal agent block |

## I. Perception backbones (mostly stable)

| Older citation | 2024–2025 update | Where to update |
|---|---|---|
| RAFT (2020) | Still canonical for optical flow; **FlowFormer++** (CVPR 2023) is faster on some benches. | optional |
| U²-Net (2020) | Still canonical for saliency. | unchanged |
| All-In-One (mir-aidj, 2023) | Still the strongest one-shot music structure model. | unchanged |
| InsightFace | Still canonical. | unchanged |

---

## Action list (what to actually patch in this round)

Per-file edits I will make below:

1. `models/video_gen/omniweaving.py` — co-cite **HunyuanVideo** (same Tencent lab, Dec 2024) as the preferred recent upgrade
2. `models/video_gen/wan_local.py` — mention **Wan2.1** (Dec 2024) + **CogVideoX-5B** as alternatives
3. `models/video_gen/api_client.py` — bump to **Veo 2**
4. `perception/captioner.py` — mention **Qwen2.5-VL** (Jan 2025) as the upgrade target
5. `configs/models/mllm.yaml` — add Qwen2.5-VL alias
6. `configs/models/video_gen.yaml` — add HunyuanVideo alias
7. `memory/lessons.py` — co-cite **Trace** (Microsoft 2024) + **AFlow** (2024)
8. `models/reward/mllm_judge.py` — co-cite **Tülu-3-RM** + **Skywork-Reward**
9. `models/reward/ensemble.py` — co-cite **JudgeLM**, **MJ-Bench**
10. `utils/preferences.py` — co-cite **KTO** + **SimPO**
11. `tools/retrieval_tool.py` — keep DIRECT cite; no newer beam-search-for-editing work
12. `agents/critic.py` — co-cite **Trace** (gradient-style trace optimization)
13. `agents/screenwriter.py` — co-cite **rStar** (sampling + selection)
14. `docs/SYSTEM_GUIDE.md` — bump §5 with 2024-2025 alternatives; add §5.4 GRPO/RLOO/OpenRLHF; §5.1.6 video agent block
15. `docs/dependencies.md` — update real-OSS table with HunyuanVideo, Qwen2.5-VL
16. `docs/decisions.md` — add D-015 "why we co-cite older + newer" and D-016 "default backbones bumped to 2024-2025"

These are pure documentation / docstring patches — no behavioural change, no test regression risk.
