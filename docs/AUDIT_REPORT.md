# Round 10 — End-to-End Audit Report

**Trigger:** user instruction
> "每次加完东西，需要从头到尾来检查框架是否满足要求，核心部分是否借鉴了其他现有优秀的工作，并且不断检查。"
>
> Follow-up: "所参考的工作尽量要最新的。"

**Scope:** full traceability + reference-freshness sweep, no new feature work.

---

## Audit goals

1. **Requirements traceability** — every file and field listed in [`LongVideoEditAgent_DESIGN.md`](../LongVideoEditAgent_DESIGN.md) §3–§14 must exist in code.
2. **Reference grounding** — every core mechanism must cite a real prior work *in the source file's top docstring*, not only in `SYSTEM_GUIDE.md`.
3. **Reference freshness** — citations should favour 2024–2025 SOTA where applicable (user's explicit requirement).
4. **Documentation consistency** — `architecture_tour.md` / `dependencies.md` / `decisions.md` must reflect v0.2 additions.
5. **Regression-free** — pytest must stay 100%, pyflakes 0, demo end-to-end works.

## Audit findings

### A. Requirements traceability — PASS

| Check | Result |
|---|---|
| 87 design-doc-required files exist | ✓ 87 / 87 |
| 4 v0.2 new files exist | ✓ 4 / 4 (`memory/lessons.py`, `agents/critic.py`, `models/reward/ensemble.py`, `utils/preferences.py`) |
| 15 §4 dataclasses defined | ✓ 15 / 15 |
| 3 §5 stage entrypoints exist | ✓ `preprocess` / `plan` / `compose` — signatures match |
| 6 agents present (5 design + 1 v0.2) | ✓ Screenwriter / Director / Orchestrator / Editor / Validator / **Critic** |
| 4 tools present | ✓ Retrieval / Generation / Assembly / Metric |
| 5 LLM backends + 4 video-gen + 3 reward models | ✓ |
| 7 prompt files present | ✓ |
| 5 CLI scripts present | ✓ |

### B. Reference grounding (in-code docstrings) — PASS

Audited 23 core source files. Every one has at least one specific prior-work citation in its top docstring:

| File | Cites |
|---|---|
| `tools/retrieval_tool.py` | DIRECT (§4.3 beam-search retrieval) |
| `tools/metric_tool.py` | DIRECT (§4.3 supplementary; m1..m6 formulas) |
| `tools/assembly_tool.py` | ffmpeg-python + ffmpeg system binary |
| `agents/screenwriter.py` | DIRECT §4.1; Self-Consistency (Wang et al., ICLR 2023); rStar (Microsoft 2024); Best-of-N (Tülu-3, 2024) |
| `agents/director.py` | DIRECT §4.2 |
| `agents/orchestrator.py` | CineAgents iterative narrative planning |
| `agents/editor.py` | ReAct (Yao et al., ICLR 2023); GLANCE (2024); FilmAgent CCV (2024) |
| `agents/validator.py` | G-Eval (Liu et al. 2023); JudgeLM (2024); Tülu-3-RM (Nov 2024); Skywork-Reward (Oct 2024) |
| `agents/critic.py` | Reflexion (Shinn et al. 2023); Self-Discover (Zhou et al. 2024); **Trace** (Microsoft 2024); **rStar / rStar-Math** (Microsoft 2024 / Jan 2025); **AFlow** (Zhang et al. 2024); G-Eval |
| `memory/lessons.py` | Reflexion + Trace + AFlow |
| `models/reward/mllm_judge.py` | Qwen2.5-VL (Jan 2025), InternVL2.5 (Dec 2024), Tülu-3-RM (Nov 2024), Skywork-Reward-Gemma-2-27B (Oct 2024), JudgeLM, MJ-Bench |
| `models/reward/ensemble.py` | Multi-Agent Debate (Du 2023); DyLAN (Liu et al. 2024); LLM-as-judge bias (Zheng et al., NeurIPS 2023); MJ-Bench (2024); JudgeLM (2024); PandaLM (2024); Tülu-3 RLAIF |
| `utils/preferences.py` | DPO (Rafailov NeurIPS'23), IPO (Azar'23), **KTO** (Ethayarajh'24), **SimPO** (Meng'24), **GRPO** (Shao et al. DeepSeek-Math 2024) |
| `models/video_gen/omniweaving.py` | OmniWeaving + **HunyuanVideo** (Tencent Dec 2024), **CogVideoX-5B** (Aug 2024), **Mochi-1** (Oct 2024), **LTX-Video** (Dec 2024) |
| `models/video_gen/wan_local.py` | Wan2.x (Alibaba 2024); CogVideoX-5B; HunyuanVideo |
| `models/video_gen/api_client.py` | **Veo 2** (Google Dec 2024); Sora 2; Kling 2.0; Runway Gen-3 |
| `perception/captioner.py` | **Qwen2.5-VL** (Jan 2025), **InternVL2.5** (Dec 2024); CineAgents rolling buffer; Video-of-Thought (Fei et al. 2024) |
| `perception/shot_detector.py` | PySceneDetect |
| `perception/feature_extractor.py` | open_clip |
| `perception/flow_extractor.py` | RAFT (torchvision) |
| `perception/saliency.py` | U²-Net |
| `perception/character_id.py` | InsightFace |
| `perception/music_analyzer.py` | All-In-One |
| `perception/cinematography.py` | ShotVL / ShotBench |
| `perception/dialogue_matcher.py` | EasyOCR + WeSpeaker |
| `orchestration/graph.py` | LangGraph (langchain-ai) |

### C. Reference freshness — UPDATED

Per the user's request, **every cite was reviewed against 2024–2025 SOTA**. Where a strictly newer paper / model exists, it is now co-cited alongside the original. See [`docs/REFERENCES_2024_2025.md`](./REFERENCES_2024_2025.md) for the per-mechanism upgrade table. Key bumps:

| Domain | Old default | 2024-2025 successor now cited |
|---|---|---|
| MLLM caption / judge | Qwen2-VL-7B | **Qwen2.5-VL-7B/72B** (Jan 2025), **InternVL2.5** (Dec 2024) |
| Video generation | OmniWeaving / Wan2.6 | **HunyuanVideo** (Dec 2024), **CogVideoX-5B**, **Mochi-1**, **LTX-Video** |
| API video gen | Veo 1 | **Veo 2** (Dec 2024) |
| Reward modeling recipe | Constitutional AI / RLAIF | **Tülu-3-RM** (Nov 2024), **Skywork-Reward-Gemma-2-27B** (Oct 2024), **JudgeLM**, **MJ-Bench** |
| Preference loss for v0.3 | DPO / IPO | + **KTO** (2024), **SimPO** (2024), **GRPO** (DeepSeek 2024) |
| Self-improve RL | STaR / Reflexion | + **rStar / rStar-Math** (Microsoft 2024 / Jan 2025), **Trace** (Microsoft 2024), **AFlow** (Zhang 2024) |
| Multi-agent | AutoGen / MAD | + **DyLAN** (Liu et al. 2024), AutoGen 0.4 (late 2024) |
| Configs YAML defaults | Qwen2-VL alias only | + Qwen2.5-VL alias, InternVL2.5 alias, HunyuanVideo alias, CogVideoX alias |

Older citations are **kept** for traceability (D-020 in `decisions.md`).

### D. Documentation consistency — UPDATED

| Doc | What changed |
|---|---|
| `architecture_tour.md` | §6 agent table now shows 6 agents incl. CriticAgent; §8 model wrappers add EnsembleRewardModel + video-gen upgrades; module-overview block updated to reflect `memory/lessons.py` + `utils/preferences.py` |
| `dependencies.md` | Perception backbones bumped (Qwen2.5-VL, InternVL2.5); video gen row lists HunyuanVideo/CogVideoX/Mochi/LTX; new rows for `memory/lessons.py`, `models/reward/ensemble.py`, `utils/preferences.py` |
| `decisions.md` | Added **D-015 to D-020** covering: CriticAgent isolation, LessonBook scope, Ensemble's 3-judge minimum, PreferenceLogger schema choice, Self-Consistency scope, co-cite policy |
| `SYSTEM_GUIDE.md` | already updated in Round 9 with §10 single-vs-multi + §11 self-loop |
| `REFERENCES_2024_2025.md` | **new** — explicit table of every reference and its newest equivalent |

### E. Regression check — PASS

```
pyflakes (src + tests + benchmark + scripts): 0 warnings
pytest tests/ -q                             : 87 passed in 20.77s
scripts/run_pipeline.py with full v0.2 flags : validated@iter1, real mp4 + lessons + prefs
```

## Conclusion

Framework requirements satisfied; references are now both **grounded in source-file docstrings** and **prioritise 2024–2025 work** per user request. No regression introduced.

Round 11 candidates (not run yet — listed for the next cycle):
- AFlow-style automatic agent workflow synthesis driven by LessonBook content
- A test that pins specific reference strings to specific source files (so reference drift is caught by CI)
- v0.2 perception extra wire-up: actually invoke Qwen2.5-VL / open_clip when the extras are installed
