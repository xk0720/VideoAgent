# Open-source Dependency Map

For every module that does non-trivial work, this table lists:
  • **Real OSS library** it will use (real PyPI package or vetted GitHub repo).
  • **Mock status in v0.1** — whether the module currently uses real or mock backend.
  • **v0.2 swap-in plan** — concrete next step.

Library names are exactly as they appear on PyPI; bare GitHub URLs are
repositories that don't (yet) publish a PyPI package.

---

## Stage 1 · perception

| File | Real OSS library (2024-2025 preferred) | PyPI / Repo | v0.1 state | v0.2 swap-in |
|---|---|---|---|---|
| `perception/shot_detector.py` | **PySceneDetect** | `scenedetect` · github.com/Breakthrough/PySceneDetect | Mock (even split into 2 s shots) | `pip install 'longvideoagent[perception]'`, flip `mocks.perception=false` |
| `perception/feature_extractor.py` | **open_clip** (or transformers `CLIPModel`) | `open_clip_torch` · github.com/mlfoundations/open_clip | Mock (hash → unit vector) | sample keyframes via `utils.video_io.iter_frames`, mean-pool CLIP visual features |
| `perception/flow_extractor.py` | **RAFT** | `torchvision.models.optical_flow.raft_large` | Mock (32×32×2 random) | call `raft_large(weights="...")` on paired boundary frames |
| `perception/saliency.py` | **U²-Net** | github.com/xuebinqin/U-2-Net (no PyPI; vendor weights) | Mock (Gaussian blob) | load checkpoint via `U2NET_CHECKPOINT` env var |
| `perception/captioner.py` | **Qwen2.5-VL-7B/72B** (Jan 2025) / **InternVL2.5** (Dec 2024) / GPT-4o / Claude Sonnet 4.5 | `transformers` · openai · anthropic | Mock templated caption | route through MLLM client; carry rolling-buffer (CineAgents-style; same scheme as Video-of-Thought 2024) |
| `perception/character_id.py` | **InsightFace** + (re-id) **SOLIDER** | `insightface` · github.com/tinyvision/SOLIDER | Mock single char_0 | face detect → embed → agglomerative cluster |
| `perception/dialogue_matcher.py` | **EasyOCR** + **WeSpeaker** | `easyocr` · `wespeaker` | Mock `None` | OCR subtitle band + voiceprint cluster |
| `perception/cinematography.py` | **ShotVL / ShotBench** | huggingface.co/Vchitect/ShotBench-3B (via `transformers`) | Mock varied tags | `AutoModelForCausalLM.from_pretrained(...)` + label decoding |
| `perception/music_analyzer.py` | **All-In-One** | `allin1` · github.com/mir-aidj/all-in-one | Mock (deterministic 4-section profile) | `allin1.analyze(path)` |

## Memory

| File | Real OSS library | PyPI / Repo | v0.1 state |
|---|---|---|---|
| `memory/store.py` | stdlib `sqlite3` + **FAISS** | `faiss-cpu` · github.com/facebookresearch/faiss | Real SQLite. FAISS optional → numpy fallback. |
| `memory/builder.py` | numpy only | — | Real (heuristic event grouping; v0.2 swaps in BaSSL / LLM summarisation) |
| `memory/retriever.py` | numpy + injected text encoder | — | Real beam search; encoder mock-by-default |
| **`memory/lessons.py` (v0.2)** | stdlib only; conceptually **Reflexion** (Shinn et al., 2023) + **Trace** (Microsoft 2024) + **AFlow** (2024) | — | Real append-only JSONL store; keyword-overlap retrieval |

## Stage 2 · agents + orchestration

| File | Real OSS library | PyPI / Repo | v0.1 state |
|---|---|---|---|
| `agents/*.py` | abstract over BaseLLMClient | — | Real agent logic. LLM = MockLLMClient. |
| `orchestration/graph.py` | **LangGraph** | `langgraph` · langchain-ai.github.io/langgraph | Python state-machine fallback; LangGraph used when `plan.use_langgraph=true` |
| `orchestration/messages.py` | aligned with LangChain Core | `langchain-core` | Dataclass only (no runtime dep yet) |

## Stage 3 · tools + models

| File | Real OSS library | PyPI / Repo | v0.1 state |
|---|---|---|---|
| `tools/retrieval_tool.py` | numpy + scipy | `scipy` | Real beam search; tied to mock CLIP encoder |
| `tools/generation_tool.py` | injected BaseVideoGenClient | — | Real wrapper; MockVideoGenClient writes coloured clip |
| `tools/assembly_tool.py` | **ffmpeg-python** + system ffmpeg | `ffmpeg-python` · github.com/kkroening/ffmpeg-python | **Always real** (required by v0.1 acceptance §11) |
| `tools/metric_tool.py` | numpy + **scipy** | `scipy` (Wasserstein, Spearman) | Real |
| `models/llm/openai_client.py` | **OpenAI SDK** | `openai` · github.com/openai/openai-python | Real surface (lazy import; only used when `mocks.llm=false`) |
| `models/llm/anthropic_client.py` | **Anthropic SDK** | `anthropic` · github.com/anthropics/anthropic-sdk-python | Real surface |
| `models/llm/deepseek_client.py` | OpenAI-compatible SDK | `openai` (via DEEPSEEK_BASE_URL) | Real surface |
| `models/llm/vllm_local.py` | OpenAI-compatible SDK + **vLLM** server | `openai` client; `vllm` for the server side | Real surface |
| `models/video_gen/omniweaving.py` | **OmniWeaving** (Tencent 2024); 2024-2025 upgrades: **HunyuanVideo** (Tencent Dec 2024), **CogVideoX-5B** (Zhipu Aug 2024), **Mochi-1** (Genmo Oct 2024), **LTX-Video** (Lightricks Dec 2024) | github.com/Tencent-Hunyuan/{OmniWeaving,HunyuanVideo} | NotImplementedError (use MockVideoGenClient until weights are wired) |
| `models/video_gen/wan_local.py` | **Wan2.x** (Alibaba 2024) | github.com/Wan-Video (HF weights) | NotImplementedError |
| `models/video_gen/api_client.py` | **Veo 2** (Google Dec 2024) via `google-genai`; alt: Sora 2 / Kling 2.0 / Runway Gen-3 | `google-genai` | NotImplementedError |
| `models/reward/mllm_judge.py` | injected LLM; v0.3 → **Tülu-3-RM** (Allen AI Nov 2024) / **Skywork-Reward-Gemma-2-27B** (Oct 2024) / **JudgeLM** (2024) | — | Mock-backed when LLM is the mock; real path is ready |
| **`models/reward/ensemble.py` (v0.2)** | conceptually **Multi-Agent Debate** (Du 2023), **DyLAN** (Liu 2024), **MJ-Bench** (2024), **PandaLM** (2024); LLM-as-judge bias paper (Zheng et al., NeurIPS 2023) | — | Real — K judges, mean ± std, quorum |

## Infrastructure

| File | Real OSS library | PyPI |
|---|---|---|
| `config.py` | **pydantic v2** + **PyYAML** + **python-dotenv** | `pydantic` · `PyYAML` · `python-dotenv` |
| `logging.py` | **loguru** | `loguru` |
| `utils/trajectory.py` | stdlib only (JSONL) — schema compatible with RAGEN / HF datasets | — |
| **`utils/preferences.py` (v0.2)** | stdlib JSONL; schema compatible with HuggingFace `trl.DPOTrainer`; supports **DPO** (Rafailov NeurIPS'23), **IPO** (Azar'23), **KTO** (Ethayarajh'24), **SimPO** (Meng'24), **GRPO** (Shao et al. DeepSeek-Math 2024) | — |
| `utils/video_io.py` | **opencv-python** + **ffmpeg-python** | `opencv-python` · `ffmpeg-python` |
| `utils/audio_io.py` | **librosa** (lazy) | `librosa` |
| `cli.py` / `scripts_impl.py` | stdlib argparse + **rich** | `rich` |

## Tests

| File | Real OSS library | PyPI |
|---|---|---|
| `tests/conftest.py` | **pytest** + ffmpeg via `utils.video_io` | `pytest` |
| Other unit / integration tests | pytest + numpy | `pytest` |

---

## "Do I need to install X?" matrix

| Task | Required `pip install -e` extra |
|---|---|
| Run `pytest tests/` | (base only) |
| Run `lva-run` with mocks=true | (base only; needs system `ffmpeg`) |
| Wire in real shot detection / CLIP / RAFT | `.[perception]` |
| Wire in real LLM calls | `.[llm]` (+ keys in `.env`) |
| Use LangGraph for Stage 2 | `.[orchestration]` |
| Music structure analysis with allin1 | `.[music]` |
| Hosted Veo video-gen | `.[video_gen]` (+ `GOOGLE_API_KEY`) |
| All of the above | `.[all]` |
