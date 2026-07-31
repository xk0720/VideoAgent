All reading complete. Here is the report.

---

# CutClaw deep-read (GVCLab/CutClaw, arXiv 2603.29664)

Clone: `/Users/kevin/.claude/jobs/5bd83ba8/tmp/CutClaw` (commit `db48d08b`, 2026-04-17). Paper: "CutClaw: Agentic Hours-Long Video Editing via Music Synchronization" (Beijing Jiaotong Univ / GVC Lab Great Bay Univ / Tencent ARC).

## (a) End-to-end and architecture

**Input:** one long source video (film or vlog, hours OK) + one music track (mp3/wav) + a one-line text instruction (+ optional SRT). **Output:** `shot_plan_*.json` (per-music-segment storyboard) + `shot_point_*.json` (exact source timestamps) → ffmpeg-rendered music-synced montage (optionally 9:16 auto-cropped, hook-dialogue intro). Orchestrator: `local_run.py`; Streamlit UI: `app.py`.

**Phase 1 — Deconstruction** (3 parallel threads, `local_run.py:384-394`): (A) ASR (whisper.cpp local or LiteLLM cloud, `src/video/preprocess/asr.py`) + pyannote diarization + LLM character ID (`src/video/deconstruction/get_character.py`); (B) shot detection (scenedetect default; autoshot/transnetv2/Qwen3VL options, `src/config.py:82`) → VLM shot captions (`video_caption.py`) → scene merge via sentence-transformers similarity, scenes capped at 300s (`scene_merge.py`, `config.py:102-110`) → per-scene VLM analysis (`scene_analysis_video.py`); (C) audio analysis (below). All cached on disk under `Output/` — one-time cost per video, reused across edits (`readme.md:292`).

**Phase 2 — Multi-agent editing** (all LLM calls through LiteLLM): **Screenwriter** (`src/Screenwriter_scene_short.py:1014`) selects a music span, then generates `shot_plan`; **EditorCoreAgent** (`src/core.py:803`) is a THINK→ACT→OBSERVE tool-calling loop (tools defined via pydantic schemas, `src/func_call_shema.py`): `semantic_neighborhood_retrieval` (scene search restricted to recommended scene ±3, `config.py:413`), `fine_grained_shot_trimming` (frame-level VLM analysis + protagonist bbox detection), `review_clip` (overlap check, `src/Reviewer.py:58`), `commit` (`src/core.py:124`). A **ParallelShotOrchestrator** (`src/core.py:1550`) runs 4 workers with conflict-rerun rounds. **Reviewer** (`src/Reviewer.py:245`) does VLM face-quality/protagonist-ratio checks. Recommended models (`readme.md:207-219`): video = Gemini-3/Qwen3.5/GPT-5.3; audio = Gemini-3 (default fallback `openai/Qwen3-Omni-30B-A3B-Instruct`, `src/audio/litellm_client.py:46`); agent = MiniMax-2.7/Kimi-2.5/Claude-4.5.

**Phase 3 — Render:** `render/render_video.py` (ffmpeg, detail in (b)).

## (b) Core mechanism: music analysis → edit mapping

**Libraries: madmom + aubio. No librosa, no allin1** (`requirements.txt:19-20`; explicit `load_audio_no_librosa` in `src/audio/audio_utils.py`). `src/audio/audio_Madmom.py:6-124` carries monkey-patches to keep madmom 0.16.1 alive on Python 3.10+/NumPy 2.x.

**Three signal detectors** (`audio_Madmom.py`, unified API `src/audio/madmom_api.py:87`), configured by `AUDIO_DETECTION_METHODS = ["downbeat","pitch","mel_energy"]` (`config.py:208`):
- **Downbeat** (`audio_Madmom.py:600-692`): `RNNDownBeatProcessor` activations → patched `DBNDownBeatTrackingProcessor` (beats_per_bar=[4], BPM 55–215) → keep bar-position-1 beats: `downbeats = beat_info[beat_info[:, 1] == 1][:, 0]`; the DBN activation at each downbeat becomes its `intensity`.
- **Pitch** (`audio_Madmom.py:248-315`): aubio `pitch("yin", 4096, 512)` in MIDI units, confidence-thresholded.
- **Mel energy** (`audio_Madmom.py:320-387`): aubio `pvoc` + 40-filter Slaney mel filterbank, per-frame total energy, peaks above 0.3×max.

All three go through greedy 1-D NMS (`nms_1d`, `audio_Madmom.py:132`: sort by strength desc, accept if ≥ min_distance from accepted points), per-type min-max normalization (`normalize_intensity_by_type`, line 695), and a weighted **composite score** (`compute_composite_score`, line 1019, weights all 1.0 by default).

**Structure segmentation is LLM-based, not signal-based**: the whole track is sent to the audio LLM with `AUDIO_OVERALL_PROMPT` (`src/audio/audio_caption_madmom.py:47-93`) asking for Intro/Verse/Chorus/Bridge/Build-up/Drop/Outro sections with MM:SS timestamps, 15–45s each; responses are validated (within-duration + 5–60s duration checks, up to 5 retries, auto-clamp) at `audio_caption_madmom.py:759-873`, then **section boundaries are snapped to the nearest detected keypoint** (Step 2.6, lines 945-1040).

**Cut-point selection = beat-constrained shot budgeting** (`filter_by_sections`, `audio_Madmom.py:1082`): a global shot budget `AUDIO_TOTAL_SHOTS=200` is allocated to sections **proportionally to keypoint density** (`allocated = max(1, round(total_shots * ratio))`, line 1173), then per section the top-k composite-score keypoints are kept; gaps > max_segment_duration are back-filled from raw keypoints nearest to ideal midpoints; adjacent points closer than min_segment_duration are merged keeping the higher score. Sub-segment boundaries per section use greedy strongest-first acceptance (`audio_caption_madmom.py:1090-1112`):

```python
for kp in section_all_kps:            # sorted by intensity desc
    if all(abs(t - a) >= min_segment_duration for a in accepted):
        accepted.append(t)
```

**Level-2 captions:** every inter-keypoint sub-segment is cut to wav (`segment_audio_file`) and captioned concurrently by the audio LLM with `AUDIO_SEG_KEYPOINT_PROMPT` (`audio_caption_madmom.py:95-109`) → `{summary, emotion, energy "1-10 + trend", rhythm "BPM + feel"}`.

**Mapping edits onto the music timeline:** Screenwriter first LLM-selects one music span (`SELECT_AUDIO_SEGMENT_PROMPT`, `src/prompt.py:1033`: "Prefer high-energy… Chorus/Drop/Build-up unless instruction suggests otherwise"; 5–15s span for short-video mode, `config.py:331-332`). `GENERATE_SHOT_PLAN_PROMPT` (`src/prompt.py:610-796`) then maps **each music sub-segment to exactly ONE shot** with `"time_duration": <float, EXACT duration from music segment>` and explicit energy-matching rules (line 722: "Fast-paced music → strong motion… Drop/climax → explosive movement… Breathing space → negative space"). The Editor agent finds real footage of that exact duration; `review_finish` (`src/Reviewer.py:110`) rejects duration mismatch beyond ±1s (`ALLOW_DURATION_TOLERANCE`) and `commit` auto-trims ≤1s overshoot (`src/core.py:202-215`). **So beat-sync is achieved by construction: cut boundaries ARE the filtered beat/energy keypoints, and clip durations are forced to equal inter-keypoint durations.**

**Render** (`render/render_video.py:446-1130`): per-clip ffmpeg extraction (with scene-cut snapping `adjust_clip_for_scene_cuts` line 239, and protagonist-bbox-driven dynamic 9:16 crop centers line 280) → concat demuxer → BGM mix: `atrim=start=<selected_audio_start>:duration=…`, `loudnorm=I=-18:LRA=11:TP=-1.5` on both stems, `amix=inputs=2:duration=longest:normalize=0`, BGM ducking during hook dialogue via `volume='if(lt(t,{hook_dur}),{0.5},1.0)'` (lines 971, 1008-1040).

## (c) Hours-long scale

Sampling at 2 fps / 240px short side via decord (`config.py:35-48`); hierarchical compression footage→shots→scenes(≤300s)→scene summaries; 50-clip summary batches (`config.py:113`); everything cached per video id; agent search confined to recommended_scenes ±3 so context stays small; 4-way parallel shot selection; concurrent VLM/audio API batches (64 captions, 8 audio). The roadmap concedes cost is the pain point: ARC-Chapter integration and a "Low-Cost Mode" that reads only relevant footage are **unimplemented TODOs** (`readme.md:59-63`).

## (d) Directly reusable for Maestro (reverse problem: footage exists, add fitting music)

1. **The whole signal stack is liftable as-is and free (local CPU):** `src/audio/audio_Madmom.py` + `src/audio/madmom_api.py` give downbeat grid + energy/pitch peaks with intensities for any BGM candidate. In Maestro's ffmpeg assemble step, snap concat cut points to nearest downbeat by micro-trimming each generated shot's tail (a few hundred ms) — reuse `_find_split_points_near_midpoints` (`audio_caption_madmom.py:398-477`) / `find_nearest_snap_point` (line 955) logic directly. Only pip pain: madmom/aubio (patches already written in this repo).
2. **Music-span selection inverts cleanly:** run the Gemini structure+caption pass once per music candidate (cheap: one whole-track call + N sub-segment calls on 16kHz/32k mono mp3, `litellm_client.py:66`), producing `{emotion, energy, rhythm}` per segment — the exact vocabulary Maestro's brain already emits per shot. `SELECT_AUDIO_SEGMENT_PROMPT` (`prompt.py:1033`) is reusable nearly verbatim with roles inverted (given video plan → pick music span/track). Maestro can even skip captions and use only the free madmom downbeat grid, since its brain already knows the story arc.
3. **Tempo fitting for fixed-duration generated shots:** madmom's `beat_info` yields BPM; pick library music whose downbeat interval divides Maestro's shot length, or nudge with ffmpeg `atempo` — this makes cut-on-beat exact with zero re-generation.
4. **The ffmpeg audio recipe** (`render_video.py:940-1108`) — `atrim` + dual `loudnorm` + `amix normalize=0` + time-windowed `volume` ducking expression — is exactly the BGM+dialogue mix-with-ducking Maestro needs; copy it.
5. **Proportional pacing allocation** (`filter_by_sections`): map chorus/drop sections → more/shorter Maestro shots; a good prior for the brain's shot-length planning when music is chosen before generation.

## (e) Honest limits

- **Music is an input, never selected or generated**: CutClaw picks a span *within one user-supplied track*; no library retrieval, no music generation, no video→music matching. The reverse direction is on us.
- **No speech synthesis, no lip-sync**: ASR/diarization only *understands* source footage; "hook dialogue" (`Screenwriter_scene_short.py:793`, `render_video.py:47`) just replays original movie audio with subtitles before the BGM starts.
- **"Sync" = cut placement + LLM vibe-matching**; no quantitative audio-visual alignment (no motion-energy-to-beat scoring); quality rides on VLM captions and the ±1s duration tolerance, so cuts can drift up to ~1s off beat when footage doesn't fit.
- **madmom is effectively unmaintained** (needs the compat patches at `audio_Madmom.py:6-124`); heavy deps (torch 2.8, pyannote) for what Maestro needs (just beat tracking).
- **Cost/latency**: full deconstruction of all footage is required today (low-cost mode is a roadmap TODO); up to 20 agent iterations per shot; large concurrent VLM fan-out.
- **Config inconsistencies**: `AUDIO_MIN/MAX_SEGMENT_DURATION` = 0.1/2.0s in `config.py:279-282` vs "3.0/5.0" in `readme.md:247-248` vs 3.0/30.0 fallbacks in `audio_caption_madmom.py:641-644`; `local_run.py:451` contains a stray `w` syntax error on main. Output is a short montage (default short-mode span 5–15s) — "hours-long" refers to *input* footage, not output.