# WaveSpeed API — Engineering Reference (verified 2026-07-02)

> Fetched from the official wavespeed.ai docs + model pages (each schema cites
> its page). Drives `models/video_gen_backends.py` + `audio_gen_backends.py`.
> Anything marked UNVERIFIED could not be confirmed from an official page.

## 0. Recommended best-quality model per capability (cost no object)

| Capability | Model id (POST `/api/v3/{model-id}`) | Key constraints |
|---|---|---|
| t2v (best) | `bytedance/seedance-2.0/text-to-video` | duration int 4–15 s; resolution 480p/720p/1080p/4k; native audio |
| i2v (best) | `bytedance/seedance-2.0/image-to-video` | `image` + optional `last_image` → native first+last frame |
| first+last frame | seedance-2.0 i2v w/ `last_image`, or `vidu/q3/start-end-to-video` (both frames required) | legacy `wavespeed-ai/wan-flf2v` still live |
| video edit | `bytedance/seedance-2.0/video-edit` | input ≤15 s (trimmed); billed input+output seconds |
| video extend | `bytedance/seedance-2.0/video-extend` | 4–15 s extension; whole input video conditions the continuation |
| foley | `wavespeed-ai/hunyuan-video-foley` (48 kHz, $0.05/run) | `mmaudio-v2` still live; seedance-2.0/kling-3.0 generate NATIVE audio |
| TTS | `minimax/speech-2.6-hd` ($0.10/1k chars) | `speech-2.6-turbo` faster/cheaper; 2.5-turbo-preview still live |
| runner-up t2v/i2v | `kwaivgi/kling-v3.0-4k/*` (native 4K, 3–15 s, `end_image`) | $0.42/s |

**Critical:** 2026 flagship ids have THREE path segments
(`bytedance/seedance-2.0/image-to-video`). `model-id` must be a free-form path
in the client — a two-segment `{provider}/{model}` builder can't address them.

## 1. Protocol (unchanged, confirmed)

- Submit: `POST https://api.wavespeed.ai/api/v3/{model-id}`, Bearer auth, JSON.
- Poll: `GET /api/v3/predictions/{id}/result` (the submit response's
  `data.urls.get` carries the exact URL — prefer it verbatim).
- Envelope: `{"code":200,"message":"success","data":{...}}`; result data:
  `{id, status: created|processing|completed|failed, outputs:[url...], error,
  timings, has_nsfw_contents}`.
- **Media upload** (use instead of base64): `POST /api/v3/media/upload/binary`,
  multipart field `file`, ≤300 MB (images JPG/PNG/WebP/…, videos MP4/MOV/…).
  Response `data.download_url` → pass as the model's image/video input.
  gen4-aleph REQUIRES a public URL (base64 data URI → 400). No JSON-body size
  cap published (UNVERIFIED) — treat large data URIs as a 400 risk everywhere.
- Optional common params: `webhook_url`, `enable_sync_mode`.

## 2. Verified payload templates (as implemented)

### seedance-2.0 family (default)
- t2v: `{prompt*, aspect_ratio: 16:9|9:16|4:3|3:4|1:1|21:9, resolution:
  480p|720p|1080p|4k (def 720p), duration: 4–15 (def 5), generate_audio (def
  true), reference_images[], reference_videos[], reference_audios[]}`
- i2v: `{prompt*, image* (URL), last_image (URL — end frame), aspect_ratio
  (def adaptive), resolution, duration, generate_audio}`
- video-edit: `{prompt*, video* (URL, ≤15 s), aspect_ratio, resolution,
  duration (auto-detected if omitted), reference_images[], generate_audio
  (false = keep input audio)}`
- video-extend: `{prompt*, video* (URL), last_image (opt target), resolution,
  duration 4–15}`

### Legacy seedance v1 (UniVA's route, still live)
- t2v `bytedance/seedance-v1-pro-t2v-480p`: `{prompt*, aspect_ratio:
  16:9|9:16|1:1, duration: 5|10 ONLY (2 → 400), camera_fixed, seed (-1 =
  random)}`. Resolution baked into the id (-480p/-720p/-1080p ids).
- i2v `…-i2v-480p`: `{image* (URL), prompt, last_image (opt), duration:
  5|10|15|20 (15/20 UNVERIFIED re-check), camera_fixed, seed}`.

### wan-flf2v (legacy first+last, still live)
`{first_image*, last_image* (URL or data URI), prompt, negative_prompt,
duration: 5|10, size: "832*480" (ASTERISK, not x; 720p = "1280*720"),
num_inference_steps: 30, guidance_scale: 5, seed, enable_safety_checker}`.
No wan-2.2 flf2v exists; newer first+last = seedance-2.0/kling-3.0 i2v.

### runwayml/gen4-aleph (still live, $0.18/s)
`{prompt*, video* (public URL ONLY — base64 400s), aspect_ratio:
16:9|4:3|1:1|3:4|9:16, reference_image (opt)}`.

### wavespeed-ai/wan-2.1-14b-vace (still live; no 2.2 version)
`{prompt*, images: [] (empty IS accepted, up to 5 refs), video (opt URL),
mask_video, mask_image, first_image, last_image, task: depth/…,
negative_prompt, duration (def 5), size: 832*480|1280*720|720*1280,
num_inference_steps: 30, guidance_scale: 5, flow_shift: 16, context_scale: 1,
seed, enable_fast_mode}`.

### vidu/q3/start-end-to-video (dedicated first+last)
`{prompt*, image*, last_image*, duration (def 5), resolution:
540p|720p|1080p, bgm, generate_audio, movement_amplitude: "auto", seed}`.

### Audio
- `wavespeed-ai/hunyuan-video-foley`: `{video* (URL or base64), prompt (opt),
  seed (opt)}` — 48 kHz.
- `wavespeed-ai/mmaudio-v2`: `{prompt*, video* (URL), negative_prompt,
  duration, num_inference_steps, guidance_scale, mask_away_clip}`.
- `minimax/speech-2.6-hd`: `{text*, voice_id* (case-sensitive: Wise_Woman,
  Friendly_Person, Deep_Voice_Man, …), speed 0.8–1.2, volume, pitch, emotion,
  sample_rate, bitrate, format, channel, language_boost,
  english_normalization}`.

## 3. Error taxonomy

- HTTP 400 = invalid request, details in body (exact JSON shape UNVERIFIED —
  our client surfaces the body since commit 1aa88d4).
- HTTP 401 = bad key, wrong Bearer format, or a key generated BEFORE the first
  top-up (keys don't activate until funded).
- HTTP 429 = tier rate limits (Bronze 5 videos/min / 3 concurrent; Silver
  60/100; Gold 120/200).
- Application codes in the prediction record (`status: failed`): 1200 content
  moderation · 1400 missing param · 1401 invalid param · 1402 media URL
  unreachable · 1403/1405 task failed · 5000/5003/5004 server/timeout.

Likely causes of our past 400s, ranked:
1. invalid `duration` (frames-as-seconds bug; 2 on a {5,10} model) — fixed by
   `_snap_duration` + timeline seconds conversion;
2. base64 data URIs to URL-only models (gen4-aleph) — fixed by `upload_media`;
3. wrong id shape for 3-segment 2026 models;
4. `size` with `x` instead of `*` on wan models;
5. missing required field (1400) / invalid enum (1401).

## 4. Client mapping (what we implemented)

- `WaveSpeedClient` defaults: model `bytedance/seedance-2.0/text-to-video`
  (i2v id derived by `/text-to-video → /image-to-video` or `-t2v- → -i2v-`),
  `resolution: 1080p`, `generate_audio: false`; `flf2v_model` / `edit_model` /
  `extend_model` configurable, legacy schemas auto-selected per id.
- `_snap_duration`: seedance-2.0 → clamp int [4,15]; legacy → {5,10} enum;
  config overrides `duration_range` / `allowed_durations`.
- `upload_media` (shared with audio): every local image/video input → official
  upload endpoint → URL; per-client cache on (path, mtime, size).
- `extend()`: dedicated video-extend endpoint (whole-video conditioning)
  replaced the decode-last-frame → i2v hack.
- Audio: foley default `hunyuan-video-foley` (mmaudio schema kept for the
  legacy id); TTS default `minimax/speech-2.6-hd`.
