"""Video-gen backend factory + graceful degradation (v0.4: no control signal —
conditioning = first_frame + reference_images only)."""
from pathlib import Path

import pytest

from maestro.models.video_gen import MockVideoGenClient, build_video_gen


def test_factory_returns_mock_by_default():
    assert isinstance(build_video_gen("mock-video-gen"), MockVideoGenClient)
    assert isinstance(build_video_gen(None), MockVideoGenClient)


def test_factory_dispatches_real_backends():
    omni = build_video_gen({"name": "omniweaving"})
    assert omni.__class__.__name__ == "OmniWeavingClient"
    wave = build_video_gen({"name": "wavespeed"})
    assert wave.__class__.__name__ == "WaveSpeedClient"
    # v0.4 conditioning contract: keyframe anchor + identity refs, NO control
    assert "first_frame" in omni.supported_conditions()
    assert "control_signal" not in omni.supported_conditions()


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_video_gen({"name": "definitely-not-a-model"})


def test_real_backend_guards_when_unwired(tmp_path: Path):
    client = build_video_gen({"name": "omniweaving"})
    with pytest.raises((RuntimeError, NotImplementedError)):
        client.generate("a ball falls", 1.0, tmp_path / "o.mp4")


def test_wavespeed_loud_without_api_key(tmp_path: Path, monkeypatch):
    """The API backend must fail LOUDLY when configured without a key —
    never silently fall back to mock output."""
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    client = build_video_gen({"name": "wavespeed"})
    with pytest.raises(RuntimeError, match="API key"):
        client.generate("a ball falls", 1.0, tmp_path / "w.mp4")


def test_mock_writes_metadata_without_control(tmp_path: Path):
    out = MockVideoGenClient().generate("a ball falls", 1.0, tmp_path / "m.mp4")
    body = out.read_text()
    assert "prompt=a ball falls" in body
    assert "control_signal" not in body      # the dead line stays dead


# ── Phase-2 capability registry seed (capabilities() + optional methods) ──
def test_capabilities_per_client():
    # Default base = t2v/i2v only.
    assert MockVideoGenClient().capabilities() == {"t2v", "i2v"}
    assert build_video_gen({"name": "omniweaving"}).capabilities() == {"t2v", "i2v"}
    # WaveSpeed declares the extras backed by optional methods, plus the v0.4
    # widened atom palette (depth_modify → vace/depth, style_transfer → runway).
    # "ref_video" (reference_videos motion conditioning) exists ONLY on the
    # seedance-2.0 family — legacy v1 ids have no such channel.
    wave = build_video_gen({"name": "wavespeed"})
    assert wave.capabilities() == {"t2v", "i2v", "flf2v", "edit", "extend",
                                   "depth", "style", "ref_video", "ref_images",
                                   "t2i", "multi_i2v"}
    legacy = build_video_gen({"name": "wavespeed",
                              "model_id": "bytedance/seedance-v1-pro-t2v-480p"})
    assert "ref_video" not in legacy.capabilities()
    # Extra capabilities are optional methods, NOT abstractmethods.
    assert hasattr(wave, "frame_to_frame") and hasattr(wave, "edit_video")
    assert hasattr(wave, "extend")
    assert hasattr(wave, "depth_modify") and hasattr(wave, "style_transfer")
    assert hasattr(wave, "repaint")
    assert not hasattr(MockVideoGenClient(), "frame_to_frame")


def test_wavespeed_flf2v_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    first = tmp_path / "a.jpg"; first.write_bytes(b"\xff\xd8\xff")
    last = tmp_path / "b.jpg"; last.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(RuntimeError, match="API key"):
        wave.frame_to_frame("morph", first, last, tmp_path / "o.mp4")


def test_wavespeed_edit_video_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        wave.edit_video("make it rain", vid, tmp_path / "o.mp4", backend="runway")


def test_wavespeed_edit_video_unknown_backend(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00")
    with pytest.raises(ValueError):
        wave.edit_video("x", vid, tmp_path / "o.mp4", backend="nope")


# ── widened atom palette (v0.4): depth_modify / style_transfer / repaint ──
def test_wavespeed_depth_modify_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        wave.depth_modify("replace the background with a beach", vid,
                          tmp_path / "o.mp4")


def test_wavespeed_style_transfer_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        wave.style_transfer("van gogh oil painting", vid, tmp_path / "o.mp4")


def test_wavespeed_depth_modify_routes_to_vace_depth(tmp_path: Path, monkeypatch):
    """depth_modify maps to edit_video(backend='vace', task='depth') — verified
    via a stubbed _run_task, NO network."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    captured = {}

    def _fake_run_task(model_id, payload, out_path):
        captured["model_id"] = model_id
        captured["payload"] = payload
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media", lambda p: "https://fake.host/in.mp4")
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    wave.depth_modify("replace bg", vid, tmp_path / "o.mp4")
    assert "vace" in captured["model_id"]
    assert captured["payload"]["task"] == "depth"
    assert captured["payload"]["video"] == "https://fake.host/in.mp4"


def test_wavespeed_style_transfer_routes_to_runway(tmp_path: Path, monkeypatch):
    """style_transfer maps to the runway route with a style-framed prompt."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    captured = {}

    def _fake_run_task(model_id, payload, out_path):
        captured["model_id"] = model_id
        captured["payload"] = payload
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media", lambda p: "https://fake.host/in.mp4")
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    wave.style_transfer("van gogh", vid, tmp_path / "o.mp4")
    assert "runway" in captured["model_id"]
    assert "van gogh" in captured["payload"]["prompt"]
    assert "style" in captured["payload"]["prompt"].lower()


def test_wavespeed_repaint_honest_skeleton_error(tmp_path: Path, monkeypatch):
    """repaint needs a segmentation backend (Sa2VA/SAM, GPU) — it must raise an
    HONEST error rather than fake a mask. Loud even with a key set."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="segmentation backend"):
        wave.repaint("a red car", vid, "car", tmp_path / "o.mp4")


def test_wavespeed_snap_duration_to_allowed_values():
    """WaveSpeed models constrain `duration` (seconds) and 400 on anything
    else. seedance-2.0 (default) = any int in [4, 15]; legacy = {5, 10} enum.
    Requests snap UP into the valid set, and a runaway value (e.g. the old
    frames-as-seconds bug sending 40) clamps instead of reaching the API."""
    wave = build_video_gen({"name": "wavespeed"})   # default = seedance-2.0
    assert wave._snap_duration(0.375) == 4
    assert wave._snap_duration(5) == 5
    assert wave._snap_duration(6.2) == 7
    assert wave._snap_duration(40) == 15
    legacy = build_video_gen({"name": "wavespeed",
                              "model_id": "bytedance/seedance-v1-pro-t2v-480p"})
    assert legacy._snap_duration(0.375) == 5
    assert legacy._snap_duration(6.2) == 10
    assert legacy._snap_duration(40) == 10
    custom = build_video_gen({"name": "wavespeed",
                              "model_id": "bytedance/seedance-v1-pro-t2v-480p",
                              "allowed_durations": [3, 6]})
    assert custom._snap_duration(4) == 6


def test_wavespeed_seedance2_payloads_match_official_docs(tmp_path: Path, monkeypatch):
    """Default family = seedance-2.0 (docs-verified 2026-07): t2v carries
    aspect_ratio + resolution + generate_audio (no legacy seed); i2v swaps the
    3-segment id to /image-to-video and passes the frame by UPLOADED URL."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    calls = []

    def _fake_run_task(model_id, payload, out_path):
        calls.append({"model_id": model_id, "payload": payload})
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media", lambda p: "https://fake.host/f.png")

    wave.generate("a ball falls", 5.0, tmp_path / "t2v.mp4")
    assert calls[0]["model_id"] == "bytedance/seedance-2.0/text-to-video"
    assert calls[0]["payload"]["aspect_ratio"] == "16:9"
    assert calls[0]["payload"]["duration"] == 5
    assert calls[0]["payload"]["resolution"] == "480p"
    assert calls[0]["payload"]["generate_audio"] is False
    assert "seed" not in calls[0]["payload"]

    frame = tmp_path / "f.png"; frame.write_bytes(b"\x89PNG\r\n")
    wave.generate("continue", 1.7, tmp_path / "i2v.mp4", first_frame=frame)
    assert calls[1]["model_id"] == "bytedance/seedance-2.0/image-to-video"
    assert "aspect_ratio" not in calls[1]["payload"]
    assert calls[1]["payload"]["duration"] == 4          # snapped up into [4, 15]
    assert calls[1]["payload"]["image"] == "https://fake.host/f.png"


def test_wavespeed_legacy_seedance_v1_payload_matches_univa_reference(
        tmp_path: Path, monkeypatch):
    """A legacy v1 model id keeps UniVA's WORKING schema: seed present,
    duration snapped to {5,10}, -t2v- → -i2v- id swap."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed",
                            "model_id": "bytedance/seedance-v1-pro-t2v-480p"})
    calls = []

    def _fake_run_task(model_id, payload, out_path):
        calls.append({"model_id": model_id, "payload": payload})
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media", lambda p: "https://fake.host/f.png")

    wave.generate("a ball falls", 5.0, tmp_path / "t2v.mp4", seed=7)
    assert calls[0]["payload"]["seed"] == 7
    assert calls[0]["payload"]["aspect_ratio"] == "16:9"
    assert calls[0]["payload"]["duration"] == 5
    assert "resolution" not in calls[0]["payload"]

    frame = tmp_path / "f.png"; frame.write_bytes(b"\x89PNG\r\n")
    wave.generate("continue", 1.7, tmp_path / "i2v.mp4", first_frame=frame)
    assert calls[1]["model_id"] == "bytedance/seedance-v1-pro-i2v-480p"
    assert calls[1]["payload"]["duration"] == 5          # {5,10} enum
    assert calls[1]["payload"]["image"] == "https://fake.host/f.png"


def test_wavespeed_flf2v_routes(tmp_path: Path, monkeypatch):
    """frame_to_frame default = seedance-2.0 i2v with image+last_image (the
    best first+last model); flf2v_model=wan-flf2v keeps the legacy schema."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    first = tmp_path / "a.png"; first.write_bytes(b"\x89PNG")
    last = tmp_path / "b.png"; last.write_bytes(b"\x89PNG")

    def _mk(client, calls):
        def _fake_run_task(model_id, payload, out_path):
            calls.append({"model_id": model_id, "payload": payload})
            out = Path(out_path); out.write_bytes(b"OUT")
            return out
        return _fake_run_task

    wave = build_video_gen({"name": "wavespeed"})
    calls = []
    monkeypatch.setattr(wave, "_run_task", _mk(wave, calls))
    monkeypatch.setattr(wave, "_upload_media", lambda p: f"https://fake.host/{Path(p).name}")
    wave.frame_to_frame("morph", first, last, tmp_path / "o.mp4", duration=2)
    assert calls[0]["model_id"] == "bytedance/seedance-2.0/image-to-video"
    assert calls[0]["payload"]["image"].endswith("a.png")
    assert calls[0]["payload"]["last_image"].endswith("b.png")
    assert calls[0]["payload"]["duration"] == 4          # snapped into [4, 15]

    legacy = build_video_gen({"name": "wavespeed",
                              "flf2v_model": "wavespeed-ai/wan-flf2v"})
    calls2 = []
    monkeypatch.setattr(legacy, "_run_task", _mk(legacy, calls2))
    monkeypatch.setattr(legacy, "_upload_media", lambda p: f"https://fake.host/{Path(p).name}")
    legacy.frame_to_frame("morph", first, last, tmp_path / "o2.mp4", duration=2)
    assert calls2[0]["model_id"] == "wavespeed-ai/wan-flf2v"
    assert calls2[0]["payload"]["first_image"].endswith("a.png")
    assert calls2[0]["payload"]["last_image"].endswith("b.png")
    assert calls2[0]["payload"]["size"] == "832*480"
    assert calls2[0]["payload"]["duration"] == 5         # {5,10} enum


def test_wavespeed_edit_video_seedance_default_route(tmp_path: Path, monkeypatch):
    """edit_video default backend = seedance-2.0/video-edit; the input video is
    passed by UPLOADED URL (gen4-aleph 400s on base64 — the URL rule is global)."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    calls = []

    def _fake_run_task(model_id, payload, out_path):
        calls.append({"model_id": model_id, "payload": payload})
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media", lambda p: "https://fake.host/in.mp4")
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    wave.edit_video("make it rain", vid, tmp_path / "o.mp4")
    assert calls[0]["model_id"] == "bytedance/seedance-2.0/video-edit"
    assert calls[0]["payload"]["video"] == "https://fake.host/in.mp4"
    assert calls[0]["payload"]["generate_audio"] is False


def test_wavespeed_400_error_surfaces_response_body(tmp_path: Path, monkeypatch):
    """REGRESSION: raise_for_status() drops the response body — the part where
    WaveSpeed explains WHICH field is wrong. The error must carry it, plus a
    payload summary with base64 blobs shortened."""
    import sys
    import types

    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})

    class FakeResp:
        status_code = 400
        text = '{"code":400,"message":"duration must be one of [5, 10]"}'

    fake_requests = types.SimpleNamespace(post=lambda *a, **k: FakeResp())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    with pytest.raises(RuntimeError, match=r"duration must be one of"):
        wave._run_task("bytedance/seedance-v1-pro-t2v-480p",
                       {"prompt": "x", "duration": 40, "image": "A" * 5000},
                       tmp_path / "o.mp4")
    # and the payload summary never embeds the full base64 blob
    try:
        wave._run_task("m/x", {"image": "A" * 5000}, tmp_path / "o.mp4")
    except RuntimeError as e:
        assert "A" * 300 not in str(e)
        assert "<5000 chars>" in str(e)


# ── video EXTEND (dedicated seedance-2.0/video-extend endpoint) ──
def test_wavespeed_extend_loud_without_api_key(tmp_path: Path, monkeypatch):
    """extend() must fail LOUDLY (no key) BEFORE any upload or network POST."""
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        wave.extend("continue the shot", vid, tmp_path / "o.mp4")


def test_wavespeed_extend_posts_dedicated_endpoint(tmp_path: Path, monkeypatch):
    """extend() posts the WHOLE input video (uploaded, by URL) to the dedicated
    video-extend model — no more decode-last-frame → i2v hack. Stubbed, NO
    network."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    calls = {}

    def _fake_run_task(model_id, payload, out_path):
        calls["model_id"] = model_id
        calls["payload"] = payload
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media", lambda p: "https://fake.host/in.mp4")
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    out = wave.extend("continue the shot", vid, tmp_path / "o.mp4", duration=5)

    assert out.exists()
    assert calls["model_id"] == "bytedance/seedance-2.0/video-extend"
    assert calls["payload"]["video"] == "https://fake.host/in.mp4"
    assert calls["payload"]["prompt"] == "continue the shot"
    assert calls["payload"]["duration"] == 5

def test_wavespeed_reference_video_channel(tmp_path: Path, monkeypatch):
    """generate(reference_video=...) rides seedance-2.0 reference_videos (by
    uploaded URL); a legacy model id raises honestly (no such channel)."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    calls = []

    def _fake_run_task(model_id, payload, out_path):
        calls.append({"model_id": model_id, "payload": payload})
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media",
                        lambda p: f"https://fake.host/{Path(p).name}")
    ref = tmp_path / "sim.mp4"; ref.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    wave.generate("a ball falls", 5.0, tmp_path / "o.mp4", reference_video=ref)
    assert calls[0]["payload"]["reference_videos"] == ["https://fake.host/sim.mp4"]

    legacy = build_video_gen({"name": "wavespeed",
                              "model_id": "bytedance/seedance-v1-pro-t2v-480p"})
    with pytest.raises(RuntimeError, match="reference_videos"):
        legacy.generate("x", 5.0, tmp_path / "o2.mp4", reference_video=ref)


def test_wavespeed_reference_images_channel(tmp_path: Path, monkeypatch):
    """generate(reference_images=[...]) rides seedance-2.0 reference_images —
    VERIFIED on the T2V endpoint only (≤9 images, @ImageN mentions). On i2v
    (first_frame present) the refs are DROPPED with a loud warning (the i2v
    schema exposes only image+last_image — refs there are UNVERIFIED, never
    hardcoded). Legacy model ids raise honestly."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    calls = []

    def _fake_run_task(model_id, payload, out_path):
        calls.append({"model_id": model_id, "payload": payload})
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media",
                        lambda p: f"https://fake.host/{Path(p).name}")
    prev_last = tmp_path / "prev_last.png"; prev_last.write_bytes(b"\x89PNG")
    kf = tmp_path / "kf.png"; kf.write_bytes(b"\x89PNG")

    # T2V route: both images ride reference_images (the ti2v_prev_plus_keyframe
    # strategy's exact backend shape)
    wave.generate("open on @Image1, move toward @Image2", 5.0,
                  tmp_path / "o.mp4", reference_images=[prev_last, kf])
    p0 = calls[0]["payload"]
    assert calls[0]["model_id"] == "bytedance/seedance-2.0/text-to-video"
    assert p0["reference_images"] == ["https://fake.host/prev_last.png",
                                      "https://fake.host/kf.png"]
    # cap at 9
    many = []
    for i in range(11):
        f = tmp_path / f"r{i}.png"; f.write_bytes(b"\x89PNG")
        many.append(f)
    wave.generate("x", 5.0, tmp_path / "o2.mp4", reference_images=many)
    assert len(calls[1]["payload"]["reference_images"]) == 9
    # i2v route: refs dropped (schema-honest), first frame kept
    wave.generate("continue", 5.0, tmp_path / "o3.mp4",
                  first_frame=prev_last, reference_images=[kf])
    p2 = calls[2]["payload"]
    assert p2["image"].endswith("prev_last.png")
    assert "reference_images" not in p2
    # legacy family has no channel at all
    legacy = build_video_gen({"name": "wavespeed",
                              "model_id": "bytedance/seedance-v1-pro-t2v-480p"})
    with pytest.raises(RuntimeError, match="reference_images"):
        legacy.generate("x", 5.0, tmp_path / "o4.mp4", reference_images=[kf])


def test_wavespeed_multi_image_to_video(tmp_path: Path, monkeypatch):
    """multi_image_to_video default = kling-video-o1/reference-to-video
    (docs: ≤7 images, or ≤4 when a reference video is ALSO passed); the
    legacy kling-v1.6 schema stays behind multi_i2v_model. Empty raises."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    calls = []

    def _fake_run_task(model_id, payload, out_path):
        calls.append({"model_id": model_id, "payload": payload})
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    monkeypatch.setattr(wave, "_upload_media",
                        lambda p: f"https://fake.host/{Path(p).name}")
    imgs = []
    for i in range(9):
        f = tmp_path / f"m{i}.png"; f.write_bytes(b"\x89PNG")
        imgs.append(f)
    # image-only → cap 7
    wave.multi_image_to_video("fuse", imgs, tmp_path / "o.mp4", duration=7)
    assert calls[0]["model_id"] == "kwaivgi/kling-video-o1/reference-to-video"
    assert len(calls[0]["payload"]["images"]) == 7
    assert calls[0]["payload"]["duration"] == 10            # {5,10} snap-up
    assert "video" not in calls[0]["payload"]
    # with a reference video → cap drops to 4 (documented rule)
    vid = tmp_path / "prev_tail.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    wave.multi_image_to_video("fuse", imgs, tmp_path / "o2.mp4", video=vid)
    assert len(calls[1]["payload"]["images"]) == 4
    assert calls[1]["payload"]["video"].endswith("prev_tail.mp4")
    # legacy schema via config
    legacy = build_video_gen({"name": "wavespeed",
                              "multi_i2v_model":
                                  "kwaivgi/kling-v1.6-multi-i2v-standard"})
    monkeypatch.setattr(legacy, "_run_task", _fake_run_task)
    monkeypatch.setattr(legacy, "_upload_media",
                        lambda p: f"https://fake.host/{Path(p).name}")
    legacy.multi_image_to_video("fuse", imgs, tmp_path / "o3.mp4")
    assert calls[2]["model_id"] == "kwaivgi/kling-v1.6-multi-i2v-standard"
    assert len(calls[2]["payload"]["images"]) == 4
    with pytest.raises(ValueError):
        wave.multi_image_to_video("x", [], tmp_path / "o4.mp4")
