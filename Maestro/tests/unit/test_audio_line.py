"""音频线回归(2026-07-29,用户批准的两条极简策略入核心代码):
(1) scene 级 BGM:剧本 music_plan 解析/持久化 → §F add_music 逐 scene
    一曲 + 音乐床偏移;空计划诚实静音。
(2) 对白音画同步:剧本 dialogue 字段 → 出口确定性口型子句(引号去重)
    + 逐镜临时开 generate_audio。全部离线,不碰网络。"""

import json

from maestro.memory.storyboard import ShotEntry, StoryboardMemory
from maestro.pipeline import audio_stage
from maestro.pipeline.window_loop import _with_dialogue, _write_outline


# ── 剧本解析:dialogue + music_plan ──────────────────────────────────

def test_outline_parses_dialogue_and_music_plan():
    class _LLM:
        def complete(self, prompt, **kw):
            return json.dumps({
                "cast": {}, "setting": "",
                "music_plan": {"scene 1": "warm strings, 95bpm",
                               "2": "tense percussion", "scene x": "bad"},
                "shots": [
                    {"description": "Shot 1: the cat looks at the camera "
                                    "in a warm living room, close-up",
                     "duration_s": 5, "end_state": "still",
                     "dialogue": "Time for breakfast!"},
                    {"description": "Shot 2: the cat trots to the bowl "
                                    "across the floor, tracking",
                     "duration_s": 5, "end_state": "trotting"}]})

    shots, durs, ends, meta, via = _write_outline(
        _LLM(), "p", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["fb"])
    assert via == "llm"
    # scene 号归一化("scene 1"/"2" 皆收;无数字的键丢弃)
    assert meta["music_plan"] == {1: "warm strings, 95bpm",
                                  2: "tense percussion"}
    assert meta["dialogues"] == ["Time for breakfast!", ""]


def test_outline_fallback_meta_has_audio_fields():
    class _Bad:
        def complete(self, prompt, **kw):
            return "not json"

    _s, _d, _e, meta, via = _write_outline(
        _Bad(), "p", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["a"])
    assert via == "fallback"
    assert meta["music_plan"] == {} and meta["dialogues"] == [""]


def test_storyboard_persists_music_plan_and_dialogue(tmp_path):
    sb = StoryboardMemory.from_outline(
        ["shot 1: a cat speaks"], path=tmp_path / "sb.json")
    sb.music_plan = {1: "warm strings"}
    sb.entries[0].dialogue = "Hello there!"
    sb.set_condition(0, {"strategy": "t2v"})    # 触发 _save
    back = StoryboardMemory.load(tmp_path / "sb.json")
    assert back.music_plan == {1: "warm strings"}
    assert back.entries[0].dialogue == "Hello there!"
    assert back.entries[0].to_brain_line()["dialogue"] == "Hello there!"


# ── 口型子句:确定性追加 + 去重 ──────────────────────────────────────

def _entry(dialogue="Time for breakfast!"):
    e = ShotEntry(shot_idx=0, scene_idx=1, label="scene 1 shot 1",
                  description="Shot 1: <the cat> looks at the camera")
    e.dialogue = dialogue
    return e


CAST = {"the cat": "static: an orange cat; dynamic: none"}


def test_with_dialogue_appends_line_and_suppression():
    out = _with_dialogue("The cat looks at the camera.", _entry(), CAST)
    assert 'the cat says: "Time for breakfast!"' in out
    assert "no background music" in out
    # 2026-08-04 用户裁决:收势句/mouth-moving 不再机械追加
    assert "micro-expression" not in out and "mouth moving" not in out


def test_with_dialogue_dedupes_and_noops():
    p = 'X says: "Time for breakfast!" already'
    assert _with_dialogue(p, _entry(), CAST) == p          # 引号串已在 → 不重复
    assert _with_dialogue("prompt", _entry(dialogue=""), CAST) == "prompt"
    assert _with_dialogue("", _entry(), CAST) == ""


# ── §F:scene 跨度与 add_music 装配 ─────────────────────────────────

def test_scene_spans_accumulates_by_entry_order(monkeypatch, tmp_path):
    sb = StoryboardMemory.from_outline(
        ["shot 1: scene 1 — a", "shot 2: scene 1 — b",
         "shot 3: scene 2 — c"], path=None)
    for i, e in enumerate(sb.entries):
        e.video_path = str(tmp_path / f"s{i}.mp4")
    monkeypatch.setattr(audio_stage, "probe", lambda p: (5.0, False))
    spans = audio_stage.scene_spans(sb)
    assert spans == [(1, 0.0, 10.0), (2, 10.0, 5.0)]


def test_add_music_empty_plan_is_honest_silence(tmp_path):
    sb = StoryboardMemory.from_outline(["shot 1: a"], path=None)
    assert audio_stage.add_music(tmp_path / "movie.mp4", sb, None,
                                 tmp_path / "scored.mp4") is None


def test_add_music_generates_per_scene_and_mixes(monkeypatch, tmp_path):
    sb = StoryboardMemory.from_outline(
        ["shot 1: scene 1 — a", "shot 2: scene 2 — b"], path=None)
    for i, e in enumerate(sb.entries):
        e.video_path = str(tmp_path / f"s{i}.mp4")
    sb.music_plan = {1: "warm strings", 2: "tense drums"}

    monkeypatch.setattr(audio_stage, "probe", lambda p: (6.0, True))
    monkeypatch.setattr(audio_stage.shutil, "which", lambda x: "/bin/ffmpeg")
    calls = []

    def _music(desc, dur, out):
        calls.append((desc, dur))
        out.write_bytes(b"mp3")
        return out

    mixed = []
    monkeypatch.setattr(audio_stage, "_build_bed",
                        lambda tracks, total, out: mixed.append(
                            ("bed", [(str(t[0]), t[1], t[2])
                                     for t in tracks], total)))
    monkeypatch.setattr(audio_stage, "_mix_onto",
                        lambda v, b, ha, t, od, op: op)
    out = audio_stage.add_music(tmp_path / "movie.mp4", sb, None,
                                tmp_path / "scored.mp4", music_fn=_music)
    assert out == tmp_path / "scored.mp4"
    assert calls == [("warm strings", 6.0), ("tense drums", 6.0)]
    # 音乐床:scene 2 的曲目从 6s 处开始铺
    (_tag, tracks, total) = mixed[0]
    assert tracks[0][1] == 0.0 and tracks[1][1] == 6.0 and total == 6.0


def test_add_music_scene_without_desc_left_silent(monkeypatch, tmp_path):
    sb = StoryboardMemory.from_outline(
        ["shot 1: scene 1 — a", "shot 2: scene 2 — b"], path=None)
    for i, e in enumerate(sb.entries):
        e.video_path = str(tmp_path / f"s{i}.mp4")
    sb.music_plan = {2: "tense drums"}          # scene 1 刻意静场
    monkeypatch.setattr(audio_stage, "probe", lambda p: (6.0, False))
    monkeypatch.setattr(audio_stage.shutil, "which", lambda x: "/bin/ffmpeg")
    calls = []

    def _music(desc, dur, out):
        calls.append(desc)
        out.write_bytes(b"mp3")
        return out

    monkeypatch.setattr(audio_stage, "_build_bed", lambda *a: None)
    monkeypatch.setattr(audio_stage, "_mix_onto",
                        lambda v, b, ha, t, od, op: op)
    audio_stage.add_music(tmp_path / "m.mp4", sb, None,
                          tmp_path / "s.mp4", music_fn=_music)
    assert calls == ["tense drums"]


# ── 2026-07-30:评审新增检查 —— 运镜衔接 + 音画同步 ──────────────────

def _review_text(tmp_path, monkeypatch, conditioning):
    """跑一次 review_shot(打桩 _generate),返回评审指令全文。"""
    import json as _json

    from maestro.models.mllm_backends import GeminiVLM
    from maestro.types import CandidateClip, ShotSpec

    vlm = GeminiVLM("gemini", {"api_key": "k"})
    captured = []

    def _fake(parts, **kw):
        captured.append(parts)
        return _json.dumps({"checks": [], "issues": [], "summary": "ok"})
    monkeypatch.setattr(vlm, "_generate", _fake)
    v = tmp_path / "s.mp4"
    v.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    clip = CandidateClip(shot_idx=0, video_path=v)
    clip.conditioning = {"video_prompt": "p", "images": [], **conditioning}
    vlm.review_shot(clip, ShotSpec(shot_idx=0, duration=5.0, prompt="p"))
    return " ".join(x.get("text", "") for x in captured[0] if "text" in x)


def test_review_adds_camera_continuity_check(tmp_path, monkeypatch):
    text = _review_text(tmp_path, monkeypatch, {
        "junction_prev_actual": "the cat trots right, camera tracking "
                                "alongside at walking pace"})
    assert "CAMERA CONTINUITY" in text
    assert "REVERSED camera direction" in text
    # 实况没提镜头 → 不注入(不查无据的东西)
    text2 = _review_text(tmp_path, monkeypatch, {
        "junction_prev_actual": "the cat sits still at the bowl"})
    assert "CAMERA CONTINUITY" not in text2


def test_review_adds_dialogue_sync_checks(tmp_path, monkeypatch):
    text = _review_text(tmp_path, monkeypatch, {"dialogue": "快画完了"})
    assert 'DIALOGUE: "快画完了"' in text
    assert "synchronized" in text          # 口型同步
    assert "no background music" in text   # 人声之外须干净
    # 无台词 → 不注入
    text2 = _review_text(tmp_path, monkeypatch, {})
    assert "DIALOGUE" not in text2
