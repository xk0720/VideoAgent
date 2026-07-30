"""§E 终版清单与接缝分诊回归(2026-07-30 用户规矩):
① 终版路径以台账为唯一权威,缺失响亮跳过并留痕;
② extend 镜生成时已裁头,此处不动;首帧锁上一镜尾帧的策略
   (_PREV_FRAME_LOCKED)只有实测确为重复帧(MAD<阈值)才裁第 0 帧,
   量不出/不像 → 不裁。全部离线(测量与裁剪打桩)。"""

from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.memory.storyboard import StoryboardMemory


def _board(tmp_path, strategies, missing=()):
    sb = StoryboardMemory.from_outline(
        [f"shot {i + 1}: scene 1 — beat {i}" for i in range(len(strategies))],
        path=None)
    for i, (e, strat) in enumerate(zip(sb.entries, strategies)):
        p = tmp_path / f"s{i}.mp4"
        if i not in missing:
            p.write_bytes(b"\x00" * 2048)
        e.video_path = str(p)
        e.condition = {"strategy": strat}
    return sb


def test_manifest_skips_missing_with_note(tmp_path):
    sb = _board(tmp_path, ["i2v_keyframe", "extend_prev"], missing=(1,))
    clips, notes = wl._final_cut(sb, tmp_path)
    assert len(clips) == 1 and clips[0] == Path(sb.entries[0].video_path)
    assert notes == [{"stage": "assemble", "label": "scene 1 shot 2",
                      "action": "skip_missing",
                      "path": sb.entries[1].video_path}]


def test_dedup_only_for_prev_locked_and_duplicate(tmp_path, monkeypatch):
    sb = _board(tmp_path, ["i2v_keyframe", "ti2v_prev_plus_keyframe",
                           "extend_prev", "ti2v_prev_last"])
    mads = {"s1.mp4": 5.0, "s3.mp4": 20.0}      # 镜2重复;镜4不像
    monkeypatch.setattr(
        wl, "_first_last_mad",
        lambda prev, cur, wd: mads.get(Path(cur).name))
    trimmed = []

    def _fake_trim(video, seconds, out):
        trimmed.append((Path(video).name, round(seconds, 4)))
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 2048)
        return out
    monkeypatch.setattr(wl, "_trim_head", _fake_trim)
    import maestro.pipeline.timeline as tl
    monkeypatch.setattr(tl, "_probe_fps", lambda p: 24.0)

    clips, notes = wl._final_cut(sb, tmp_path)
    # 只有镜2(prev-locked 且 MAD<8)被裁一帧(1/24s);extend/i2v/不像的都不动
    assert trimmed == [("s1.mp4", round(1 / 24, 4))]
    assert clips[0].name == "s0.mp4"
    assert clips[1].name == "s1_dedup.mp4"
    assert clips[2].name == "s2.mp4" and clips[3].name == "s3.mp4"
    acts = {n["label"]: n["action"] for n in notes}
    assert acts == {"scene 1 shot 2": "dedup_first_frame",
                    "scene 1 shot 4": "dedup_not_needed"}


def test_first_shot_never_deduped(tmp_path, monkeypatch):
    sb = _board(tmp_path, ["ti2v_prev_last"])          # 首镜即便策略同名
    called = []
    monkeypatch.setattr(wl, "_first_last_mad",
                        lambda *a: called.append(1) or 0.0)
    clips, notes = wl._final_cut(sb, tmp_path)
    assert not called and notes == []
    assert clips[0].name == "s0.mp4"


def test_unmeasurable_mad_keeps_original(tmp_path, monkeypatch):
    sb = _board(tmp_path, ["i2v_keyframe", "flf2v_bridge"])
    monkeypatch.setattr(wl, "_first_last_mad", lambda *a: None)
    trimmed = []
    monkeypatch.setattr(wl, "_trim_head",
                        lambda *a: trimmed.append(1))
    clips, notes = wl._final_cut(sb, tmp_path)
    assert not trimmed and notes == []                 # 量不出 → 不裁不记
    assert [c.name for c in clips] == ["s0.mp4", "s1.mp4"]
