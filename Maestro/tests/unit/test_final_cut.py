"""§E 终版清单 + 生成时接缝去重回归(2026-07-30 用户规矩,二次简化:
切割全部前移到生成时 —— extend 裁头(既有)+ 首帧去重(_drop_first_
frame:硬锁无条件切、软锁先量后切);拼装层只留"终版路径确定并核验"
的清单。全部离线。"""

from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.memory.storyboard import StoryboardMemory


# ── 拼装清单:台账权威 + 缺失响亮跳过 ────────────────────────────────

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


def test_manifest_passes_files_through_untouched(tmp_path):
    # 切割已前移到生成时 —— 拼装清单绝不再动文件
    sb = _board(tmp_path, ["ti2v_prev_last", "ti2v_prev_plus_keyframe"])
    clips, notes = wl._final_cut(sb, tmp_path)
    assert [c.name for c in clips] == ["s0.mp4", "s1.mp4"]
    assert notes == []


# ── 生成时去重:_drop_first_frame ────────────────────────────────────

def _stub_trim(monkeypatch, record, succeed=True):
    def _fake(video, seconds, out):
        record.append((Path(video).name, round(seconds, 4)))
        if not succeed:
            return None
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 2048)
        return out
    monkeypatch.setattr(wl, "_trim_head", _fake)
    import maestro.pipeline.timeline as tl
    monkeypatch.setattr(tl, "_probe_fps", lambda p: 24.0)


def test_hard_lock_cuts_unconditionally(tmp_path, monkeypatch):
    # 硬锁(ti2v_prev_last / flf2v_bridge):不量,直接切一帧(API 保证重复)
    calls = []
    _stub_trim(monkeypatch, calls)
    monkeypatch.setattr(wl, "_first_last_mad",
                        lambda *a: (_ for _ in ()).throw(AssertionError(
                            "hard lock must not measure")))
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * 2048)
    cond = {}
    out = wl._drop_first_frame(v, cond)
    assert calls == [("clip.mp4", round(1 / 24, 4))]
    assert out.name == "clip_dedup.mp4"
    assert cond["dedup_first_frame"] is True


def test_soft_lock_measures_then_cuts_or_keeps(tmp_path, monkeypatch):
    calls = []
    _stub_trim(monkeypatch, calls)
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * 2048)
    prev = tmp_path / "prev.mp4"

    # 服从了锁(MAD 5 < 8)→ 切
    monkeypatch.setattr(wl, "_first_last_mad", lambda *a: 5.0)
    cond = {}
    out = wl._drop_first_frame(v, cond, measured_prev=prev)
    assert out.name == "clip_dedup.mp4"
    assert cond == {"junction_mad": 5.0, "dedup_first_frame": True}

    # 没服从(MAD 20)→ 不切,首帧是真内容,证据留给评审
    monkeypatch.setattr(wl, "_first_last_mad", lambda *a: 20.0)
    cond = {}
    out = wl._drop_first_frame(v, cond, measured_prev=prev)
    assert out == v
    assert cond == {"junction_mad": 20.0, "dedup_first_frame": False}

    # 量不出 → 不切(不猜)
    monkeypatch.setattr(wl, "_first_last_mad", lambda *a: None)
    cond = {}
    out = wl._drop_first_frame(v, cond, measured_prev=prev)
    assert out == v
    assert cond == {"junction_mad": None, "dedup_first_frame": False}


def test_trim_failure_keeps_original_honestly(tmp_path, monkeypatch):
    calls = []
    _stub_trim(monkeypatch, calls, succeed=False)
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * 2048)
    cond = {}
    out = wl._drop_first_frame(v, cond)
    assert out == v and cond["dedup_first_frame"] is False
