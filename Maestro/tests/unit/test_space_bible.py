"""空间圣经(2026-08-10 用户令):多视图注册表/挑图/语义行/清场回流。"""
import json
from pathlib import Path
from types import SimpleNamespace

from maestro.memory.storyboard import StoryboardMemory
from maestro.pipeline.space_bible import (build_space_views,
                                          pick_space_view,
                                          space_semantic_line,
                                          washed_frame_upgrade)


def _sb(tmp_path):
    sb = StoryboardMemory.from_outline(["shot 1: x"],
                                       path=tmp_path / "sb.json")
    master = tmp_path / "bg_1.png"
    master.write_bytes(b"x")
    sb.backgrounds["bg_1"] = {"path": str(master), "src": "t2i"}
    return sb


class _Edit:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def edit(self, keyframe, instruction, out_path, references=None):
        self.calls.append(instruction)
        if self.fail:
            raise RuntimeError("down")
        Path(out_path).write_bytes(b"v")
        return out_path


class _MLLM:
    def __init__(self, caps):
        self.caps = list(caps)

    def caption_image(self, p):
        return self.caps.pop(0) if self.caps else "a view"


def test_build_views_derives_three_and_captions(tmp_path, monkeypatch):
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    ed = _Edit()
    build_space_views(sb, ed, _MLLM(["主视图", "左", "右", "反打"]),
                      tmp_path / "spaces", {"bg_1": "a rooftop"})
    views = sb.spaces["bg_1"]
    assert set(views) == {"master", "left", "right", "reverse"}
    assert views["reverse"]["src"] == "derived"
    assert "OPPOSITE" in ed.calls[-1]        # 反打指令
    assert "a rooftop" in ed.calls[0]        # 场景语境进指令
    # 持久化断言(2026-08-10 用户令:全链落台账)
    back = StoryboardMemory.load(tmp_path / "sb.json")
    assert set(back.spaces["bg_1"]) == set(views)


def test_build_views_edit_failure_leaves_master(tmp_path, monkeypatch):
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    build_space_views(sb, _Edit(fail=True), _MLLM(["m"]),
                      tmp_path / "spaces", None)
    assert list(sb.spaces["bg_1"]) == ["master"]   # 缺席留痕不断链


def test_pick_view_by_caption_and_fallback(tmp_path):
    sb = _sb(tmp_path)
    sb.spaces["bg_1"] = {
        "master": {"path": "/m.png", "caption": "小圆桌与天际线"},
        "reverse": {"path": "/r.png", "caption": "红砖墙与灰色铁门"}}

    class _LLM:
        def complete(self, prompt, **kw):
            return '{"view": "reverse"}'
    got = pick_space_view(_LLM(), sb, "bg_1", "背影走向天台门")
    assert got["view"] == "reverse"

    class _Bad:
        def complete(self, prompt, **kw):
            return "nope"
    got2 = pick_space_view(_Bad(), sb, "bg_1", "x")
    assert got2["view"] == "master"          # 坏输出退 master
    assert pick_space_view(None, sb, "bg_2", "x") is None


def test_semantic_line_carries_caption():
    line = space_semantic_line({"caption": "红砖墙,灰色铁门"}, zh=True)
    assert "红砖墙" in line and "位置与外观" in line
    en = space_semantic_line({"caption": "brick wall"}, zh=False)
    assert "brick wall" in en and "framing is free" in en


def test_washed_upgrade_confident_replaces(tmp_path, monkeypatch):
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    sb.spaces["bg_1"] = {"master": {"path": "/m.png",
                                    "caption": "桌与天际线",
                                    "src": "t2i", "shot_idx": None}}
    tail = tmp_path / "tail.png"
    tail.write_bytes(b"t")

    class _LLM:
        def complete(self, prompt, **kw):
            return '{"view": "master", "confident": true}'
    v = washed_frame_upgrade(sb, "bg_1", tail, _Edit(),
                             _MLLM(["empty rooftop, table, skyline"]),
                             _LLM(), tmp_path / "spaces", 3)
    assert v == "master"
    assert sb.spaces["bg_1"]["master"]["src"] == "frame"
    assert sb.spaces["bg_1"]["master"]["shot_idx"] == 3


def test_washed_upgrade_person_detected_skips(tmp_path, monkeypatch):
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    sb.spaces["bg_1"] = {"master": {"path": "/m.png", "caption": "x",
                                    "src": "t2i", "shot_idx": None}}
    tail = tmp_path / "tail.png"
    tail.write_bytes(b"t")
    v = washed_frame_upgrade(sb, "bg_1", tail, _Edit(),
                             _MLLM(["a man stands on the rooftop"]),
                             None, tmp_path / "spaces", 3)
    assert v is None                          # 清场失败 → 放弃顶替
    assert sb.spaces["bg_1"]["master"]["src"] == "t2i"


def test_washed_upgrade_unsure_appends(tmp_path, monkeypatch):
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    sb.spaces["bg_1"] = {"master": {"path": "/m.png", "caption": "x",
                                    "src": "t2i", "shot_idx": None}}
    tail = tmp_path / "tail.png"
    tail.write_bytes(b"t")

    class _Unsure:
        def complete(self, prompt, **kw):
            return '{"view": null, "confident": false}'
    v = washed_frame_upgrade(sb, "bg_1", tail, _Edit(),
                             _MLLM(["empty alley"]), _Unsure(),
                             tmp_path / "spaces", 5)
    assert v == "new_0"                       # 拿不准 → 追加不顶替
    assert sb.spaces["bg_1"]["new_0"]["src"] == "frame"
