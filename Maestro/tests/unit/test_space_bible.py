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


def test_washed_upgrade_appends_only(tmp_path, monkeypatch):
    """2026-08-10 用户裁决:回流【纯追加】不顶替 —— master 恒不被改。"""
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    sb.spaces["bg_1"] = {"master": {"path": "/m.png",
                                    "caption": "桌与天际线",
                                    "src": "t2i", "shot_idx": None}}
    tail = tmp_path / "tail.png"
    tail.write_bytes(b"t")
    v = washed_frame_upgrade(sb, "bg_1", tail, _Edit(),
                             _MLLM(["empty rooftop, table, skyline"]),
                             None, tmp_path / "spaces", 3)
    assert v == "new_0"
    assert sb.spaces["bg_1"]["master"]["src"] == "t2i"   # 恒不被改
    assert sb.spaces["bg_1"]["new_0"]["src"] == "frame"


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

    v = washed_frame_upgrade(sb, "bg_1", tail, _Edit(),
                             _MLLM(["empty alley"]), None,
                             tmp_path / "spaces", 5)
    assert v == "new_0"                       # 拿不准 → 追加不顶替
    assert sb.spaces["bg_1"]["new_0"]["src"] == "frame"


def test_build_views_pan_video_frames(tmp_path, monkeypatch):
    """2026-08-10 二版(用户裁决):环视视频抽帧 —— 首帧硬钉主板,
    1/4、1/2、3/4 抽帧为 pan_90/180/270;视频端失败退图像编辑。"""
    import subprocess
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)

    class _VG:
        generate_audio = False

        def __init__(self):
            self.calls = []

        def generate(self, prompt, duration, out_path, fps=24, seed=0,
                     first_frame=None, **kw):
            self.calls.append({"prompt": prompt,
                               "first": str(first_frame)})
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                 "-i", "color=c=gray:s=160x90:d=2", str(out_path)],
                check=True)
            return out_path

    vg = _VG()
    build_space_views(sb, None, _MLLM(["m", "a", "b", "c"]),
                      tmp_path / "spaces", {"bg_1": "a rooftop"},
                      video_gen=vg)
    views = sb.spaces["bg_1"]
    assert {"pan_90", "pan_180", "pan_270"} <= set(views)
    assert views["pan_180"]["src"] == "derived"
    assert vg.calls[0]["first"].endswith("bg_1.png")     # 首帧硬钉主板
    assert "360 degrees" in vg.calls[0]["prompt"]
    assert "no people" in vg.calls[0]["prompt"]


def test_build_views_pan_failure_falls_back_to_edit(tmp_path, monkeypatch):
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)

    class _DeadVG:
        generate_audio = False

        def generate(self, *a, **k):
            raise RuntimeError("down")

    ed = _Edit()
    build_space_views(sb, ed, _MLLM(["m", "l", "r", "v"]),
                      tmp_path / "spaces", {"bg_1": "x"},
                      video_gen=_DeadVG())
    views = sb.spaces["bg_1"]
    assert {"left", "right", "reverse"} <= set(views)     # 退老法


def test_pan_frames_deduped_by_mad(tmp_path, monkeypatch):
    """2026-08-10 用户裁决:环视是采样器不是量角器 —— 没转动的帧
    (与 master/已收帧近似)丢弃留痕,宁缺勿滥。"""
    import subprocess
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    # master 用真图(纯灰),环视视频全程同色 → 三帧全部近似 master
    master = tmp_path / "bg_1.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.1", "-frames:v", "1",
         str(master)], check=True)
    sb.backgrounds["bg_1"] = {"path": str(master), "src": "t2i"}

    class _VG:
        generate_audio = False

        def generate(self, prompt, duration, out_path, fps=24, seed=0,
                     first_frame=None, **kw):
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                 "-i", "color=c=gray:s=160x90:d=2", str(out_path)],
                check=True)
            return out_path

    build_space_views(sb, None, _MLLM(["m"]), tmp_path / "spaces",
                      {"bg_1": "x"}, video_gen=_VG())
    views = sb.spaces["bg_1"]
    assert not any(v.startswith("pan_") for v in views)   # 全部判重丢弃


def test_washed_frame_dup_not_appended(tmp_path, monkeypatch):
    import subprocess
    import maestro.cinegraph.first_frame_factory as fff
    monkeypatch.setattr(fff, "_SPACED_WAITS_S", (0,))
    sb = _sb(tmp_path)
    gray = tmp_path / "gray.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.1", "-frames:v", "1",
         str(gray)], check=True)
    sb.spaces["bg_1"] = {"master": {"path": str(gray), "caption": "x",
                                    "src": "t2i", "shot_idx": None}}

    class _GrayEdit:
        def edit(self, keyframe, instruction, out_path, references=None):
            import shutil as _sh
            _sh.copy(gray, out_path)
            return out_path

    tail = tmp_path / "tail.png"
    tail.write_bytes(b"t")
    v = washed_frame_upgrade(sb, "bg_1", gray, _GrayEdit(),
                             _MLLM(["empty room"]), None,
                             tmp_path / "spaces", 7)
    assert v is None                          # 与 master 重复 → 不追加
    assert "new_0" not in sb.spaces["bg_1"]


def test_camera_facing_parsed_and_prompt_clean():
    """camera_facing 字段(2026-08-10 用户设计):分镜解析入台账;
    description 保持纯戏剧内容 —— 字段永不进 prompt 由结构保证
    (它只流向 pick_space_view)。"""
    import json as _json
    from maestro.pipeline.window_loop import _write_outline

    good = {"cast": {"魔术师": "static: x; dynamic: y"},
            "setting": "rooftop at sunset",
            "shots": [{"description": "Shot 1: <魔术师>走向天台门。",
                       "duration_s": 5, "end_state": "静止。",
                       "variation": "small", "camera": 0,
                       "camera_facing": "反打朝天台门与红砖墙,中景",
                       "bg": "bg_1"}],
            "music_plan": {}}

    class _LLM:
        def complete(self, prompt, **kw):
            return _json.dumps(good, ensure_ascii=False)
    shots, durs, ends, meta, via = _write_outline(
        _LLM(), "天台上魔术师走向门。", [], episode_guidance={},
        max_shots=6, fallback_fn=lambda: ["x"], cast_canon={},
        prompt_language="zh")
    assert meta["camera_facings"] == ["反打朝天台门与红砖墙,中景"]


def test_skill_carries_camera_facing_field():
    from pathlib import Path as _P
    sw = _P("src/maestro/skills/brain_skills/scene_write/SKILL.md"
            ).read_text()
    assert "camera_facing" in sw
    assert "NEVER enters any" in sw.replace("\n   ", " ")
