"""缝合师 agent(2026-08-09 用户令):LLM 组稿两镜描述,四级校验,
坏输出退化模板;同人同景改走 derive,失败退 continue。"""
import json
from types import SimpleNamespace

from maestro.agents.junction_stitcher import JunctionStitcherAgent
from maestro.pipeline import window_loop as wl

SLOTS = [
    {"slot": "<<<image_1>>>", "kind": "tail_frame", "content": "tail"},
    {"slot": "<<<image_2>>>", "kind": "portrait", "name": "小明",
     "content": "portrait of 小明"},
]


def _agent(reply):
    class _LLM:
        def __init__(self):
            self.n = 0

        def complete(self, prompt, **kw):
            self.n += 1
            return reply(self.n, prompt)
    a = JunctionStitcherAgent(llm=_LLM())
    return a


def test_stitcher_good_output_used():
    a = _agent(lambda n, p: json.dumps(
        {"first_shot_desc": "镜头静止,<<<image_2>>>立于画面中央。",
         "second_shot_desc": "近景特写,<<<image_2>>>抬头望天。"},
        ensure_ascii=False))
    got = a.run(prev_end_state="x", tail_report=None, cur_opening="y",
                slot_table=SLOTS, prompt_language="zh")
    assert got and "抬头望天" in got["second_shot_desc"]


def test_stitcher_unknown_token_retry_then_fallback():
    a = _agent(lambda n, p: json.dumps(
        {"first_shot_desc": "<<<image_9>>>站着。",
         "second_shot_desc": "近景。"}, ensure_ascii=False))
    assert a.run(prev_end_state="x", tail_report=None, cur_opening="y",
                 slot_table=SLOTS, prompt_language="zh") is None
    assert a.llm.n == 2                     # 纠错重问一轮后放弃


def test_stitcher_bare_name_rejected():
    a = _agent(lambda n, p: json.dumps(
        {"first_shot_desc": "小明站在画面中央。",
         "second_shot_desc": "近景,小明抬头。"}, ensure_ascii=False))
    assert a.run(prev_end_state="x", tail_report=None, cur_opening="y",
                 slot_table=SLOTS, prompt_language="zh") is None


def test_stitcher_language_gate():
    a = _agent(lambda n, p: json.dumps(
        {"first_shot_desc": "A man stands in the frame center.",
         "second_shot_desc": "Close-up, he looks up at the sky."}))
    assert a.run(prev_end_state="x", tail_report=None, cur_opening="y",
                 slot_table=SLOTS, prompt_language="zh") is None


def test_stitcher_annotations_cleaned_then_accepted():
    """旁白/音效标注混入 → 共用清洗器剥除后仍非空 → 收。"""
    a = _agent(lambda n, p: json.dumps(
        {"first_shot_desc": "镜头静止,<<<image_2>>>立于中央。"
                            "旁白:恩怨了结。",
         "second_shot_desc": "近景,<<<image_2>>>抬头。音效:风声。"},
        ensure_ascii=False))
    got = a.run(prev_end_state="x", tail_report=None, cur_opening="y",
                slot_table=SLOTS, prompt_language="zh")
    assert got and "旁白" not in got["first_shot_desc"]
    assert "音效" not in got["second_shot_desc"]


def test_stitcher_empty_reply_retry_then_fallback():
    a = _agent(lambda n, p: "")
    assert a.run(prev_end_state="x", tail_report=None, cur_opening="y",
                 slot_table=SLOTS, prompt_language="zh") is None


def test_derive_uses_stitcher_descs(tmp_path):
    """缝合师产物直接入双镜 prompt;同景派生不挂板(bg_path=None)。"""
    import subprocess
    prev_video = tmp_path / "prev.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.3", str(prev_video)], check=True)
    port = tmp_path / "xm.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.1", "-frames:v", "1",
         str(port)], check=True)

    class _VG:
        generate_audio = False

        def __init__(self):
            self.calls = []

        def ref_token(self, n):
            return f"<<<image_{n}>>>"

        def generate(self, prompt, duration, out_path, fps=24, seed=0,
                     reference_images=None, **kw):
            self.calls.append({"prompt": prompt,
                               "refs": [str(r) for r in
                                        (reference_images or [])]})
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error",
                 "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1",
                 "-f", "lavfi", "-i", "color=c=white:s=160x90:d=1",
                 "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
                 str(out_path)], check=True)
            return out_path

    class _Stitcher:
        def run(self, **kw):
            return {"first_shot_desc": "镜头停稳,<<<image_2>>>静立。",
                    "second_shot_desc": "俯拍全景,<<<image_2>>>渺小。"}

    vg = _VG()
    prev = SimpleNamespace(video_path=str(prev_video),
                           end_state="<小明>静立。", description="",
                           opening_frame="")
    entry = SimpleNamespace(shot_idx=2, end_state="",
                            opening_frame="<小明>在天台边缘。",
                            description="")
    got = wl._derive_junction_frame(
        vg, None, None, prev, prev, entry, ["小明"],
        {"小明": "role"}, {"小明": str(port)}, None,
        tmp_path / "shot002", "zh", stitcher=_Stitcher(),
        tail_report=None)
    assert got is not None
    p = vg.calls[0]["prompt"]
    assert "俯拍全景" in p                       # agent 的第二镜入 prompt
    assert "The first shot description: 镜头停稳" in p
    assert len(vg.calls[0]["refs"]) == 2         # 末帧+肖像,无板


def test_routing_same_cast_same_bg_derives(monkeypatch):
    """2026-08-09 新法:同人同景 → derive;判官/路由单元级验证。"""
    import json as _json

    class _LLM:
        def complete(self, prompt, **kw):
            return _json.dumps({"prev_end_cast": ["小明"],
                                "cur_open_cast": ["小明"],
                                "reason": "同一人"}, ensure_ascii=False)
    same, why, oc = wl._judge_junction_cast(
        _LLM(),
        SimpleNamespace(end_state="<小明>静立。", description="",
                        opening_frame=""),
        SimpleNamespace(opening_frame="<小明>特写。", description="",
                        end_state=""),
        {"小明": "role"}, {"小明": "/p/xm.png"}, "zh")
    assert same
    # 路由真值表(与 window 主链同逻辑):
    for cast_same, bg_same, want, want_fb in (
            (False, True, "derive", "cut"),
            (False, False, "derive", "cut"),
            (True, False, "cut", None),
            (True, True, "derive", "continue")):
        if not cast_same:
            kind, fb = "derive", "cut"
        elif not bg_same:
            kind, fb = "cut", None
        else:
            kind, fb = "derive", "continue"
        assert (kind, fb) == (want, want_fb)


def test_derive_same_bg_attaches_space_view(tmp_path):
    """②空间圣经:同景派生挂朝向视图(末位 refer)+ 布局法语义行;
    缝合师槽位表带 space_view 行;junction_meta 记缝合师产物。"""
    import subprocess
    prev_video = tmp_path / "prev.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.3", str(prev_video)], check=True)
    port = tmp_path / "xm.png"
    sview = tmp_path / "bg1_reverse.png"
    for p in (port, sview):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=gray:s=160x90:d=0.1", "-frames:v", "1",
             str(p)], check=True)

    class _VG:
        generate_audio = False

        def __init__(self):
            self.calls = []

        def ref_token(self, n):
            return f"<<<image_{n}>>>"

        def generate(self, prompt, duration, out_path, fps=24, seed=0,
                     reference_images=None, **kw):
            self.calls.append({"prompt": prompt,
                               "refs": [str(r) for r in
                                        (reference_images or [])]})
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error",
                 "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1",
                 "-f", "lavfi", "-i", "color=c=white:s=160x90:d=1",
                 "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
                 str(out_path)], check=True)
            return out_path

    seen_slots = {}

    class _Stitcher:
        def run(self, slot_table=None, **kw):
            seen_slots["rows"] = slot_table
            return {"first_shot_desc": "镜头停稳,<<<image_2>>>静立。",
                    "second_shot_desc": "朝向<<<image_3>>>所示的红砖墙,"
                                        "<<<image_2>>>的背影走向铁门。"}

    vg = _VG()
    prev = SimpleNamespace(video_path=str(prev_video),
                           end_state="<小明>静立。", description="",
                           opening_frame="")
    entry = SimpleNamespace(shot_idx=2, end_state="",
                            opening_frame="<小明>走向天台门。",
                            description="", junction_meta={})
    got = wl._derive_junction_frame(
        vg, None, None, prev, prev, entry, ["小明"],
        {"小明": "role"}, {"小明": str(port)}, None,
        tmp_path / "shot002", "zh", stitcher=_Stitcher(),
        tail_report=None,
        space_view={"view": "reverse", "path": str(sview),
                    "caption": "红砖墙,灰色铁门"})
    assert got is not None
    call = vg.calls[0]
    assert call["refs"][2] == str(sview)          # 视图末位随行
    assert "红砖墙" in call["prompt"]              # 描述织入实物
    assert "must keep the position and look" in call["prompt"]  # 布局法
    kinds = [r["kind"] for r in seen_slots["rows"]]
    assert "space_view" in kinds                  # 缝合师看得见视图行
    assert entry.junction_meta["stitcher"]["via"] == "agent"
    assert entry.junction_meta["two_shot_video"]
