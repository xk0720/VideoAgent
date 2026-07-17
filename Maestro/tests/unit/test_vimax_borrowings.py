"""ViMax 借鉴回归(2026-07-17,用户令"都做吧"):
P1-① <角括号> 出场标记 —— 机器解析本镜出场角色 + 出口一律剥标记;
P2-⑥ variation 首尾帧变化幅度(策略提示,词表校验);
P2-⑦ opening_frame 开场静态快照(首帧 t2i 底稿,确定性兜底也用)。
"""

import json

from maestro.memory.storyboard import ShotEntry
from maestro.pipeline.window_loop import (
    _cast_in_shot,
    _execute_image_plan,
    _make_keyframe,
    _strip_markers,
    _write_outline,
)

CAST = {"the boy": "static: a boy of eight, red striped t-shirt; dynamic: none",
        "the cat": "static: an orange tabby; dynamic: none"}


def _entry(description="the cat jumps", **kw):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description=description)
    for k, v in kw.items():
        setattr(e, k, v)
    return e


# ── P1-①:标记解析与剥除 ─────────────────────────────────────────────

def test_strip_markers_keeps_words_drops_brackets():
    assert _strip_markers("<the boy> kneels by <the cat>") == \
        "the boy kneels by the cat"
    # 无标记原样;空/None 安全
    assert _strip_markers("no markers here") == "no markers here"
    assert _strip_markers("") == ""
    assert _strip_markers(None) == ""


def test_cast_in_shot_filters_by_markers():
    got = _cast_in_shot("Shot 3: <the boy> kneels by the shards", CAST)
    assert set(got) == {"the boy"}
    # 大小写不敏感
    got = _cast_in_shot("<The Boy> stands up", CAST)
    assert set(got) == {"the boy"}


def test_cast_in_shot_honest_degradation_to_full_cast():
    # 无任何标记(旧剧本/兜底层)→ 全量注入,宁多勿丢
    assert set(_cast_in_shot("the boy kneels", CAST)) == set(CAST)
    # 标记全对不上 cast 键 → 同样全量(标记白写,但契约不丢)
    assert set(_cast_in_shot("<a stranger> walks by", CAST)) == set(CAST)
    # cast 为空 → 空(没有契约可注入)
    assert _cast_in_shot("<the boy> kneels", {}) == {}


# ── P2-⑥/⑦:outline 解析 variation + opening_frame ───────────────────

def test_outline_parses_variation_and_opening_frame():
    class _LLM:
        def complete(self, prompt, **kw):
            return json.dumps({
                "cast": {}, "setting": "",
                "shots": [
                    {"description": "Shot 1: a glass teeters on the table "
                                    "edge in warm light, close-up",
                     "duration_s": 5, "end_state": "rocking",
                     "variation": "Small",
                     "opening_frame": "a glass stands at the table edge"},
                    {"description": "Shot 2: the glass falls and shatters "
                                    "on the tile floor, low angle",
                     "duration_s": 4, "end_state": "shards at rest",
                     "variation": "huge"}]})  # 非法词表 → 空

    shots, durs, ends, meta, via = _write_outline(
        _LLM(), "p", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["fb"])
    assert via == "llm"
    assert meta["variations"] == ["small", ""]
    assert meta["opening_frames"] == ["a glass stands at the table edge", ""]


def test_outline_fallback_meta_lists_align():
    class _Bad:
        def complete(self, prompt, **kw):
            return "not json"

    shots, durs, ends, meta, via = _write_outline(
        _Bad(), "p", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["a", "b"])
    assert via == "fallback"
    assert meta["variations"] == ["", ""]
    assert meta["opening_frames"] == ["", ""]


def test_shot_entry_brain_line_carries_new_fields():
    e = _entry(variation="small", opening_frame="a static snapshot")
    line = e.to_brain_line()
    assert line["variation"] == "small"
    assert line["opening_frame"] == "a static snapshot"


# ── 出口剥标记:t2i prompt / 检索词永远不带角括号 ─────────────────────

class _T2IGen:
    def __init__(self):
        self.prompts = []

    def capabilities(self):
        return {"t2i"}

    def text_to_image(self, prompt, out, seed=0):
        self.prompts.append(prompt)
        out.write_bytes(b"\x89PNG\r\n")
        return out


def test_make_keyframe_strips_markers_from_t2i(tmp_path):
    gen = _T2IGen()
    img, actual = _make_keyframe(
        "t2i", _entry(description="<the boy> kneels by the shards"),
        gen, None, None, tmp_path, seed=0)
    assert img is not None
    assert gen.prompts == ["the boy kneels by the shards"]
    assert "<" not in actual


def test_image_plan_fallback_first_frame_uses_opening_frame(tmp_path):
    # P2-⑦ 确定性兜底:brain 没给逐张 spec → 首帧槽位用开场静态快照
    gen = _T2IGen()
    entry = _entry(description="Shot 1: <the cat> leaps onto the sill",
                   opening_frame="the cat sits below the windowsill")
    plan, images, degraded = _execute_image_plan(
        {"strategy": "single_first_frame"}, entry, gen, None, None, tmp_path)
    assert plan == "single_first_frame" and degraded == ""
    assert gen.prompts == ["the cat sits below the windowsill"]
    assert images[0]["description"] == "the cat sits below the windowsill"


def test_image_plan_fallback_without_opening_frame_uses_description(tmp_path):
    gen = _T2IGen()
    entry = _entry(description="Shot 2: <the cat> naps on the sill")
    plan, images, degraded = _execute_image_plan(
        {"strategy": "single_first_frame"}, entry, gen, None, None, tmp_path)
    # 快照缺席 → 照旧 shot 描述,且标记已剥
    assert gen.prompts == ["Shot 2: the cat naps on the sill"]


def test_image_plan_brain_spec_query_stripped(tmp_path):
    # brain 抄 shot 描述连标记一起抄 → 出口剥除
    gen = _T2IGen()
    plan, images, degraded = _execute_image_plan(
        {"strategy": "single_first_frame",
         "images": [{"source": "t2i",
                     "description": "<the boy> stands in the kitchen"}]},
        _entry(), gen, None, None, tmp_path)
    assert gen.prompts == ["the boy stands in the kitchen"]
