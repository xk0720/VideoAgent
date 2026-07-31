"""attempt3 循环病因回归(2026-07-18 裁决,用户实测裁剪版 prompt 有效):
P0-A 锚定/无锚条件行注记(setting 不再诱导建景句);
P0-B 全修 prompt 合成:hint 替换原动作,绝不 " Fix: " 追加;
P0-C "static:/dynamic:" 契约标签永不进 prompt(精确清洗 + 出口收口);
P1-D episode/fallback 决策 log 与 llm 路径同口径(带 context)。
"""

import json

from maestro.pipeline.window_loop import (
    _conditions_for_prompt,
    _decide,
    _regen_prompt,
    _scrub_cast_labels,
    _PIN_SENTENCE,
)
from maestro.memory.storyboard import ShotEntry

CAST = {"the cat": "static: small orange-and-white shorthair cat with "
                   "amber eyes and white paws; dynamic: expression, pose, "
                   "movement, and eating"}


def _entry(description="the cat trots to the bowl"):
    return ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                     description=description)


# ── P0-C:契约标签清洗 ───────────────────────────────────────────────

def test_scrub_replaces_verbatim_contract_with_static_half():
    # attempt3 实拍:enhancer 把契约值连标签整串贴进了 prompt
    txt = ("@Image2 supplies the cat's appearance: " + CAST["the cat"]
           + ". The cat reaches the bowl and eats.")
    out = _scrub_cast_labels(txt, CAST)
    assert "static:" not in out and "dynamic:" not in out
    assert ("small orange-and-white shorthair cat with amber eyes and "
            "white paws" in out)
    assert "The cat reaches the bowl and eats." in out


def test_scrub_strips_bare_static_label():
    out = _scrub_cast_labels("From this frame, the static: small cat "
                             "wakes up.", None)
    assert "static:" not in out
    assert "the small cat wakes up." in out


def test_scrub_safe_on_empty_and_no_cast():
    assert _scrub_cast_labels("", None) == ""
    assert _scrub_cast_labels("plain prompt", {}) == "plain prompt"


def test_scrub_leaves_paraphrased_dynamic_alone(caplog):
    # brain 改写过整串 → 无法安全定界,只告警不动刀(诚实边界)
    import logging

    logging.getLogger("maestro").propagate = True
    txt = "the cat, dynamic: running fast, crosses the floor"
    with caplog.at_level(logging.WARNING):
        out = _scrub_cast_labels(txt, CAST)
    assert out == txt
    assert any("dynamic" in r.getMessage() for r in caplog.records)


# ── P0-B:全修 prompt 合成 ───────────────────────────────────────────

SLOTS = [
    {"slot": "@Image1", "referenceable": True,
     "content": "the previous shot's final frame"},
    {"slot": "@Image2", "referenceable": True,
     "content": "user asset: an orange and white cat"},
]


def test_regen_hint_replaces_base_with_pin():
    base = "OLD long noisy prompt with a scene sentence"
    hint = ("The cat reaches the bowl, stops naturally, and eats; "
            "preserve the established scene, lighting and camera.")
    out = _regen_prompt("ti2v_prev_plus_keyframe", base, hint, SLOTS)
    assert out.startswith(_PIN_SENTENCE)
    assert hint in out
    assert "OLD long noisy prompt" not in out
    assert " Fix: " not in out
    # hint 没提 @Image2 → 引用闸门自动补句(素材不白传)
    assert "@Image2" in out


def test_regen_no_pin_for_channel_locked_routes():
    # i2v 路线的真实槽位 = 不可引用的 FIRST_FRAME(通道级锁,无 @ 语法)
    i2v_slots = [{"slot": "FIRST_FRAME", "referenceable": False,
                  "content": "the previous shot's final frame"}]
    out = _regen_prompt("ti2v_prev_last", "base", "the cat eats.",
                        i2v_slots)
    assert out == "the cat eats."


def test_regen_empty_hint_keeps_base():
    assert _regen_prompt("ti2v_prev_plus_keyframe", "base prompt", "",
                         SLOTS) == "base prompt"


def test_regen_unknown_ref_in_hint_falls_back_to_base():
    out = _regen_prompt("ti2v_prev_plus_keyframe", "base prompt",
                        "continue @Image9 exactly", SLOTS)
    assert out == "base prompt"


# ── P0-A:锚定/无锚条件行注记 ────────────────────────────────────────

def test_setting_note_anchored_forbids_scene_sentence():
    conds = _conditions_for_prompt(
        "ti2v_prev_plus_keyframe", _entry(), None, False,
        cast=CAST, setting="a cozy living room with warm sunlight")
    row = next(c for c in conds if c["kind"] == "setting")
    assert "do NOT restate" in row["note"]
    assert "preserve" in row["note"]
    cast_row = next(c for c in conds if c["kind"] == "cast")
    assert "ONE short identity clause" in cast_row["note"]
    assert "never enter the prompt" in cast_row["note"]


def test_setting_note_unanchored_requires_weaving():
    conds = _conditions_for_prompt(
        "t2v", _entry(), None, False,
        cast=CAST, setting="a cozy living room with warm sunlight")
    row = next(c for c in conds if c["kind"] == "setting")
    assert "unanchored" in row["note"]
    assert "only scene carrier" in row["note"]
    cast_row = next(c for c in conds if c["kind"] == "cast")
    assert "full static half" in cast_row["note"]


# ── P1-D:重放/兜底决策 log 带 context(观测口径一致)─────────────────

def test_decide_replay_and_fallback_log_context(tmp_path):
    """2026-07-31 修订:episode 命中不再短路 —— 建议注入上下文后照常走
    LLM/兜底;log 口径仍须带 context(观测诚实)。"""
    from maestro.logging_utils import set_brain_log

    logf = tmp_path / "brain.jsonl"
    set_brain_log(logf)
    try:
        menu = [{"name": "t2v"}, {"name": "ti2v_prev_last"}]
        ctx = {"shot": {"label": "scene 1 shot 3"},
               "junction": {"prev_last_frame_actual": "cat at bowl"}}
        # llm=None → _brain_pick 失败 → 确定性兜底;建议应已注入 context
        d = _decide(None, "generation-condition", menu, ctx,
                    replay_hint="ti2v_prev_last", priority=["t2v"])
        assert d["via"] == "fallback"
        d = _decide(None, "generation-condition", menu, ctx,
                    replay_hint=None, priority=["t2v"])
        assert d["via"] == "fallback"
    finally:
        set_brain_log(None)
    recs = [json.loads(l) for l in logf.read_text().splitlines()]
    fb = [r for r in recs if r.get("via") == "fallback"]
    assert len(fb) == 2
    # 第一条(有 replay_hint)的 context 必须带 episode_recommendation
    assert fb[0]["context"]["episode_recommendation"]["strategy"] == \
        "ti2v_prev_last"
    for r in fb:
        assert r["context"]["junction"]["prev_last_frame_actual"] == \
            "cat at bowl"
        assert r["menu"] == ["t2v", "ti2v_prev_last"]


# ── 二轮修订(2026-07-18):hint 动作保底 + 建景句拦截 + 警戒线 ────────

def test_regen_action_anchor_guarantees_motion():
    # hint 只写外观(纯身份缺陷)→ 剧本动作锚保证 motion 在场
    hint = "Ensure the cat's coat matches the reference exactly."
    out = _regen_prompt(
        "ti2v_prev_plus_keyframe", "base", hint, SLOTS,
        action="Shot 4: scene 1 — the cat reaches the bowl and eats",
        end_state="the cat stands at the bowl, head lowered, eating")
    assert out.startswith(_PIN_SENTENCE)
    assert hint in out
    # "Shot N: scene N —" 台账前缀已剥;起点/过程/终点三件套齐
    assert "This shot's scripted action: the cat reaches the bowl" in out
    assert "Shot 4" not in out
    assert "ending as: the cat stands at the bowl" in out


def test_regen_action_anchor_without_end_state():
    out = _regen_prompt("extend_prev", "base", "fix the paw.", [],
                        action="the cat trots on")
    assert out == ("fix the paw. This shot's scripted action: "
                   "the cat trots on.")


def test_setting_scrub_verbatim_replaced_case_insensitive():
    from maestro.pipeline.window_loop import (
        _scrub_setting_sentence,
        _PRESERVE_CLAUSE,
    )

    setting = ("a cozy living room with a wooden windowsill, honey-colored "
               "wood floor, cream sofa and warm morning sunlight")
    # attempt3 实拍:enhancer 把 setting 首字母大写后整句贴进 prompt
    prompt = ("The shot opens EXACTLY on @Image1. A cozy living room with "
              "a wooden windowsill, honey-colored wood floor, cream sofa "
              "and warm morning sunlight. The cat reaches the bowl.")
    out = _scrub_setting_sentence(prompt, setting,
                                  "ti2v_prev_plus_keyframe")
    assert "cozy living room" not in out
    assert _PRESERVE_CLAUSE.rstrip(".") in out
    assert "The cat reaches the bowl." in out


def test_setting_scrub_unanchored_untouched():
    from maestro.pipeline.window_loop import _scrub_setting_sentence

    setting = "a cozy living room with warm morning sunlight"
    prompt = f"A wide shot. {setting}. The cat enters."
    assert _scrub_setting_sentence(prompt, setting, "t2v") == prompt


def test_setting_scrub_paraphrase_warns_only(caplog):
    import logging

    from maestro.pipeline.window_loop import _scrub_setting_sentence

    logging.getLogger("maestro").propagate = True
    setting = ("a cozy living room with a wooden windowsill, honey-colored "
               "wood floor, cream sofa and warm morning sunlight")
    # 改写版:词都在,句子变了 → 无法安全定界,不动刀只告警
    prompt = ("Inside the cozy living room, near the wooden windowsill and "
              "cream sofa, warm morning sunlight over the honey-colored "
              "wood floor, the cat trots.")
    with caplog.at_level(logging.WARNING):
        out = _scrub_setting_sentence(prompt, setting,
                                      "ti2v_prev_plus_keyframe")
    assert out == prompt
    assert any("paraphrase" in r.getMessage() for r in caplog.records)
