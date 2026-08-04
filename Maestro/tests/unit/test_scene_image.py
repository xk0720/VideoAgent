"""场景板三修复回归(2026-08-04 run8 背景污染事故):
① scene_image skill 存在且立五律(空景/时代/几何/光/一空间);
② 出口闸确定性:角色名剥除、人群词段丢弃、空景后缀恒定;
③ _write_bg_prompts:LLM 走 skill、失败落确定性模板(无 Location
   context —— 幽灵新人来源已废除);
④ 帧升级默认关(实拍帧带主人公 = 身份噪声扩散器)。全部离线。"""
import inspect
import json
from pathlib import Path

import maestro.pipeline.window_loop as wl


def test_scene_image_skill_carries_the_laws():
    t = Path("src/maestro/skills/brain_skills/scene_image/SKILL.md").read_text()
    assert "EMPTY-PLATE LAW" in t
    assert "PERIOD CONTRACT" in t
    assert "no modern objects" in t
    assert "ONE bg_id, ONE space" in t


def test_scrub_bg_prompt_strips_names_and_crowd_segments():
    out = wl._scrub_bg_prompt(
        "A vast royal ballroom with crystal chandeliers, gilded columns, "
        "masked aristocratic guests, uniformed officers, warm candlelight. "
        "安娜 crosses toward 芬莱克殿下", ["安娜", "芬莱克殿下"])
    assert "guests" not in out and "officers" not in out
    assert "安娜" not in out and "芬莱克殿下" not in out
    assert "crystal chandeliers" in out and "warm candlelight" in out
    assert out.endswith("no modern objects.")
    assert "no people" in out


class _SB:
    setting = "A vast royal ballroom with chandeliers"
    cast = {"安娜": "static: x; dynamic: y"}
    entries = []


def test_write_bg_prompts_llm_path_scrubbed():
    class _LLM:
        def complete(self, prompt, **k):
            assert "EMPTY-PLATE LAW" in prompt          # skill 进了上下文
            return json.dumps({"backgrounds": {"bg_1": {
                "prompt": "Empty palace hall, marble floor, 安娜 waiting, "
                          "elegant guests, candlelight"}}})
    got, via = wl._write_bg_prompts(_LLM(), _SB(), ["bg_1"])
    assert via == "llm"
    assert "安娜" not in got["bg_1"] and "guests" not in got["bg_1"]
    assert got["bg_1"].endswith("no modern objects.")


def test_write_bg_prompts_fallback_has_no_location_context():
    class _Bad:
        def complete(self, prompt, **k):
            return "not json"
    got, via = wl._write_bg_prompts(_Bad(), _SB(), ["bg_1"])
    assert via == "fallback"
    assert "Location context" not in got["bg_1"]
    assert "no people" in got["bg_1"]


def test_bg_frame_upgrade_default_off():
    sig = inspect.signature(wl.generate_movie_windowed)
    assert sig.parameters["enable_bg_frame_upgrade"].default is False
