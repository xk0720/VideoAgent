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
    assert "NO-PRINCIPALS LAW" in t
    assert "SCRIPTED POPULATION" in t
    assert "PERIOD CONTRACT" in t
    assert "no modern objects" in t
    assert "ONE bg_id, ONE space" in t


def test_scrub_bg_prompt_strips_names_keeps_scripted_crowd():
    """2026-08-05 用户令:板上无主角,剧本人群放行。"""
    out = wl._scrub_bg_prompt(
        "A vast royal ballroom with crystal chandeliers, gilded columns, "
        "masked aristocratic guests, uniformed officers, warm candlelight. "
        "安娜 crosses toward 芬莱克殿下", ["安娜", "芬莱克殿下"])
    assert "guests" in out and "officers" in out          # 剧本人群保留
    assert "安娜" not in out and "芬莱克殿下" not in out    # 主角名剥除
    assert out.endswith("no modern objects.")
    assert "no principal characters" in out


class _SB:
    setting = "A vast royal ballroom with chandeliers"
    cast = {"安娜": "static: x; dynamic: y"}
    entries = []


def test_write_bg_prompts_llm_path_scrubbed():
    class _LLM:
        def complete(self, prompt, **k):
            assert "NO-PRINCIPALS LAW" in prompt        # skill 进了上下文
            return json.dumps({"backgrounds": {"bg_1": {
                "prompt": "Empty palace hall, marble floor, 安娜 waiting, "
                          "elegant guests, candlelight"}}})
    got, via = wl._write_bg_prompts(_LLM(), _SB(), ["bg_1"])
    assert via == "llm"
    assert "安娜" not in got["bg_1"]
    assert "guests" in got["bg_1"]                       # 剧本人群放行
    assert got["bg_1"].endswith("no modern objects.")


def test_write_bg_prompts_fallback_has_no_location_context():
    class _Bad:
        def complete(self, prompt, **k):
            return "not json"
    got, via = wl._write_bg_prompts(_Bad(), _SB(), ["bg_1"])
    assert via == "fallback"
    assert "Location context" not in got["bg_1"]
    assert "no principal characters" in got["bg_1"]


def test_bg_frame_upgrade_default_off():
    sig = inspect.signature(wl.generate_movie_windowed)
    assert sig.parameters["enable_bg_frame_upgrade"].default is False


# ── 出场矢量 + 静接律(接缝一致性机制)───────────────────────────────

def test_junction_instructions_demand_exit_vector_json():
    from maestro.models.mllm_backends import (_JUNCTION_INSTRUCTION,
                                              _JUNCTION_VIDEO_INSTRUCTION)
    for t in (_JUNCTION_VIDEO_INSTRUCTION, _JUNCTION_INSTRUCTION):
        assert "EXIT VECTOR" in t and "STRICT JSON" in t
        assert '"unfinished_action"' in t and '"camera"' in t


def test_parse_exit_vector_json_and_prose_fallback():
    import maestro.pipeline.window_loop as wl
    v = wl._parse_exit_vector(
        '{"subjects": [{"who": "the woman", "motion": "moving", '
        '"direction": "left", "pace": "walking"}], '
        '"camera": {"framing": "wide", "motion": "pan-left", '
        '"speed": "slow"}, "unfinished_action": null}')
    assert v["camera"]["motion"] == "pan-left"
    assert wl._parse_exit_vector("the woman stands at rest") is None
    assert wl._parse_exit_vector("") is None


def test_skills_carry_exit_vector_and_settle_laws():
    from pathlib import Path as _P
    base = _P("src/maestro/skills/brain_skills")
    pe = (base / "prompt_enhancer/SKILL.md").read_text()
    for marker in ("ENTRY ALIGNMENT", "VELOCITY HANDOFF",
                   "FINISH THE GESTURE", "SETTLE-TO-CUT"):
        assert marker in pe, marker
    sw = (base / "scene_write/SKILL.md").read_text()
    assert "SETTLE-TO-CUT" in sw
    wg = (base / "window_generation/SKILL.md").read_text()
    assert "EXIT VECTOR" in wg and "VELOCITY HANDOFF" in wg


def test_draft_prompt_persists_in_ledger(tmp_path):
    from maestro.memory.storyboard import StoryboardMemory
    sb = StoryboardMemory.from_outline(["shot 1: x"],
                                       path=tmp_path / "sb.json")
    sb.entries[0].draft_prompt = "raw brain draft"
    sb._save()
    back = StoryboardMemory.load(tmp_path / "sb.json")
    assert back.entries[0].draft_prompt == "raw brain draft"


def test_ablation_strip_pin_dependency():
    sys_path_hack = __import__("sys").path
    from pathlib import Path as _P
    sys_path_hack.insert(0, str(_P("scripts").resolve().parent / "scripts"))
    from ablation_nopin import strip_pin_dependency
    out = strip_pin_dependency(
        "The shot opens exactly on the pinned first frame. "
        "<<<image_2>>> lowers the fan slowly. "
        "Taking over from the previous shot, guests stir. "
        "The camera stays static.")
    assert "pinned" not in out and "previous shot" not in out
    assert "<<<image_2>>> lowers the fan slowly." in out
    assert "camera stays static" in out


def test_junction_video_instruction_demands_named_subjects():
    from maestro.models.mllm_backends import _JUNCTION_VIDEO_INSTRUCTION as t
    assert "OFFICIAL PORTRAITS" in t
    assert "MUST be that exact character name" in t


def test_junction_state_passes_portraits(tmp_path, monkeypatch):
    """具名矢量:接点调用把肖像表随片发给 VLM(旧后端无参 → 兼容降级)。"""
    import maestro.pipeline.window_loop as wl

    tail = tmp_path / "tail.mp4"
    tail.write_bytes(b"\x00" * 32)
    monkeypatch.setattr(wl, "_cut_tail", lambda v, s, o: tail)
    seen = {}

    class _M:
        def describe_junction_video(self, media, portraits=None):
            seen["portraits"] = portraits
            return '{"subjects": [], "camera": {}}'

    class _Prev:
        video_path = str(tail)
    wl._JUNCTION_CACHE.clear()
    got = wl._junction_state(_M(), _Prev(), tmp_path,
                             portraits={"安娜": "p.png"})
    assert seen["portraits"] == {"安娜": "p.png"}
    assert "subjects" in got

    class _Old:
        def describe_junction_video(self, media):
            return "prose state"
    wl._JUNCTION_CACHE.clear()
    tail2 = tmp_path / "tail2.mp4"
    tail2.write_bytes(b"\x00" * 32)
    monkeypatch.setattr(wl, "_cut_tail", lambda v, s, o: tail2)

    class _Prev2:
        video_path = str(tail2)
    assert wl._junction_state(_Old(), _Prev2(), tmp_path,
                              portraits={"x": "y"}) == "prose state"


def test_skills_carry_named_binding_law():
    from pathlib import Path as _P
    base = _P("src/maestro/skills/brain_skills")
    assert "NAMED SUBJECT BINDING" in \
        (base / "window_generation/SKILL.md").read_text()
    assert "bind each token to its own name's vector entry" in \
        (base / "prompt_enhancer/SKILL.md").read_text()
