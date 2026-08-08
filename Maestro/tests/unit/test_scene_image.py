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

def test_junction_instructions_demand_tail_report_json():
    """片尾理解报告(2026-08-07 用户令):两部分 —— camera_angle +
    character_actions;具名法保留(who=角色名,按肖像认人)。"""
    from maestro.models.mllm_backends import (_JUNCTION_INSTRUCTION,
                                              _JUNCTION_VIDEO_INSTRUCTION)
    for t in (_JUNCTION_VIDEO_INSTRUCTION, _JUNCTION_INSTRUCTION):
        assert "STRICT JSON" in t
        assert '"camera_angle"' in t and '"character_actions"' in t
    assert "MATCH them against the portraits" in _JUNCTION_VIDEO_INSTRUCTION


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


def test_skills_carry_tail_report_and_settle_laws():
    """三条件融合派(2026-08-07 用户令):enhancer 承接 = 片尾报告;
    cut/derive 禁写连续性;SETTLE-TO-CUT 恒在。"""
    from pathlib import Path as _P
    base = _P("src/maestro/skills/brain_skills")
    pe = (base / "prompt_enhancer/SKILL.md").read_text()
    for marker in ("OPENING FROM REALITY", "CONTINUATIVE PHRASING",
                   "SETTLE-TO-CUT", "prev_tail_report", "junction_kind"):
        assert marker in pe, marker
    sw = (base / "scene_write/SKILL.md").read_text()
    assert "SETTLE-TO-CUT" in sw
    wg = (base / "window_generation/SKILL.md").read_text()
    for marker in ("junction_kind", "prev_tail_report", "`cut`",
                   "`derive`", "pin_frame"):
        assert marker in wg, marker


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
    """具名绑定法(2026-08-05)三条件版:报告的 who 已记号化,写手照
    条目绑定,绝不按剧本臆测重排位。"""
    from pathlib import Path as _P
    base = _P("src/maestro/skills/brain_skills")
    assert "already tokenized" in \
        (base / "window_generation/SKILL.md").read_text()
    assert "Bind each token" in \
        (base / "prompt_enhancer/SKILL.md").read_text()


def test_map_junction_tokenizes_and_strips_unresolved():
    """用户令(2026-08-05):矢量映射在数据层做 —— 有槽者 who→记号;
    无槽者(未上肖像的 cast + 幻觉路人)个体描述删除,只留无描述计数。"""
    import maestro.pipeline.window_loop as wl
    vec = {"subjects": [
        {"who": "安娜", "position": "center near", "pose": "back to camera",
         "motion": "at_rest"},
        {"who": "安莉希娅", "position": "left near", "pose": "gold gown",
         "motion": "at_rest"},
        {"who": "woman in pink dress", "position": "right near",
         "pose": "pink dress", "motion": "at_rest"}],
        "camera": {"framing": "medium", "motion": "static", "speed": "none"},
        "unfinished_action": None}
    ns = {"安娜": "<<<image_2>>>"}
    out = wl._map_junction(vec, ns, {"安娜": "x", "安莉希娅": "y"})
    assert out["subjects"] == [{"who": "<<<image_2>>>",
                                "position": "center near",
                                "pose": "back to camera",
                                "motion": "at_rest"}]
    bg = out["background_figures"]
    assert "2 unresolved" in bg and "NEVER describe" in bg
    assert "pink" not in bg and "gold" not in bg          # 个体描述已删
    assert out["camera"]["motion"] == "static"


def test_map_markers_replaces_with_tokens():
    import maestro.pipeline.window_loop as wl
    out = wl._map_markers("<安娜>背对镜头静立前景，<芬莱克殿下>严厉面对她",
                          {"安娜": "<<<image_2>>>",
                           "芬莱克殿下": "<<<image_3>>>"})
    assert out == "<<<image_2>>>背对镜头静立前景，<<<image_3>>>严厉面对她"
    # 无槽者:去尖括号留名(名字泄漏闸兜底)
    assert wl._map_markers("<安莉希娅>仍挽住", {}) == "安莉希娅仍挽住"


def test_map_junction_resolves_by_portrait_path():
    """用户令:同脸不同名(共用肖像)按 portraits 路径判等 —— 矢量认作
    男性军官,本镜清单只有军官甲(同一张图)→ 归到军官甲的记号。"""
    import maestro.pipeline.window_loop as wl
    vec = {"subjects": [
        {"who": "男性军官", "position": "left", "pose": "navy coat",
         "motion": "at_rest"}],
        "camera": {}, "unfinished_action": None}
    portraits = {"男性军官": "/x/ComfyUI_00002_.png",
                 "军官甲": "/x/ComfyUI_00002_.png"}
    out = wl._map_junction(vec, {"军官甲": "<<<image_2>>>"}, {},
                           portraits=portraits)
    assert out["subjects"][0]["who"] == "<<<image_2>>>"
    assert "background_figures" not in out


def test_junction_router_pin_vs_cut():
    """钉/切路由(2026-08-05 用户令):本镜主体 ⊆ 末帧可见主体 → 续拍;
    否则切换(+运镜桥)。名字直配或肖像路径判等;矢量缺失保守钉帧。"""
    import maestro.pipeline.window_loop as wl
    P = {"安娜": "/x/14.png", "安莉希娅": "/x/01.png",
         "芬莱克殿下": "/x/16.png", "军官甲": "/x/02.png",
         "男性军官": "/x/02.png"}
    vec_anna = {"subjects": [{"who": "安娜", "position": "c"}]}
    # 3→4:末帧只有安娜,新镜是安莉希娅+王子 → 切
    assert not wl._junction_is_continuation(
        ["安莉希娅", "芬莱克殿下"], vec_anna, P)
    # 1→2:安娜在末帧 → 钉
    assert wl._junction_is_continuation(["安娜"], vec_anna, P)
    # 肖像路径判等:矢量认作男性军官,本镜是军官甲(同图)→ 钉
    vec_officer = {"subjects": [{"who": "男性军官", "position": "l"}]}
    assert wl._junction_is_continuation(["军官甲"], vec_officer, P)
    # 矢量缺失 → 保守钉帧
    assert wl._junction_is_continuation(["安娜"], None, P)
