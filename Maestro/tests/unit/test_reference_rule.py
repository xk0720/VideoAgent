"""引用铁律(2026-08-04 用户令)机器层回归:
① 对白句说话人记号化(查槽位清单;查不到才落名字);
② 槽位清单肖像行带 name(名字→记号唯一对照源);
③ 正典闸豁免 ref2v(带参考图 = 像素承载身份,永不强插外观);
④ 转场默认关(enable_transitions=False);
⑤ 同 scene 续拍钉帧(菜单收缩为 i2v_first)。全部离线。"""
import inspect

import maestro.pipeline.window_loop as wl
from maestro.memory.storyboard import ShotEntry


def _entry(dialogue="你不配", speaker="王子"):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="shot 2: <安娜> confronts <王子>")
    e.dialogue = dialogue
    e.dialogue_speaker = speaker
    return e


CAST = {"安娜": "static: a; dynamic: b", "王子": "static: c; dynamic: d"}


def test_dialogue_speaker_rendered_as_token():
    slots = [{"slot": "<<<image_1>>>", "referenceable": True,
              "content": "bg"},
             {"slot": "<<<image_2>>>", "referenceable": True,
              "name": "安娜", "content": "official portrait of 安娜"},
             {"slot": "<<<image_3>>>", "referenceable": True,
              "name": "王子", "content": "official portrait of 王子"}]
    out = wl._with_dialogue("base prompt", _entry(), CAST,
                            name_to_slot=wl._name_slot_map(slots))
    assert '<<<image_3>>>说:"你不配"' in out    # 中文台词→中文脚手架
    # 2026-08-04 用户裁决:兜底只补台词 —— 收势句/mouth-moving 不再机械化
    assert "micro-expression" not in out
    assert "mouth moving" not in out
    assert "王子说" not in out


def test_dialogue_speaker_falls_back_to_name_without_slot():
    out = wl._with_dialogue("base prompt", _entry(), CAST,
                            name_to_slot={})       # t2v 降级:无参考图
    assert '王子说:"你不配"' in out


def test_slot_manifest_portrait_rows_carry_name():
    class _VG:
        def capabilities(self):
            return {"t2v", "flf2v", "ref_images", "first_frame_plus_refs"}

        def ref_token(self, n):
            return f"<<<image_{n}>>>"
    e = ShotEntry(shot_idx=1, scene_idx=1, label="s", description="d")
    rows = wl._slot_manifest("ref2v", e, None, portraits={"安娜": "p.png"},
                             video_gen=_VG())
    m = wl._name_slot_map(rows)
    assert m == {"安娜": "<<<image_1>>>"}


def test_ref2v_exempt_from_cast_canon():
    assert "ref2v" in wl._ANCHORED_STRATEGIES        # 像素承载身份 → 豁免
    assert "t2v" not in wl._ANCHORED_STRATEGIES      # 纯文字路线仍受契约


def test_transitions_default_off():
    sig = inspect.signature(wl.generate_movie_windowed)
    assert sig.parameters["enable_transitions"].default is False


def test_foreign_dialogue_scrubbed_before_backstop():
    """2026-08-05 run10 事故回归:brain 把台词译成英文写进节拍,兜底又
    补中文原句 → 英中双份。硬闸:言说动词+引号里不是原句的台词整段
    剥除,再补唯一的原文台词。"""
    e = _entry(dialogue="真可怜啊，公爵千金……", speaker="王子")
    slots = [{"slot": "<<<image_2>>>", "referenceable": True,
              "name": "王子", "content": "official portrait of 王子"}]
    p = ('He leans toward the officer and whispers beneath the glitter: '
         '"How pitiful, the duke\'s daughter..." The camera settles.')
    out = wl._with_dialogue(p, e, CAST,
                            name_to_slot=wl._name_slot_map(slots))
    assert "How pitiful" not in out                 # 英文台词整段剥除
    assert out.count("真可怜啊，公爵千金……") == 1     # 原文只出现一次
    assert '<<<image_2>>>说:"真可怜啊，公爵千金……"' in out
    # 已有原文台词 → 不重复台词,但压制句永远确保在场(run11b 事故)
    p2 = '<<<image_2>>>说:"真可怜啊，公爵千金……"。The frame settles.'
    out2 = wl._with_dialogue(p2, e, CAST)
    assert out2.count("真可怜啊，公爵千金……") == 1
    assert "无背景音乐" in out2


def test_prompt_language_follows_screenplay():
    """2026-08-05 用户令:剧本中文 → prompt 全中文(摘抄优先)。"""
    assert wl._prompt_lang("第一场·盛大舞会,宫廷大舞厅") == "zh"
    assert wl._prompt_lang("SCENE 1 — a bakery, morning") == "en"
    assert wl._prompt_lang("") == "en"
    # skill 法条齐备
    from pathlib import Path as _P
    base = _P("src/maestro/skills/brain_skills")
    wg = (base / "window_generation/SKILL.md").read_text()
    assert "PROMPT LANGUAGE FOLLOWS THE SCRIPT" in wg
    assert "EXCERPT" in wg
    sw = (base / "scene_write/SKILL.md").read_text()
    assert "SCRIPT LANGUAGE LAW" in sw


def test_background_slot_is_always_image_1(tmp_path):
    """布局恒定律(2026-08-05):背景行前插后,即使本镜另有计划图,
    <<<image_1>>> 恒为背景板 —— 编号不再逐镜漂移。"""
    class _VG:
        def capabilities(self):
            return {"t2v", "flf2v", "ref_images", "first_frame_plus_refs"}

        def ref_token(self, n):
            return f"<<<image_{n}>>>"

    e = ShotEntry(shot_idx=1, scene_idx=1, label="s",
                  description="d: <安娜>")
    # 模拟前插后的列表形态:[背景, 计划图]
    bg = tmp_path / "bg.png"
    planned = tmp_path / "planned.png"
    for f in (bg, planned):
        f.write_bytes(b"\x89PNG\r\n")
    e.images = [
        {"path": str(bg), "role": "reference",
         "source": "background", "description": "the OFFICIAL look"},
        {"path": str(planned), "role": "reference",
         "description": "a planned prop"},
    ]
    rows = wl._slot_manifest("i2v_first", e, None,
                             portraits={"安娜": "p.png"}, video_gen=_VG())
    ref_rows = [r for r in rows if r.get("referenceable")]
    assert ref_rows[0]["slot"] == "<<<image_1>>>"
    assert "OFFICIAL look" in ref_rows[0]["content"]
    assert ref_rows[1]["slot"] == "<<<image_2>>>"
    assert wl._name_slot_map(rows)["安娜"] == "<<<image_3>>>"


def test_decision_prompt_language_aware():
    """2026-08-05 run11 事故:decision_prompt 尾部硬编码"全英文"压死中文
    制。zh 上下文 → video_prompt 要求中文;en 上下文 → 维持全英文令。"""
    zh = wl.decision_prompt("SKILL", [], {"prompt_language": "zh"})
    assert "video_prompt in CHINESE" in zh
    en = wl.decision_prompt("SKILL", [], {"prompt_language": "en"})
    assert "must be in ENGLISH" in en


def test_outline_language_gate_retries_then_falls_back():
    """zh 项目分镜写成英文 → 纠正重试;仍英文 → 摘抄兜底(拆剧本原文)。"""
    import json as _json
    replies = [
        _json.dumps({"cast": {}, "setting": "hall", "shots": [
            {"description": "Shot 1: the camera finds the prince",
             "duration_s": 5, "end_state": "still"}]}),
        _json.dumps({"cast": {}, "setting": "hall", "shots": [
            {"description": "Shot 1: 镜头从大远景推向<王子>",
             "duration_s": 5, "end_state": "王子静立"}]}),
    ]

    class _LLM:
        def __init__(self):
            self.n = 0

        def complete(self, prompt, **k):
            r = replies[min(self.n, 1)]
            self.n += 1
            return r
    shots, _d, _e, _m, via = wl._write_outline(
        _LLM(), "第一场·盛大舞会", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["兜底"], prompt_language="zh")
    assert via == "llm" and "镜头从大远景" in shots[0]   # 纠正重试拿到中文

    class _AlwaysEn:
        def complete(self, prompt, **k):
            return replies[0]
    shots2, _d2, _e2, _m2, via2 = wl._write_outline(
        _AlwaysEn(), "第一场·盛大舞会", [], episode_guidance={},
        max_shots=3, fallback_fn=lambda: ["剧本原文摘抄"],
        prompt_language="zh")
    assert via2 == "fallback" and shots2 == ["剧本原文摘抄"]


def test_dynamic_output_language_policy():
    """2026-08-05 动态语言:zh 项目所有模型输出中文(键/枚举保持英文);
    en 项目零约束;t2i 中文正典经 LLM 译英护栏后才进 flux。"""
    from maestro.language import lang_clause, output_lang, set_output_lang
    set_output_lang("zh")
    assert output_lang() == "zh"
    assert "CHINESE" in lang_clause("x")
    set_output_lang("en")
    assert lang_clause("x") == ""
    # 肖像翻译护栏:中文 static → llm 译英进模板
    import maestro.pipeline.window_loop as wl
    from maestro.memory.storyboard import StoryboardMemory

    class _T2I:
        def __init__(self):
            self.prompts = []

        def capabilities(self):
            return {"t2i"}

        def text_to_image(self, prompt, out, seed=0):
            self.prompts.append(prompt)
            from pathlib import Path as _P
            p = _P(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x89PNG\r\n" + b"\x00" * 8)
            return p

    class _LLM:
        def complete(self, prompt, **k):
            return "slender woman in a deep purple velvet gown"
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path as _P
        sb = StoryboardMemory.from_outline(["shot 1: <甲> 出场"],
                                           path=_P(td) / "sb.json")
        sb.cast = {"甲": "static: 深紫色丝绒长裙的女子; dynamic: 折扇"}
        sb.setting = "a grand hall"
        gen = _T2I()
        wl._ensure_cast_portraits(sb, None, gen, _P(td), llm=_LLM())
        assert gen.prompts and "深紫色" not in gen.prompts[0]
        assert "purple velvet" in gen.prompts[0]


def test_speaker_pronoun_replaced_with_token_when_line_present():
    """2026-08-05 run12 事故:台词在场但说话人是代词("他…并说:")→
    机器闸把言说动词前主语确定性换成说话人记号;已是记号则不动。"""
    e = _entry(dialogue="你这种女人不配做王后，安莉希娅才是真正合适的人选！",
               speaker="王子")
    slots = [{"slot": "<<<image_4>>>", "referenceable": True,
              "name": "王子", "content": "official portrait of 王子"}]
    p = ('待他的脸清晰入镜后，他严厉地公开退婚并说："你这种女人不配做王后，'
         '安莉希娅才是真正合适的人选！"。最终她的肩背静止于前景。')
    out = wl._with_dialogue(p, e, CAST,
                            name_to_slot=wl._name_slot_map(slots))
    import re as _re
    assert _re.search(r"<<<image_4>>>(?:并|随后)?说[:：]", out)   # 记号紧贴言说动词
    assert "他严厉地公开退婚并说" not in out
    # 已是记号主语 → 原样保留
    p2 = '<<<image_4>>>说："你这种女人不配做王后，安莉希娅才是真正合适的人选！"。'
    out2 = wl._with_dialogue(p2, e, CAST,
                             name_to_slot=wl._name_slot_map(slots))
    assert out2.count("<<<image_4>>>") == 1
