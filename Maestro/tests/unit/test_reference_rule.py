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
    # 已有原文台词 → 不重复、不剥
    p2 = '<<<image_2>>>说:"真可怜啊，公爵千金……"。The frame settles.'
    assert wl._with_dialogue(p2, e, CAST) == p2


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
