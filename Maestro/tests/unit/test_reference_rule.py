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
    assert '<<<image_3>>> says: "你不配"' in out
    assert "<<<image_3>>> holds a natural micro-expression" in out
    assert "王子 says" not in out and "王子 holds" not in out


def test_dialogue_speaker_falls_back_to_name_without_slot():
    out = wl._with_dialogue("base prompt", _entry(), CAST,
                            name_to_slot={})       # t2v 降级:无参考图
    assert '王子 says: "你不配"' in out


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
