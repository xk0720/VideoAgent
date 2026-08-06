"""钉/切提前路由(2026-08-05 用户令):分镜剧本人物比对,生成前定策略。

规则:本镜开场人物 ⊆ 上镜收尾人物 → 钉帧续拍;出现新人物 → 转场
(不钉帧+自动桥);提取不到 → 保守判续拍。同脸不同名按肖像路径判等。
"""
from types import SimpleNamespace

from maestro.pipeline import window_loop as wl

CAST = {"小明": "role", "阿浪": "role", "军官甲": "role", "军官乙": "role"}
PORTRAITS = {"小明": "/p/xm.png", "阿浪": "/p/al.png",
             "军官甲": "/p/officer.png", "军官乙": "/p/officer.png"}


def _e(end_state="", opening="", desc=""):
    return SimpleNamespace(end_state=end_state, opening_frame=opening,
                           description=desc)


def test_same_cast_pins():
    cont, why = wl._script_cast_continuity(
        _e(end_state="<小明>静立海边,镜头停稳。"),
        _e(opening="<小明>的面部特写。"), CAST, PORTRAITS)
    assert cont


def test_new_character_routes_transition():
    cont, why = wl._script_cast_continuity(
        _e(end_state="<小明>凝望海面。"),
        _e(opening="<阿浪>落在礁石上。"), CAST, PORTRAITS)
    assert not cont
    assert "阿浪" in why


def test_subset_still_pins():
    cont, _ = wl._script_cast_continuity(
        _e(end_state="<小明>与<阿浪>对视。"),
        _e(opening="<阿浪>歪头。"), CAST, PORTRAITS)
    assert cont


def test_same_face_different_name_pins():
    """军官甲/乙共用一张肖像 → 按脸判等,不误切。"""
    cont, _ = wl._script_cast_continuity(
        _e(end_state="<军官甲>低声说完。"),
        _e(opening="<军官乙>转过脸回应。"), CAST, PORTRAITS)
    assert cont


def test_no_markers_conservative_pin():
    """旧剧本无 <标记> → _cast_in_shot 诚实降级 → 保守判续拍。"""
    cont, _ = wl._script_cast_continuity(
        _e(end_state="人物静止。"), _e(opening="特写。"), CAST, PORTRAITS)
    assert cont


def test_empty_cast_conservative_pin():
    cont, why = wl._script_cast_continuity(
        _e(end_state="<小明>静立。"), _e(opening="<小明>特写。"), {}, {})
    assert cont
