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


# ── 2026-08-06 xiaoming run2 三连事故回归 ─────────────────────────────

def test_dialogue_coverage_detects_truncation():
    """scene_write 台词截半("大海真大啊。"丢后半句)→ 判据必须报警。"""
    sp = '小明说："大海真大啊，大到能吞掉我所有的失败。"他望着海。'
    bad = [{"dialogue": "大海真大啊。"}]
    cov = wl._dialogue_coverage(sp, bad)
    assert not cov["ok"]
    good = [{"dialogue": "大海真大啊，大到能吞掉我所有的失败。"}]
    assert wl._dialogue_coverage(sp, good)["ok"]


def test_dialogue_patch_completes_sentence():
    """确定性补丁:腰斩句沿剧本补到句末;句级完整摘抄不动。"""
    sp = '小明说："大海真大啊，大到能吞掉我所有的失败。"完。'
    shots = [{"dialogue": "大海真大啊。"}]
    wl._patch_dialogue_coverage(sp, shots)
    assert shots[0]["dialogue"] == "大海真大啊，大到能吞掉我所有的失败。"
    shots2 = [{"dialogue": "大海真大啊，大到能吞掉我所有的失败。"}]
    assert wl._patch_dialogue_coverage(sp, shots2) == []


def test_short_quoted_names_not_speech():
    """引号内短名号("阿浪")不是台词块,不得报缺。"""
    sp = '海鸥"阿浪"俯冲下来。阿浪说："喂，人类。"'
    cov = wl._dialogue_coverage(sp, [{"dialogue": "喂，人类。"}])
    assert cov["ok"]


def test_names_to_tokens_outside_quotes_only():
    """共享名字终换闸:引号外裸名 → 记号;台词引号内原样。"""
    ns = {"小明": "<<<image_2>>>"}
    out = wl._names_to_tokens('小明凝望海面，说："小明不会认输。"', ns)
    assert out.startswith("<<<image_2>>>凝望")
    assert '"小明不会认输。"' in out


def test_regen_anchor_language_follows_project():
    """全修锚句语言随项目:zh 时不得出现英文脚手架。"""
    from maestro.language import set_output_lang
    set_output_lang("zh")
    try:
        out = wl._regen_prompt(
            "ref2v", "base", "修正提示",
            [{"slot": "<<<image_1>>>", "referenceable": True,
              "content": "c"}],
            action="<<<image_1>>>紧攥拳头", end_state="静止")
        assert "scripted action" not in out
        assert "本镜剧本动作" in out
    finally:
        set_output_lang("en")


def test_dialogue_coverage_accepts_dict_shape():
    """2026-08-06 run4 事故回归:scene_write 的 dialogue 是 {speaker,line}
    字典,判据必须读 line,不得把字典串当台词。"""
    sp = '小明说："大海真大啊，大到能吞掉我所有的失败。"完。'
    shots = [{"dialogue": {"speaker": "小明",
                           "line": "大海真大啊，大到能吞掉我所有的失败。"}}]
    assert wl._dialogue_coverage(sp, shots)["ok"]
    bad = [{"dialogue": {"speaker": "小明", "line": "大海真大啊。"}}]
    cov = wl._dialogue_coverage(sp, bad)
    assert not cov["ok"]
    wl._patch_dialogue_coverage(sp, bad)
    assert bad[0]["dialogue"]["line"] == "大海真大啊，大到能吞掉我所有的失败。"


# ── 2026-08-06 rainnight 治本回归 ─────────────────────────────────────

def test_given_character_absent_from_screenplay_is_skipped():
    """role 字典复制残留:名字不在剧本 → 跳过绑定(闸在 generate 主链,
    这里锁声词判据与黑名单,主链行为由集成测试覆盖)。"""
    words = wl._scripted_sounds("雨滴敲击车窗声,枪声响起,他低声说话")
    assert "枪声" in words
    assert all(not w.endswith("低声") for w in words)   # 言说姿态≠环境声
    assert any(w.endswith("声") for w in words)


def test_sound_blacklist_rejects_speech_manner():
    for t in ("他低声说", "她轻声回应", "全场无声", "厉声喝道"):
        assert wl._scripted_sounds(t) == []


def test_sound_words_are_whole_names_per_shot():
    """2026-08-06 rainnight 回归:锚定边界整名捕获,拒前缀垃圾;
    "轮回应声"这类跨词误切必须消失。"""
    t = ("近景特写。旁白:还是命运的轮回应声开启？"
         "音效:冰冷金属摩擦声、玻璃碎裂声")
    words = wl._scripted_sounds(t)
    assert "冰冷金属摩擦声" in words and "玻璃碎裂声" in words
    assert all("轮回" not in w for w in words)


def test_character_extract_enforces_name_language():
    """2026-08-06 rainnight run2 事故回归:zh 剧本派生角色名必须中文,
    英文名触发纠正重试;重试后合规即收。"""
    class _RetryLLM:
        def __init__(self):
            self.n = 0
        def complete(self, prompt, **kw):
            self.n += 1
            if self.n == 1:
                return ('{"characters": {"the gunman": "static: tall; '
                        'dynamic: suit"}}')
            return ('{"characters": {"黑帮老大": "static: tall; '
                    'dynamic: suit"}}')
    llm = _RetryLLM()
    chars, via = wl._extract_characters(llm, "雨夜,黑帮老大点燃雪茄。",
                                        prompt_language="zh")
    assert llm.n == 2                       # 纠正重试发生
    assert list(chars) == ["黑帮老大"]


def test_sound_coverage_detects_dropped_sounds():
    """2026-08-06 rainnight run3 回归:剧本载明"雨滴敲击车窗声",分镜
    描述丢声词 → 判据必须报缺;带上则通过。"""
    sp = "雨夜。音效:雨滴敲击车窗声、巨大枪声"
    bad = [{"description": "雨滴敲击车身,轿车静停。"},
           {"description": "枪口火焰照亮雨巷,巨大枪声爆发。"}]
    missing = wl._sound_coverage(sp, bad)
    assert "雨滴敲击车窗声" in missing and "巨大枪声" not in missing
    good = [{"description": "雨滴敲击车窗声中,轿车静停。"},
            {"description": "巨大枪声爆发。"}]
    assert wl._sound_coverage(sp, good) == []


def test_narration_stripped_from_prompts():
    """2026-08-06 rainnight run4 回归:旁白文本不得进视频 prompt
    (无旁白通道,烧字幕风险),声效句保留。"""
    import re
    pat = (r"(?:画外)?旁白[:：]?\s*[\"“][^\"“”]*[\"”]。?\s*")
    t = ('雨滴敲击车窗。旁白:“霓虹闪烁的雨夜。”伴随雪茄吸入声。')
    out = re.sub(pat, "", t)
    assert "旁白" not in out and "霓虹闪烁的雨夜" not in out
    assert "雪茄吸入声" in out
