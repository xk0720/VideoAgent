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


def test_pin_frame_slot_never_auto_appended():
    """2026-08-06 rainnight run4 回归:pin_frame 行是执行器专有——
    补挂闸不得把它的英文说明文本塞进 prompt。"""
    from maestro.pipeline.ref_slots import validate_references
    slots = [{"slot": "<<<image_1>>>", "referenceable": True,
              "content": "the scene plate"},
             {"slot": "<<<image_2>>>", "referenceable": True,
              "source": "pin_frame",
              "content": "the first frame itself (executor owns its "
                         "mention — never reference this slot yourself)"}]
    fixed, audit = validate_references("<<<image_1>>>内,人物静立。", slots)
    assert audit["ok"] and "<<<image_2>>>" not in fixed
    assert "executor owns" not in fixed


def test_pinned_shots_drop_background_plate(tmp_path):
    """2026-08-06 rainnight run4 板磁铁事故回归:硬钉上镜末帧的
    i2v_first 清单不得携带背景板 refer(空板会把人清场)。"""
    from types import SimpleNamespace

    class _E:
        keyframe_path = None

        def __init__(self, images):
            self.images = images

        def images_by_role(self, role):
            return [im for im in self.images if im.get("role") == role]

    bg = tmp_path / "bg.png"; bg.write_bytes(b"x")
    prop = tmp_path / "prop.png"; prop.write_bytes(b"x")
    entry = _E([
        {"path": str(bg), "role": "reference",
         "source": "background", "description": "plate"},
        {"path": str(prop), "role": "reference",
         "source": "asset_image", "description": "prop"}])
    prev = SimpleNamespace(video_path=str(tmp_path / "prev.mp4"))
    rows = wl._slot_manifest("i2v_first", entry, prev,
                             portraits={"甲": str(tmp_path / "p.png")})
    contents = " | ".join(str(r.get("content")) for r in rows)
    assert "plate" not in contents          # 板被剔除
    assert "prop" in contents               # 非板自有图保留


def test_sound_coverage_mid_sentence_no_false_alarm():
    """2026-08-06 cinegraph run3 误报:声词嵌在句中(距标点>7字),锚定
    正则重提取为空,在场声词被冤成全缺 → 覆盖判据直查全文子串。"""
    from maestro.pipeline.window_loop import (_sound_coverage,
                                              set_run_ambience,
                                              set_run_sound_lexicon)
    set_run_ambience()
    set_run_sound_lexicon()
    sp = ("暗巷里,冰冷金属摩擦声。窗碎,枪声。最后,突发的巨大枪声。")
    shots = [
        {"description": "高对比度戏剧性打光下响起冰冷金属摩擦声",
         "end_state": ""},
        {"description": "他扣动扳机,伴随突发的巨大枪声、金属回音",
         "end_state": ""},
    ]
    assert _sound_coverage(sp, shots) == []       # 全部在场,不得误报


def test_scripted_sounds_lexicon_catches_embedded():
    """词典词直查子串:分镜句中嵌入的剧本声词必须被逐镜提取认出
    (否则音效镜被判无声直接哑掉);包含去重留超集。"""
    from maestro.pipeline.window_loop import (_scripted_sounds,
                                              set_run_ambience,
                                              set_run_sound_lexicon)
    set_run_ambience()
    set_run_sound_lexicon("雨夜,冰冷金属摩擦声。之后,枪声。")
    got = _scripted_sounds("打光下响起冰冷金属摩擦声,他扣动扳机,"
                           "伴随突发的巨大枪声")
    assert "冰冷金属摩擦声" in got                 # 句中嵌入,词典兜住
    assert "枪声" in got                           # 剧本词在超集词内
    got2 = _scripted_sounds("窗碎,突发的巨大枪声。")
    assert got2 == ["突发的巨大枪声"]              # 标点后正则得超集,
    set_run_sound_lexicon()                        # 子串去重;清词典


def test_outline_gate_budget_survives_empty_reply():
    """2026-08-06 cinegraph run3:空回复烧掉唯一 attempt,声效闸失守。
    预算拆帐后:空回复 → 坏回复预算;闸门纠错额度不受影响。"""
    import json as _json
    from maestro.pipeline.window_loop import (_write_outline,
                                              set_run_ambience,
                                              set_run_sound_lexicon)
    set_run_ambience()
    set_run_sound_lexicon()
    good = {"cast": {"男人": "static: x; dynamic: y"},
            "setting": "night alley",
            "shots": [{"description": "Shot 1: <男人>开枪,枪声。",
                       "duration_s": 5, "end_state": "<男人>静止。",
                       "variation": "small", "camera": 0, "bg": "bg_1"}],
            "music_plan": {}}
    bad = {k: v for k, v in good.items()}
    bad["shots"] = [{"description": "Shot 1: <男人>开枪。",
                     "duration_s": 5, "end_state": "<男人>静止。",
                     "variation": "small", "camera": 0, "bg": "bg_1"}]

    class _LLM:
        def __init__(self):
            self.n = 0

        def complete(self, prompt, **kw):
            self.n += 1
            if self.n == 1:
                return ""                          # 空回复(传输性)
            if self.n == 2:
                return _json.dumps(bad, ensure_ascii=False)   # 丢声词
            return _json.dumps(good, ensure_ascii=False)      # 纠错后

    llm = _LLM()
    shots, durs, ends, meta, via = _write_outline(
        llm, "雨夜。男人开枪,枪声。", [], episode_guidance={},
        max_shots=6, fallback_fn=lambda: ["fallback"],
        cast_canon={}, prompt_language="zh")
    assert llm.n == 3                # 空回复没吞掉声效闸的纠错机会
    assert "枪声" in shots[0]


# ── 三条件融合派(2026-08-07 用户令)────────────────────────────────


def test_judge_llm_thinks_beyond_mentions():
    """判官核心场景:上镜文字只提<小明>在哭,本镜只提<小红>——但剧情
    上两人全程同框 → LLM 判在场集合相同 → same=True。"""
    import json as _json

    class _LLM:
        def complete(self, prompt, **kw):
            return _json.dumps({"prev_end_cast": ["小明", "阿浪"],
                                "cur_open_cast": ["阿浪", "小明"],
                                "reason": "两人全程同车"},
                               ensure_ascii=False)
    same, why, oc = wl._judge_junction_cast(
        _LLM(), _e(end_state="<小明>在哭。"),
        _e(opening="<阿浪>在做什么。"), CAST, PORTRAITS, "zh")
    assert same and oc == ["小明", "阿浪"]


def test_judge_bad_llm_falls_back_to_set_equality():
    """坏输出 → 确定性兜底;新法:离场也算变(集合相等才 same)。"""
    class _Bad:
        def complete(self, prompt, **kw):
            return "not json"
    same, why, oc = wl._judge_junction_cast(
        _Bad(), _e(end_state="<小明>与<阿浪>对视。"),
        _e(opening="<阿浪>歪头。"), CAST, PORTRAITS, "zh")
    assert not same          # 小明离场 → 集合不等 → 变
    assert oc == ["阿浪"]


def test_judge_fallback_same_face_equivalence():
    class _Bad:
        def complete(self, prompt, **kw):
            raise RuntimeError("down")
    same, _, _ = wl._judge_junction_cast(
        _Bad(), _e(end_state="<军官甲>低声说完。"),
        _e(opening="<军官乙>转脸。"), CAST, PORTRAITS, "zh")
    assert same              # 同脸不同名按肖像判等


def test_judge_rejects_unknown_names():
    """LLM 幻觉出 cast 外的名字 → 剔除;两侧剔空 → 走兜底。"""
    import json as _json

    class _Hallu:
        def complete(self, prompt, **kw):
            return _json.dumps({"prev_end_cast": ["路人甲"],
                                "cur_open_cast": ["路人乙"],
                                "reason": "x"}, ensure_ascii=False)
    same, why, _ = wl._judge_junction_cast(
        _Hallu(), _e(end_state="<小明>静立。"),
        _e(opening="<小明>特写。"), CAST, PORTRAITS, "zh")
    assert same and "兜底" in why


def test_tail_report_parse_and_map():
    rep = wl._parse_tail_report(
        '{"camera_angle": "medium shot, camera left of <小明>, static", '
        '"character_actions": [{"who": "小明", "position": "center", '
        '"action": "stands still"}]}')
    assert rep["camera_angle"].startswith("medium")
    mapped = wl._map_tail_report(rep, {"小明": "<<<image_2>>>"}, CAST,
                                 portraits=PORTRAITS)
    assert mapped["character_actions"][0]["who"] == "<<<image_2>>>"
    assert wl._parse_tail_report("prose sentence") is None
    assert wl._parse_tail_report("") is None


def test_ref2v_manifest_pin_row_last_and_plate_dropped():
    """条件② ref2v:派生帧 = 末位 pin_frame 行(执行器专属提及);
    挂 pin 时背景板剔除(板磁铁法)。"""
    entry = SimpleNamespace(
        images=[
            {"path": "/bg/plate.png", "role": "reference",
             "source": "background", "description": "bg plate"},
            {"path": "/d/derived.png", "role": "reference",
             "source": "pin_frame", "description": "derived"}],
        keyframe_path=None)
    entry.images_by_role = lambda role: [
        im for im in entry.images if im.get("role") == role]
    rows = wl._slot_manifest("ref2v", entry, None, True,
                             portraits={"小明": "/p/xm.png"},
                             video_gen=None)
    assert rows[-1].get("source") == "pin_frame"
    assert "executor owns" in rows[-1]["content"]
    assert not any("bg plate" in str(r.get("content")) for r in rows)
    # 无 pin 时板照常在列
    entry2 = SimpleNamespace(
        images=[{"path": "/bg/plate.png", "role": "reference",
                 "source": "background", "description": "bg plate"}],
        keyframe_path=None)
    entry2.images_by_role = lambda role: [
        im for im in entry2.images if im.get("role") == role]
    rows2 = wl._slot_manifest("ref2v", entry2, None, True,
                              portraits={}, video_gen=None)
    assert any("bg plate" in str(r.get("content")) for r in rows2)


def test_derive_junction_frame_assembly(tmp_path):
    """条件②装配:refs=[上镜末帧, 开场人物肖像, 新背景板];双镜 prompt
    含两镜描述+记号语义;切后帧返回。"""
    import subprocess
    prev_video = tmp_path / "prev.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.3", str(prev_video)], check=True)
    port = tmp_path / "xm.png"
    bg = tmp_path / "bg.png"
    for p in (port, bg):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=gray:s=160x90:d=0.1", "-frames:v", "1",
             str(p)], check=True)

    class _VG:
        generate_audio = False

        def __init__(self):
            self.calls = []

        def ref_token(self, n):
            return f"<<<image_{n}>>>"

        def generate(self, prompt, duration, out_path, fps=24, seed=0,
                     reference_images=None, **kw):
            self.calls.append({"prompt": prompt,
                               "refs": [str(r) for r in
                                        (reference_images or [])]})
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error",
                 "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1",
                 "-f", "lavfi", "-i", "color=c=white:s=160x90:d=1",
                 "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
                 str(out_path)], check=True)
            return out_path

    vg = _VG()
    prev = SimpleNamespace(video_path=str(prev_video),
                           end_state="<小明>静立。", description="",
                           opening_frame="")
    entry = SimpleNamespace(shot_idx=3, end_state="",
                            opening_frame="<阿浪>落在礁石上。",
                            description="")
    got = wl._derive_junction_frame(
        vg, None, None, prev, prev, entry, ["阿浪"], CAST,
        {"阿浪": str(port)}, str(bg), tmp_path / "shot003", "zh")
    assert got is not None and got.exists()
    call = vg.calls[0]
    assert call["refs"][1] == str(port)          # 末帧后第一位 = 肖像
    assert call["refs"][2] == str(bg)            # 背景板殿后
    assert "Two shots" in call["prompt"]
    assert "<小明>静立" not in call["prompt"]     # 标记已剥
    assert "小明静立" in call["prompt"]           # 第一镜描述在场
    assert "<<<image_2>>>" in call["prompt"]      # 第二镜人物记号化


def test_derive_junction_frame_fail_returns_none(tmp_path):
    import subprocess
    prev_video = tmp_path / "prev.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.3", str(prev_video)], check=True)

    class _Dead:
        generate_audio = False

        def ref_token(self, n):
            return f"<<<image_{n}>>>"

        def generate(self, *a, **k):
            raise RuntimeError("network down")
    import maestro.cinegraph.first_frame_factory as fff
    old = fff._SPACED_WAITS_S
    fff._SPACED_WAITS_S = (0,)
    try:
        prev = SimpleNamespace(video_path=str(prev_video),
                               end_state="x", description="",
                               opening_frame="")
        entry = SimpleNamespace(shot_idx=1, opening_frame="y",
                                description="", end_state="")
        got = wl._derive_junction_frame(
            _Dead(), None, None, prev, prev, entry, [], CAST, {},
            None, tmp_path / "shot001", "zh")
        assert got is None                       # 降级条件①,不炸
    finally:
        fff._SPACED_WAITS_S = old
