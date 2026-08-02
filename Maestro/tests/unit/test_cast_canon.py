"""E 案(2026-08-02 用户批准)回归:正典描述符逐字契约。

实锤病根:enhancer 把 "dark green raincoat" 改写成 "teal raincoat",
清洗器只认原句精确匹配 → 同一角色两副长相。
契约:出场角色 static 内容词覆盖率 < 0.75 → 确定性追加正典身份子句
(恢复瘦身法则的"唯一身份子句")+ notes 记账;覆盖足够/空 prompt/
短描述符 → 原样放行。"""
import maestro.pipeline.window_loop as wl

CAST = {"the umbrella customer":
        "static: middle-aged woman, shoulder-length brown hair, dark "
        "green raincoat, black trousers, rain boots, long dark-blue "
        "umbrella; dynamic: pose, umbrella open or closed"}


def test_paraphrased_descriptor_gets_canon_appended():
    p = ("The shot opens EXACTLY on @Image1. A young woman in a teal "
         "raincoat pushes open the door; the baker turns and nods.")
    out, notes = wl._enforce_cast_canon(p, ["the umbrella customer"], CAST)
    assert notes and notes[0]["action"] == "canon_appended"
    assert notes[0]["coverage"] < 0.75
    # 追加的是无标签的 static 半句,原文保留
    assert out.startswith(p)
    assert "dark green raincoat" in out and "rain boots" in out
    assert "static:" not in out and "dynamic" not in out


def test_verbatim_descriptor_passes_untouched():
    p = ("A middle-aged woman with shoulder-length brown hair, in a dark "
         "green raincoat, black trousers and rain boots, holding her long "
         "dark-blue umbrella, steps inside.")
    out, notes = wl._enforce_cast_canon(p, ["the umbrella customer"], CAST)
    assert out == p and notes == []


def test_empty_prompt_and_absent_cast_skip():
    # 空 prompt = 走兜底模板(模板自带身份)→ 不处理
    out, notes = wl._enforce_cast_canon("", ["the umbrella customer"], CAST)
    assert out == "" and notes == []
    # 不出场的角色不检查
    out2, notes2 = wl._enforce_cast_canon("a quiet empty bakery", [], CAST)
    assert out2 == "a quiet empty bakery" and notes2 == []


def test_short_descriptor_skipped():
    cast = {"cat": "static: a cat; dynamic: pose"}
    out, notes = wl._enforce_cast_canon("a dog runs", ["cat"], cast)
    assert notes == [] and out == "a dog runs"
