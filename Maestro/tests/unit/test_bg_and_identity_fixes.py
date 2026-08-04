"""2026-08-04 三大实跑事故修复回归:
① 引用记号归一:<<image_1>>/<image_2> 等变体归一为 <<<image_N>>>,
   不再"漏检+补句双份";
② 对白说话人:{speaker,line} 契约解析;口型子句对准真实说话人;
   旧字符串形态兼容;
③ 背景资产:bg 字段解析进台账;登记表持久化;注入强绑定行(t2i 与
   实拍帧两种措辞);首镜出片后升级为实拍帧;
④ A1 禁钉律:出场者带官方肖像 → t2i 首帧槽确定性拦截(生成前,
   不花钱),asset 来源首帧放行。全部离线。"""
import json
from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.memory.storyboard import ShotEntry, StoryboardMemory
from maestro.pipeline.ref_slots import normalize_ref_tokens, \
    validate_references


def _png(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n" + b"\x00" * 16)
    return p


# ── ① 记号归一 ──────────────────────────────────────────────────────

def test_token_variants_normalized_and_counted():
    assert normalize_ref_tokens("<<image_1>> and <Image 2> and "
                                "<<<<video_3>>>>") \
        == "<<<image_1>>> and <<<image_2>>> and <<<video_3>>>"
    slots = [{"slot": "<<<image_1>>>", "referenceable": True,
              "content": "c1"},
             {"slot": "<<<image_2>>>", "referenceable": True,
              "content": "c2"}]
    fixed, audit = validate_references(
        "<<image_1>> fixes identity; <<image_2>> holds the hall.", slots)
    assert audit["ok"]
    assert audit["appended"] == []          # 变体算"已提及",不再补双份
    assert fixed.count("<<<image_1>>>") == 1   # 归一后只有一份规范记号


# ── ② 对白说话人 ────────────────────────────────────────────────────

def test_dialogue_speaker_contract_parsed():
    reply = json.dumps({
        "cast": {"王子": "static: tall; dynamic: p",
                 "安娜": "static: slender; dynamic: p"},
        "setting": "a hall",
        "shots": [{"description": "Shot 1: <安娜> nears <王子>",
                   "duration_s": 5,
                   "end_state": "they face; camera: static",
                   "dialogue": {"speaker": "王子", "line": "你不配"},
                   "bg": "bg_1"}]})

    class _LLM:
        def complete(self, prompt, **k):
            return reply
    _o, _d, _e, meta, via = wl._write_outline(
        _LLM(), "script", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["shot 1: x"])
    assert via == "llm"
    assert meta["dialogues"] == ["你不配"]
    assert meta["dialogue_speakers"] == ["王子"]
    assert meta["bgs"] == ["bg_1"]


def test_with_dialogue_uses_speaker_not_first_marker():
    cast = {"安娜": "static: a; dynamic: b", "王子": "static: c; dynamic: d"}
    e = ShotEntry(shot_idx=0, scene_idx=1, label="scene 1 shot 1",
                  description="shot 1: <安娜> approaches <王子>")
    e.dialogue = "你不配"
    e.dialogue_speaker = "王子"
    out = wl._with_dialogue("base prompt", e, cast)
    assert '王子 says: "你不配"' in out          # 不再猜第一个出场者(安娜)
    # speaker 缺失/不在 cast → 兼容旧行为(第一个出场者)
    e.dialogue_speaker = ""
    out2 = wl._with_dialogue("base prompt", e, cast)
    assert '安娜 says:' in out2


# ── ③ 背景资产 ──────────────────────────────────────────────────────

def test_background_registry_persists(tmp_path):
    sb = StoryboardMemory.from_outline(["shot 1: x"],
                                       path=tmp_path / "sb.json")
    sb.backgrounds["bg_1"] = {"path": str(tmp_path / "b.png"),
                              "src": "t2i"}
    sb.entries[0].bg_id = "bg_1"
    sb._save()
    back = StoryboardMemory.load(tmp_path / "sb.json")
    assert back.backgrounds["bg_1"]["src"] == "t2i"
    assert back.entries[0].bg_id == "bg_1"


# ── ④ A1 禁钉律 ─────────────────────────────────────────────────────

def _entry():
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="shot 2: <安娜> smiles")
    return e


def test_t2i_first_frame_blocked_for_portrait_cast(tmp_path):
    decision = {"strategy": "single_first_frame",
                "images": [{"source": "t2i", "description": "opening"}]}

    class _T2I:
        def __init__(self):
            self.calls = 0

        def text_to_image(self, prompt, out, seed=0):
            self.calls += 1
            return _png(Path(out))

        def capabilities(self):
            return {"t2i"}
    gen = _T2I()
    plan, imgs, degraded = wl._execute_image_plan(
        decision, _entry(), gen, None, None, tmp_path / "kf",
        has_portrait_cast=True)
    assert gen.calls == 0                      # 生成前拦截,一分钱不花
    assert plan == "none" and degraded == "single_first_frame"
    # 无肖像出场者 → 照常放行
    plan2, imgs2, _ = wl._execute_image_plan(
        decision, _entry(), gen, None, None, tmp_path / "kf2",
        has_portrait_cast=False)
    assert gen.calls == 1 and plan2 == "single_first_frame"
    # asset 来源首帧(真实用户像素)不受禁令约束
    am_img = _png(tmp_path / "user.png")
    from maestro.types import AssetMemory, Identity
    am = AssetMemory()
    am.identity_anchors["u"] = Identity(identity_id="u", name="u",
                                        source=str(am_img),
                                        description="opening smiles frame")
    d3 = {"strategy": "single_first_frame",
          "images": [{"source": "asset_image",
                      "description": "opening smiles frame"}]}
    plan3, imgs3, _ = wl._execute_image_plan(
        d3, _entry(), gen, am, None, tmp_path / "kf3",
        has_portrait_cast=True)
    assert plan3 == "single_first_frame" and imgs3


def test_t2i_person_reference_blocked_for_portrait_cast(tmp_path):
    """A1 律扩展(run10 实跑):出场者全有官方肖像时,t2i 重画【人】当
    参考 = 第二个身份锚,生成前拦截;道具/场景类 t2i 参考放行。"""
    class _T2I:
        def __init__(self):
            self.calls = 0

        def text_to_image(self, prompt, out, seed=0):
            self.calls += 1
            return _png(Path(out))

        def capabilities(self):
            return {"t2i"}

    gen = _T2I()
    person = {"strategy": "single_reference",
              "images": [{"source": "t2i",
                          "description": "The character has an oval face "
                                         "and a purple velvet gown"}]}
    plan, imgs, degraded = wl._execute_image_plan(
        person, _entry(), gen, None, None, tmp_path / "k1",
        has_portrait_cast=True)
    assert gen.calls == 0 and plan == "none"
    # 道具类参考不受拦
    prop = {"strategy": "single_reference",
            "images": [{"source": "t2i",
                        "description": "an ornate black lace folding fan "
                                       "with gold sticks on dark velvet"}]}
    plan2, imgs2, _ = wl._execute_image_plan(
        prop, _entry(), gen, None, None, tmp_path / "k2",
        has_portrait_cast=True)
    assert gen.calls == 1 and plan2 == "single_reference"
