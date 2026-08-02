"""角色官方肖像回归(2026-07-31 用户批准:单视图视觉锚 + 跨片库"现在做"
+ episode 只作 guidance):
① CharacterLibrary:同名+描述符重叠才命中,跨实例持久;
② _ensure_cast_portraits 三来源优先级:用户素材 > 跨片库 > t2i(入库);
③ 槽位清单:肖像行编号与装配顺序严格一致;
④ 肖像进 conditioning(评审可见)由 window_loop 主循环负责(集成层)。
全部离线。"""

from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.memory.character_library import CharacterLibrary
from maestro.memory.storyboard import ShotEntry, StoryboardMemory
from maestro.types import AssetMemory, Identity

DESC = ("static: small orange-and-white shorthair cat with amber eyes "
        "and white paws; dynamic: pose")


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 64)
    return path


# ── ① 跨片角色库 ─────────────────────────────────────────────────────

def test_library_hit_requires_name_and_descriptor(tmp_path):
    lib = CharacterLibrary(tmp_path / "lib")
    src = _png(tmp_path / "cat.png")
    assert lib.add("the cat", "orange and white shorthair amber eyes", src)
    # 同名 + 描述符重叠 → 命中;跨实例持久
    lib2 = CharacterLibrary(tmp_path / "lib")
    hit = lib2.lookup("The Cat", "orange white shorthair cat amber eyes")
    assert hit is not None and Path(hit).exists()
    # 同名不同长相 → 不命中(另一部片里另一只猫也叫 the cat)
    assert lib2.lookup("the cat", "black longhair green eyes bell") is None
    # 不同名 → 不命中
    assert lib2.lookup("the dog", "orange and white shorthair") is None


# ── ② 肖像阶段三来源 ─────────────────────────────────────────────────

class _T2I:
    def __init__(self):
        self.prompts = []

    def text_to_image(self, prompt, out, seed=0):
        self.prompts.append(prompt)
        return _png(Path(out))


def _board(tmp_path, cast):
    sb = StoryboardMemory.from_outline(["shot 1: x"], path=tmp_path / "sb.json")
    sb.cast = dict(cast)
    sb.setting = "a warm daylit living room with a wooden floor"
    return sb


def test_portrait_from_t2i_registers_library_and_assets(tmp_path):
    sb = _board(tmp_path, {"the cat": DESC})
    am = AssetMemory()
    lib = CharacterLibrary(tmp_path / "lib")
    gen = _T2I()
    notes = wl._ensure_cast_portraits(sb, am, gen, tmp_path, library=lib)
    assert notes[0]["via"] == "t2i"
    # prompt 内嵌 static 半句(无标签)+ 全片 setting(不学 ViMax 白底)
    p = gen.prompts[0]
    assert "orange-and-white shorthair cat" in p
    assert "static:" not in p and "living room" in p
    # 三处登记:台账(持久化)/ 素材库 / 跨片库
    assert "the cat" in sb.portraits
    back = StoryboardMemory.load(tmp_path / "sb.json")
    assert back.portraits["the cat"] == sb.portraits["the cat"]
    assert "portrait:the cat" in am.identity_anchors
    assert lib.lookup("the cat", DESC) is not None


def test_portrait_prefers_user_asset_then_library(tmp_path):
    # 用户素材命中 → 不生成
    sb = _board(tmp_path, {"the cat": DESC})
    am = AssetMemory()
    photo = _png(tmp_path / "user_cat.png")
    am.identity_anchors["cat"] = Identity(
        identity_id="cat", name="cat", source=str(photo),
        description="an orange and white shorthair cat with amber eyes "
                    "sleeping on a windowsill")
    gen = _T2I()
    notes = wl._ensure_cast_portraits(sb, am, gen, tmp_path, library=None)
    assert notes[0]["via"] == "user_asset" and not gen.prompts
    assert sb.portraits["the cat"] == str(photo)

    # 跨片库命中 → 不生成
    sb2 = _board(tmp_path / "b", {"the cat": DESC})
    lib = CharacterLibrary(tmp_path / "lib2")
    lib.add("the cat", "orange-and-white shorthair cat amber eyes white "
                       "paws", _png(tmp_path / "old.png"))
    gen2 = _T2I()
    notes2 = wl._ensure_cast_portraits(sb2, AssetMemory(), gen2,
                                       tmp_path / "b", library=lib)
    assert notes2[0]["via"] == "library" and not gen2.prompts


def test_portrait_failure_is_honest(tmp_path):
    sb = _board(tmp_path, {"the cat": DESC})

    class _Boom:
        def text_to_image(self, prompt, out, seed=0):
            raise RuntimeError("t2i down")
    notes = wl._ensure_cast_portraits(sb, AssetMemory(), _Boom(), tmp_path)
    assert notes[0]["via"] == "none"
    assert "the cat" not in sb.portraits          # 不放占位图冒充


# ── ③ 槽位编号与装配一致 ─────────────────────────────────────────────

def test_slot_manifest_appends_portrait_rows_in_order(tmp_path):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="shot 2: <the cat> trots")
    ref = _png(tmp_path / "ref.png")
    e.images = [{"path": str(ref), "role": "reference",
                 "description": "a planned ref"}]

    class _Prev:
        video_path = str(tmp_path / "prev.mp4")
        keyframe_path = None
        images = []

    portraits = {"the cat": str(_png(tmp_path / "cat_portrait.png"))}
    rows = wl._slot_manifest("ti2v_prev_plus_keyframe", e, _Prev(),
                             use_prev_tail=True, portraits=portraits)
    slots = {r["slot"]: r["content"] for r in rows}
    # @Image1=上镜尾帧, @Image2=自有图, @Image3=肖像 —— 与装配顺序一致
    assert "previous shot's final frame" in slots["@Image1"]
    assert "planned ref" in slots["@Image2"]      # 语义跟着图走(裁决 1.2)
    # B 案(2026-08-02):防拷贝子句内置在槽位语义行里(单源,四写手继承)
    assert "official portrait of the cat" in slots["@Image3"]
    assert "NEVER copy its pose" in slots["@Image3"]

    rows2 = wl._slot_manifest("t2v_own_refs", e, None,
                              use_prev_tail=False, portraits=portraits)
    slots2 = {r["slot"]: r["content"] for r in rows2}
    assert "planned ref" in slots2["@Image1"]
    assert "official portrait" in slots2["@Image2"]
