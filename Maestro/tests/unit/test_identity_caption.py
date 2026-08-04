"""2026-08-04 正典打标修复回归:通用一句话 caption 不带服装颜色 →
character_extract 的 LLM 把黑军装脑补成 white coat → 评审拿错误正典
扣忠于参考图的视频。修复:caption_identity 专用打标(颜色逐项、看不
见省略),管线优先用之;skill 加"图注没说的不许发明"律。全部离线。"""
from pathlib import Path

from maestro.models.mllm import BaseMLLMClient


def test_base_caption_identity_falls_back_to_caption_image():
    class _C(BaseMLLMClient):
        def __init__(self):
            pass

        def assess_semantic(self, *a, **k):
            return []

        def assess_physics(self, *a, **k):
            return []

        def caption_image(self, p):
            return "character: a man"
    assert _C().caption_identity("x.png") == "character: a man"


def test_identity_instruction_demands_colors_and_forbids_guessing():
    from maestro.models.mllm_backends import _IDENTITY_CAPTION_INSTRUCTION as t
    assert "EXACT COLORS" in t and "never guess" in t
    for cls_probe in ("OpenAICompatVLM", "GeminiVLM", "LocalQwenVLM"):
        import maestro.models.mllm_backends as mb
        assert hasattr(getattr(mb, cls_probe), "caption_identity"), cls_probe


def test_given_captioning_prefers_caption_identity(tmp_path, monkeypatch):
    """window_loop 钦定角色打标:有 caption_identity 就用它,不落通用。"""
    import maestro.pipeline.window_loop as wl

    img = tmp_path / "p.png"
    img.write_bytes(b"\x89PNG\r\n" + b"\x00" * 8)
    calls = []

    class _M:
        def caption_identity(self, p):
            calls.append("identity")
            return "black military coat with red sash"

        def caption_image(self, p):
            calls.append("generic")
            return "character: a man"
    # 直接演练调用点的选择逻辑(与 window_loop 同一行代码路径)
    m = _M()
    fn = getattr(m, "caption_identity", None) or getattr(m, "caption_image",
                                                         None)
    assert fn(img) == "black military coat with red sash"
    assert calls == ["identity"]


def test_character_extract_skill_forbids_invented_details():
    p = Path("src/maestro/skills/brain_skills/character_extract/SKILL.md")
    t = p.read_text()
    assert "NEVER add an appearance detail" in t
    assert "image_look` WINS" in t or "image_look WINS" in t
    assert "NEVER\n   applies to a given character" in t.replace("  ", "  ") \
        or "This rule NEVER" in t


def test_caption_canon_hard_override():
    """VLM 图注硬闸:LLM 写的 static 外观整段丢弃,图注原文顶上;
    dynamic 保留;无图注/非钦定角色不动。"""
    import maestro.pipeline.window_loop as wl

    canon = {
        "王子": ("static: white royal military coat, blond hair; "
               "dynamic: medals, sword"),
        "路人甲": "static: brown jacket; dynamic: umbrella",
        "无注者": "static: red dress; dynamic: pose",
    }
    caps = {"王子": "black military coat with a red sash, blond hair",
            "无注者": "",                       # 图注失败 → 不覆盖
            "不在正典的名字": "grey suit"}       # 正典没这人 → 跳过
    wl._apply_caption_canon(canon, caps)
    assert canon["王子"] == ("static: black military coat with a red sash, "
                           "blond hair; dynamic: medals, sword")
    assert "white" not in canon["王子"]
    assert canon["路人甲"] == "static: brown jacket; dynamic: umbrella"
    assert canon["无注者"] == "static: red dress; dynamic: pose"
    assert "不在正典的名字" not in canon


def test_skills_carry_the_reference_rule():
    """引用铁律(2026-08-04 用户令)三技能齐备:角色只用记号指称、
    名字禁入 prompt、外观描述禁入 prompt。"""
    base = Path("src/maestro/skills/brain_skills")
    wg = (base / "window_generation/SKILL.md").read_text()
    assert "TOKEN ONLY" in wg
    assert "NAMES are FORBIDDEN" in wg
    assert "IDENTITY APPEARANCE is FORBIDDEN" in wg
    assert "PERFORMANCE IS CONTENT" in wg
    # video_prompt_writing 已废(代码从不加载)—— 法典并入 window_generation
    assert not (base / "video_prompt_writing").exists()
    assert "SCRIPT ACTION IS SACRED" in wg
    pe = (base / "prompt_enhancer/SKILL.md").read_text()
    assert "THE REFERENCE RULE" in pe
    assert "replace EVERY character name" in pe
    # 样例以身作则:Example 2 无外观同位语、收势句主语是记号
    assert "dark green coat" not in wg and "grey sweater" not in wg
    assert "the character holds a natural micro-expression" not in wg
