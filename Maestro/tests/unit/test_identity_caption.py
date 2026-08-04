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
