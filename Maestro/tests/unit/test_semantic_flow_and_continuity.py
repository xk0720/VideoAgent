"""2026-07-15 连贯性大修回归:问题一(语义跟图走/输入日志)+ 问题二
(end_state 交接棒/接点实况/评审衔接)。CPU-only,无网络。"""
import json
from pathlib import Path

from maestro.memory.storyboard import ShotEntry
from maestro.pipeline.window_loop import (
    _execute_image_plan,
    _generate_with_condition,
    _mention,
)
from maestro.types import AssetMemory, Identity, ShotSpec


def _entry(images=None, description="the cat jumps onto the windowsill"):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description=description)
    e.images = list(images or [])
    return e


def test_asset_image_ledger_keeps_real_semantics_and_query(tmp_path):
    """裁决 1.2:检索命中后,台账 description = 素材真实标签;检索词另存
    retrieval_query 供审计。"""
    cat = tmp_path / "cat.png"
    cat.write_bytes(b"\x89PNG\r\n")
    mem = AssetMemory(identity_anchors={
        "cat": Identity(identity_id="cat", name="cat", source=str(cat),
                        description="an orange tabby cat curled on a sofa")})
    plan, images, degraded = _execute_image_plan(
        {"strategy": "single_first_frame",
         "images": [{"source": "asset_image",
                     "description": "the user's cat photo"}]},
        _entry(), video_gen=None, asset_memory=mem, retrieval=None,
        out_dir=tmp_path / "kf")
    assert plan == "single_first_frame" and not degraded
    im = images[0]
    assert im["description"] == "an orange tabby cat curled on a sofa"
    assert im["retrieval_query"] == "the user's cat photo"


def test_fallback_prompt_carries_image_content(tmp_path):
    """裁决 1.2:兜底模板不写空话 —— @ImageN 带实况语义。"""
    class _Gen:
        def __init__(self):
            self.calls = []

        def capabilities(self):
            return {"t2v", "i2v", "ref_images", "ref_video"}

        def generate(self, prompt, duration, out_path, fps=8,
                     first_frame=None, reference_images=None, seed=0,
                     reference_video=None):
            self.calls.append(prompt)
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("MOCK")
            return p

    kf = tmp_path / "cat.png"
    kf.write_bytes(b"\x89PNG\r\n")
    entry = _entry(images=[{"path": str(kf), "role": "reference",
                            "description": "an orange tabby cat"}])
    gen = _Gen()
    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="the cat jumps up")
    _generate_with_condition("t2v_own_refs", entry, None, spec, gen,
                             tmp_path / "s", seed=0, fps=8, window_tail_s=2.0)
    assert "@Image1 shows: an orange tabby cat" in gen.calls[-1]
    # helper 语义缺失时诚实退化,绝不编内容
    assert "planned image" in _mention(_entry(), tmp_path / "nope.png", 1)


def test_brain_log_records_input_context(tmp_path):
    """裁决 1.3:brain_calls.jsonl 每条带 context(喂给 brain 的输入)。"""
    from maestro.logging_utils import set_brain_log
    from maestro.pipeline.window_loop import _decide

    logf = tmp_path / "brain.jsonl"
    set_brain_log(logf)
    try:
        class _Brain:
            def complete(self, prompt, **kw):
                return json.dumps({"strategy": "t2v", "reason": "r"})

        _decide(_Brain(), "generation-condition",
                [{"name": "t2v", "description": "d"}],
                {"shot": {"label": "scene 1 shot 1",
                          "images": [{"description": "an orange tabby cat"}]}},
                replay_hint=None, priority=["t2v"])
        rec = json.loads(logf.read_text().splitlines()[0])
        assert rec["context"]["shot"]["images"][0]["description"] \
            == "an orange tabby cat"
    finally:
        set_brain_log(None)


def test_junction_state_honest_chain_and_cache(tmp_path, monkeypatch):
    """需求 ②:无上镜/无 VLM/尾帧抽不出 → ""(不编);正常路径出一句实况
    并按 (帧文件, mtime) 缓存 —— 一镜只调一次 VLM。"""
    import maestro.pipeline.window_loop as wl

    class _Prev:
        video_path = str(tmp_path / "prev.mp4")
        end_state = "the apple is still rolling toward the window"

    class _VLM:
        def __init__(self):
            self.calls = 0

        def describe_junction(self, path):
            self.calls += 1
            return "the apple is at rest at the center of the floor"

    wl._JUNCTION_CACHE.clear()
    assert wl._junction_state(None, _Prev(), tmp_path) == ""      # 无 VLM
    assert wl._junction_state(_VLM(), None, tmp_path) == ""       # 无上镜
    # 尾帧抽不出(mock 视频不可解码)→ ""
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    assert wl._junction_state(_VLM(), _Prev(), tmp_path) == ""
    # 正常:出实况 + 缓存生效
    frame = tmp_path / "prev_last.png"
    frame.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: frame)
    vlm = _VLM()
    got = wl._junction_state(vlm, _Prev(), tmp_path)
    assert "at rest" in got
    assert wl._junction_state(vlm, _Prev(), tmp_path) == got
    assert vlm.calls == 1                                         # 缓存命中
    wl._JUNCTION_CACHE.clear()


def test_conditions_include_state_facts(tmp_path):
    """需求 ②:润色 agent 的条件清单带三类状态事实(实况/上镜剧本
    end_state/本镜 required end_state)。"""
    from maestro.pipeline.window_loop import _conditions_for_prompt

    e = _entry()
    e.end_state = "the cat is curled up asleep on the windowsill"

    class _Prev:
        video_path = tmp_path / "prev.mp4"
        end_state = "the cat is mid-leap toward the windowsill"

    conds = _conditions_for_prompt(
        "ti2v_prev_last", e, _Prev(), False,
        junction="the cat is airborne above the sofa, moving right")
    roles = {c["role"]: c["description"] for c in conds
             if c["kind"] == "state"}
    assert "airborne" in roles["opening_state_actual"]
    assert "mid-leap" in roles["previous_end_state_script"]
    assert "asleep" in roles["required_end_state"]


def test_local_qwen_registry_and_honest_review_silence():
    """qwen-local 注册可解析(构造零加载);评审职责不归它 —— assess 返回
    [](警告),绝不伪造判定。"""
    from maestro.models.mllm import build_mllm
    from maestro.models.mllm_backends import LocalQwenVLM

    vlm = build_mllm({"name": "qwen-local", "model": "Qwen/Qwen2.5-VL-7B-Instruct"})
    assert isinstance(vlm, LocalQwenVLM)
    assert vlm._model is None                       # 惰性:构造不加载权重
    assert vlm.assess_semantic(None, None) == []
    assert vlm.assess_physics(None, None, 24) == []


def test_review_instruction_carries_junction_checks(tmp_path, monkeypatch):
    """需求 ④:clip.conditioning 带 end_state/junction_prev_actual 时,
    评审指令必须包含"开头延续上一镜实况 + 结尾落在剧本 end_state"两条
    要求(各出一条 check)。"""
    import json as _json

    from maestro.models.mllm_backends import GeminiVLM
    from maestro.types import CandidateClip

    vlm = GeminiVLM("gemini", {"api_key": "k"})
    captured = []

    def _fake_generate(parts):
        captured.append(parts)
        return _json.dumps({"checks": [], "issues": [], "summary": "ok"})
    monkeypatch.setattr(vlm, "_generate", _fake_generate)

    v = tmp_path / "shot.mp4"
    v.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    clip = CandidateClip(shot_idx=2, video_path=v)
    clip.conditioning = {
        "video_prompt": "the apple rolls on",
        "images": [],
        "reference_video": None,
        "end_state": "the apple is still rolling toward the window",
        "junction_prev_actual": "the apple is at rest at the center "
                                "of the floor",
    }
    spec = ShotSpec(shot_idx=2, duration=5.0, prompt="the apple rolls on")
    vlm.review_shot(clip, spec)
    text = " ".join(p.get("text", "") for p in captured[0] if "text" in p)
    assert "ACTUALLY ended in this state" in text
    assert "at rest at the center" in text            # 上一镜实况原文在场
    assert "requires this shot to END" in text
    assert "still rolling toward the window" in text  # 剧本 end_state 在场
