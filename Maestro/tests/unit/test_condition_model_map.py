"""映射表(docs/CONDITION_MODEL_MAP.md)回归:tiv2v_window 永远走 t2v、
i2v×reference_videos 被后端拒绝、每次调用落 call_log、duration=None 不进
payload。CPU-only,无网络(requests 全部打桩)。"""
import json
from pathlib import Path

import pytest

from maestro.models.video_gen_backends import WaveSpeedClient
from maestro.pipeline.window_loop import _generate_with_condition
from maestro.types import ShotSpec


class _RecordingGen:
    """记录 generate/multi_image_to_video 收到的 kwargs 的桩后端。"""

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_video", "ref_images", "multi_i2v"}

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        self.calls.append({"fn": "generate", "prompt": prompt,
                           "duration": duration, "first_frame": first_frame,
                           "reference_images": reference_images,
                           "reference_video": reference_video})
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("MOCK VIDEO")
        return p


def _Entry(images=None):
    """真实 ShotEntry(_entry_images 依赖 images_by_role,桩会漏角色逻辑)。"""
    from maestro.memory.storyboard import ShotEntry

    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="the apple falls")
    e.images = list(images or [])
    return e


class _Prev:
    def __init__(self, path):
        self.video_path = path


@pytest.fixture()
def prev_clip(tmp_path):
    p = tmp_path / "prev.mp4"
    p.write_text("MOCK PREV")
    return _Prev(p)


def _spec(duration=6.0):
    return ShotSpec(shot_idx=1, duration=duration, prompt="the apple falls")


def test_tiv2v_window_routes_t2v_with_keyframe(tmp_path, prev_clip):
    """映射表 #8:有本镜图时绝不走 first_frame(i2v)—— 图走 reference_images
    软锚,尾段走 reference_video,同一次 t2v 调用。"""
    kf = tmp_path / "kf.png"
    kf.write_bytes(b"\x89PNG\r\n")
    gen = _RecordingGen()
    entry = _Entry(images=[{"path": str(kf), "role": "first_frame"}])
    _, cond = _generate_with_condition(
        "tiv2v_window", entry, prev_clip, _spec(), gen, tmp_path / "s1",
        seed=0, fps=8, window_tail_s=2.0)
    call = gen.calls[-1]
    assert call["first_frame"] is None                    # 绝不切 i2v 端点
    assert call["reference_video"] is not None            # 尾段视频在场
    assert [str(p) for p in call["reference_images"]] == [str(kf)]
    assert call["duration"] == 6.0                        # brain 时长原样传
    assert cond["strategy"] == "tiv2v_window"
    assert cond["reference_images"] == [str(kf)]
    assert cond["anchoring"] == "soft_t2v_video_refs"
    assert "@Video1" in call["prompt"] and "@Image1" in call["prompt"]


def test_tiv2v_window_without_image_still_t2v(tmp_path, prev_clip):
    gen = _RecordingGen()
    _, cond = _generate_with_condition(
        "tiv2v_window", _Entry(), prev_clip, _spec(duration=None), gen,
        tmp_path / "s1", seed=0, fps=8, window_tail_s=2.0)
    call = gen.calls[-1]
    assert call["first_frame"] is None
    assert call["reference_images"] is None
    assert call["reference_video"] is not None
    assert call["duration"] is None            # brain 没规划 → None 透传
    assert cond.get("reference_images") is None
    assert "@Video1" in call["prompt"]


def test_backend_rejects_first_frame_plus_reference_video(tmp_path):
    """i2v schema 没有 reference_videos —— 未验证组合必须在任何上传前拒绝。"""
    ws = WaveSpeedClient(config={"api_key": "k"})
    ff = tmp_path / "f.png"
    ff.write_bytes(b"\x89PNG\r\n")
    rv = tmp_path / "r.mp4"
    rv.write_bytes(b"\x00" * 8)
    with pytest.raises(RuntimeError, match="not a verified"):
        ws.generate("p", 5, tmp_path / "o.mp4", first_frame=ff,
                    reference_video=rv)


def _fake_requests(monkeypatch, submit_payloads):
    """打桩 requests:capture submit payload,秒回 completed + 假视频字节。"""
    import maestro.models.video_gen_backends as vgb

    class _Resp:
        def __init__(self, code, data=None, content=b""):
            self.status_code = code
            self._data = data or {}
            self.content = content
            self.text = json.dumps(self._data)

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    class _Req:
        @staticmethod
        def post(url, json=None, headers=None, timeout=0, **kw):
            submit_payloads.append({"url": url, "payload": json})
            return _Resp(200, {"data": {"id": "t1"}})

        @staticmethod
        def get(url, headers=None, timeout=0, **kw):
            if url.endswith("/result"):
                return _Resp(200, {"data": {"status": "completed",
                                            "outputs": ["http://x/v.mp4"]}})
            return _Resp(200, {}, content=b"FAKEVIDEO")

    monkeypatch.setattr("requests.post", _Req.post, raising=False)
    monkeypatch.setattr("requests.get", _Req.get, raising=False)
    import sys
    monkeypatch.setitem(sys.modules, "requests", _Req)
    return _Req


def test_call_log_records_model_and_params(tmp_path, monkeypatch):
    """任务 0:每次调用 → JSONL {event, model, payload};duration=None 时
    payload 里没有 duration 字段(API 默认)。"""
    sent = []
    _fake_requests(monkeypatch, sent)
    logf = tmp_path / "calls.jsonl"
    ws = WaveSpeedClient(config={"api_key": "k", "call_log": str(logf)})
    ws.generate("a red apple rolls", None, tmp_path / "o.mp4")

    assert sent[0]["url"].endswith("bytedance/seedance-2.0/text-to-video")
    assert "duration" not in sent[0]["payload"]           # None → 不传字段
    lines = [json.loads(x) for x in logf.read_text().splitlines()]
    events = [x["event"] for x in lines]
    assert events == ["submit", "completed"]
    assert all(x["model"] == "bytedance/seedance-2.0/text-to-video"
               for x in lines)
    assert lines[0]["payload"]["prompt"] == "a red apple rolls"

    ws.generate("p2", 7, tmp_path / "o2.mp4")             # brain 规划 7s
    assert sent[1]["payload"]["duration"] == 7            # 4-10 ⊂ [4,15] 原样传
    assert len(logf.read_text().splitlines()) == 4


def test_brain_log_records_raw_output(tmp_path, monkeypatch):
    """debug 日志:brain 选策略时,原始输出 + 解析结果 + 菜单逐行落 JSONL
    (episode/fallback 路径也留痕,via 如实)。"""
    from maestro.logging_utils import set_brain_log
    from maestro.pipeline.window_loop import _decide

    logf = tmp_path / "brain.jsonl"
    set_brain_log(logf)
    try:
        menu = [{"name": "tiv2v_window", "description": "d"},
                {"name": "t2v", "description": "d"}]

        class _Brain:
            def complete(self, prompt, **kw):
                return json.dumps({
                    "strategy": "tiv2v_window",
                    "reason": "the scene continues",
                    "video_prompt": "Continue @Video1's motion seamlessly"})

        d = _decide(_Brain(), "generation-condition", menu,
                    {"shot": {"label": "scene 1 shot 2"}},
                    replay_hint=None, priority=["t2v"])
        assert d["strategy"] == "tiv2v_window" and d["via"] == "llm"

        # LLM 回复不可用 → fallback 也要留痕
        class _Garbage:
            def complete(self, prompt, **kw):
                return "sure, maybe use something nice?"

        d2 = _decide(_Garbage(), "generation-condition", menu,
                     {"shot": {"label": "scene 1 shot 3"}},
                     replay_hint=None, priority=["t2v"])
        assert d2["via"] == "fallback"

        lines = [json.loads(x) for x in logf.read_text().splitlines()]
        assert len(lines) == 3            # llm 决策 1 条 + (垃圾 raw + fallback)
        ok = lines[0]
        assert ok["stage"] == "window/generation-condition"
        assert ok["label"] == "scene 1 shot 2"
        assert "tiv2v_window" in ok["raw"]              # 原始输出全量保留
        assert ok["parsed"]["strategy"] == "tiv2v_window"
        assert "tiv2v_window" in ok["menu"]
        # 装载证据(2026-07-15 用户令):每次 brain 调用必须自证技能进了
        # prompt —— window_generation 技能真实存在,chars 必须大于 1000。
        assert ok["skill"] == "window_generation"
        assert ok["skill_loaded"] is True and ok["skill_chars"] > 1000
        bad, fb = lines[1], lines[2]
        assert bad["usable"] is False and "maybe" in bad["raw"]
        assert fb["parsed"]["via"] == "fallback"
    finally:
        set_brain_log(None)               # 全局状态复位,别污染其他测试


def test_call_log_keeps_prompt_readable(tmp_path, monkeypatch):
    """审计教训(attempt_1):prompt 绝不能被缩写成 '<446 chars>' —— 语义
    文本字段全文保留(≤4000),真正缩写的是非语义长串。"""
    sent = []
    _fake_requests(monkeypatch, sent)
    logf = tmp_path / "calls.jsonl"
    ws = WaveSpeedClient(config={"api_key": "k", "call_log": str(logf)})
    long_prompt = "a glossy red apple rolls across the counter. " * 12  # >200ch
    ws.generate(long_prompt, 5, tmp_path / "o.mp4")
    rec = json.loads(logf.read_text().splitlines()[0])
    assert rec["payload"]["prompt"] == long_prompt          # 全文,无缩写
    blob = {"prompt": "p", "image": "x" * 999}              # 非语义长串仍缩写
    s = ws._summarize_payload(blob)
    assert s["image"].endswith("<999 chars>") and s["prompt"] == "p"
