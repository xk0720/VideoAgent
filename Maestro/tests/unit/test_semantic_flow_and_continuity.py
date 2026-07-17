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


def test_video_asset_captioning_chain(tmp_path, monkeypatch):
    """2026-07-16 裁决:用户大概率只给一个路径。视频素材:VLM 看中间帧
    补 caption(检索键 + 剧本可见语义);无 VLM → 文件名兜底 + 大声警告;
    有用户描述 → 一个字不动。"""
    import maestro.pipeline.window_loop as wl
    from maestro.pipeline.window_loop import ensure_asset_descriptions
    from maestro.types import AssetMemory, Shot

    vid = tmp_path / "IMG_4032.mp4"
    vid.write_bytes(b"\x00" * 32)
    frame = tmp_path / "mid.png"
    frame.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(wl, "extract_frame", lambda v, i, o: frame)

    class _VLM:
        def caption_image(self, path):
            return "character: a man walking along a wooden boardwalk"

    # VLM 路径:caption 写回并标注来源
    mem = AssetMemory(video_shots={"vid0": Shot(
        shot_id="vid0", source_video=str(vid), start_time=0, end_time=4)})
    n = ensure_asset_descriptions(mem, _VLM(), cache_dir=tmp_path / "labels")
    assert n == 1
    assert mem.video_shots["vid0"].caption == (
        "character: a man walking along a wooden boardwalk "
        "(from the user's video clip)")
    # 无 VLM → 文件名兜底(caption 是检索键,不能留空)
    mem2 = AssetMemory(video_shots={"vid0": Shot(
        shot_id="vid0", source_video=str(vid), start_time=0, end_time=4)})
    ensure_asset_descriptions(mem2, None, cache_dir=tmp_path / "labels")
    assert mem2.video_shots["vid0"].caption == "IMG 4032"
    # 用户描述在 → 不覆盖
    mem3 = AssetMemory(video_shots={"vid0": Shot(
        shot_id="vid0", source_video=str(vid), start_time=0, end_time=4,
        caption="my dog running on the beach")})
    assert ensure_asset_descriptions(mem3, _VLM(),
                                     cache_dir=tmp_path / "labels") == 0
    assert mem3.video_shots["vid0"].caption == "my dog running on the beach"


def test_media_catalog_shows_videos_but_retrieval_stays_images(tmp_path):
    """剧本/图计划的目录含视频条目(kind=video,带语义);图片检索
    (_retrieve_asset_image)绝不返回视频文件。"""
    from maestro.pipeline.window_loop import (
        _media_catalog,
        _retrieve_asset_image,
    )
    from maestro.types import AssetMemory, Identity, Shot

    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n")
    vid = tmp_path / "walk.mp4"
    vid.write_bytes(b"\x00" * 8)
    mem = AssetMemory(
        identity_anchors={"cat": Identity(
            identity_id="cat", name="cat", source=str(img),
            description="an orange tabby cat")},
        video_shots={"vid0": Shot(
            shot_id="vid0", source_video=str(vid), start_time=0, end_time=4,
            caption="a man walking along a boardwalk")})
    cat = _media_catalog(mem)
    kinds = {c["kind"] for c in cat}
    assert kinds == {"identity", "video"}
    vrow = next(c for c in cat if c["kind"] == "video")
    assert vrow["desc"] == "a man walking along a boardwalk"
    # 图片检索:哪怕检索词更像视频内容,也只在图片里选
    path, _ = _retrieve_asset_image("a man walking along a boardwalk", mem)
    assert path == img


def test_source_videos_ride_t2v_reference_channel(tmp_path, monkeypatch):
    """2026-07-17 G-1:上镜尾帧+素材 → t2v 路线;用户源视频挂 @VideoN
    (≤3、逐条≤15s),清单/兜底模板/payload 三处一致。"""
    import maestro.pipeline.window_loop as wl
    from maestro.pipeline.window_loop import (
        _generate_with_condition,
        _prepared_source_videos,
        _slot_manifest,
    )
    from maestro.types import AssetMemory, Shot

    vid = tmp_path / "walk.mp4"
    vid.write_bytes(b"\x00" * 16)
    mem = AssetMemory(video_shots={"v0": Shot(
        shot_id="v0", source_video=str(vid), start_time=0, end_time=4,
        caption="a man in a red jacket walking on a boardwalk")})
    monkeypatch.setattr(wl, "_probe_seconds", lambda v: 4.0)   # ≤15s 不裁
    src = _prepared_source_videos(mem, tmp_path / "labels")
    assert len(src) == 1 and src[0][0] == vid
    assert "red jacket" in src[0][1]

    kf = tmp_path / "kf.png"
    kf.write_bytes(b"\x89PNG\r\n")
    e = _entry([{"path": str(kf), "role": "reference",
                 "description": "the red-jacket man"}])

    class _P:
        video_path = str(tmp_path / "prev.mp4")
    Path(_P.video_path).write_text("PREV")

    # 清单:@Image1(尾帧)+ @Image2(本镜图)+ @Video1(素材,user asset 前缀)
    rows = _slot_manifest("ti2v_prev_plus_keyframe", e, _P(),
                          source_videos=src)
    assert [r["slot"] for r in rows] == ["@Image1", "@Image2", "@Video1"]
    assert rows[2]["content"].startswith("user asset:")

    class _Gen:
        def __init__(self):
            self.calls = []

        def capabilities(self):
            return {"t2v", "i2v", "ref_images", "ref_video"}

        def generate(self, prompt, duration, out_path, fps=8,
                     first_frame=None, reference_images=None, seed=0,
                     reference_video=None):
            self.calls.append({"prompt": prompt,
                               "reference_video": reference_video,
                               "first_frame": first_frame})
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("MOCK")
            return p

    prev_last = tmp_path / "prev_last.png"
    prev_last.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: prev_last)
    gen = _Gen()
    from maestro.types import ShotSpec
    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="the man walks on")
    _, cond = _generate_with_condition(
        "ti2v_prev_plus_keyframe", e, _P(), spec, gen, tmp_path / "g",
        seed=0, fps=8, window_tail_s=2.0, source_videos=src)
    call = gen.calls[-1]
    assert call["first_frame"] is None                    # t2v 路线
    assert call["reference_video"] == [vid]               # 素材视频进通道
    assert "opens EXACTLY on @Image1" in call["prompt"]   # G-2 首帧强锁
    assert "@Video1" in call["prompt"]
    assert cond["reference_videos"] == [str(vid)]


def test_flf2v_bridge_never_uses_identity_ref_as_closing_anchor(tmp_path,
                                                                monkeypatch):
    """护栏:身份参考图绝不当收场锚 —— 只有 'last'/首帧角色图可以。"""
    import maestro.pipeline.window_loop as wl
    from maestro.pipeline.window_loop import _generate_with_condition
    from maestro.types import ShotSpec

    ref = tmp_path / "cat_photo.png"
    ref.write_bytes(b"\x89PNG\r\n")
    e = _entry([{"path": str(ref), "role": "reference",
                 "description": "an orange tabby cat"}])   # 仅参考角色

    class _P:
        video_path = str(tmp_path / "prev.mp4")
    Path(_P.video_path).write_text("PREV")
    prev_last = tmp_path / "prev_last.png"
    prev_last.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: prev_last)

    class _Gen:
        def __init__(self):
            self.flf = []

        def capabilities(self):
            return {"t2v", "i2v", "flf2v"}

        def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                           duration=None, seed=0):
            self.flf.append((str(first_frame), str(last_frame)))
            p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("FLF"); return p

        def generate(self, prompt, duration, out_path, fps=8,
                     first_frame=None, reference_images=None, seed=0,
                     reference_video=None):
            p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("MOCK"); return p

    gen = _Gen()
    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="p")
    _, cond = _generate_with_condition(
        "flf2v_bridge", e, _P(), spec, gen, tmp_path / "g",
        seed=0, fps=8, window_tail_s=2.0)
    # 无 last/首帧角色图 → flf2v 不执行,诚实降级(绝不拿身份照片当尾帧)
    assert gen.flf == []
    assert cond.get("degraded_from") == "flf2v_bridge"


def test_cast_and_setting_flow_end_to_end(tmp_path, monkeypatch):
    """跨镜一致性载体(2026-07-17 审计):剧本产出 cast/setting → 台账 →
    enhancer 条件行 → 评审指令(逐角色一致性 check)。"""
    import json as _json

    from maestro.pipeline.window_loop import (
        _conditions_for_prompt,
        _write_outline,
    )

    class _LLM:
        def complete(self, prompt, **kw):
            return _json.dumps({
                "cast": {"the cat": "an orange-and-white cat with a white "
                                    "chest and a thin blue collar"},
                "setting": "a warm sunlit living room with a wooden floor",
                "shots": [{"description": "Shot 1: the cat wakes on the "
                                          "windowsill and stretches slowly",
                           "duration_s": 5, "end_state": "stretching"}]})

    shots, durs, ends, meta, via = _write_outline(
        _LLM(), "p", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["fb"])
    assert via == "llm"
    assert meta["cast"]["the cat"].startswith("an orange-and-white cat")
    assert "living room" in meta["setting"]

    # enhancer 条件行带 cast/setting
    conds = _conditions_for_prompt("t2v", _entry(), None, False,
                                   cast=meta["cast"],
                                   setting=meta["setting"])
    kinds = {c["kind"] for c in conds}
    assert "cast" in kinds and "setting" in kinds
    cast_row = next(c for c in conds if c["kind"] == "cast")
    assert cast_row["role"] == "the cat"

    # 评审指令带官方 cast 契约 + 出场角色逐一 check 要求
    from maestro.models.mllm_backends import GeminiVLM
    from maestro.types import CandidateClip

    vlm = GeminiVLM("gemini", {"api_key": "k"})
    captured = []

    def _fake(parts):
        captured.append(parts)
        return _json.dumps({"checks": [], "issues": [], "summary": "ok"})
    monkeypatch.setattr(vlm, "_generate", _fake)
    v = tmp_path / "s.mp4"
    v.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    clip = CandidateClip(shot_idx=0, video_path=v)
    clip.conditioning = {"video_prompt": "p", "images": [],
                         "cast": meta["cast"], "setting": meta["setting"]}
    vlm.review_shot(clip, ShotSpec(shot_idx=0, duration=5.0, prompt="p"))
    text = " ".join(x.get("text", "") for x in captured[0] if "text" in x)
    assert "CANONICAL CAST" in text
    assert "orange-and-white cat" in text
    assert "CANONICAL SETTING" in text
