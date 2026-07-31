"""2026-07-31 双大 bug + 裁决 1 回归。

实锤病根(outputs/movie_20260731_144652):
  ②a shot2 起 image_plan 把官方肖像检索回来当本镜图 —— 同图双通道进
     引用列表、正面全身像支配开场 → 首帧全成肖像照;
  ②b 肖像生成时 setting 还没赋值 + 全角分号绕过拆分器 → 标签原文进
     t2i、背景成了影棚白布。
修法:
  ① 目录排除(portrait: 前缀)+ 计划图出口守卫 + own 空时参考路线照常
    (编号与槽位清单一致)+ 菜单肖像感知;
  ② _static_half 全角兼容 + 肖像背景按影片 setting;
  ③ 裁决 1:repair_keyframe_identity(ViMax 肖像替换)—— 多图编辑 +
    原条件重跑,菜单三重门控。全部离线。"""
from pathlib import Path

import pytest

import maestro.pipeline.window_loop as wl
from maestro.agents.orchestrator import OrchestratorAgent, _clip_portraits
from maestro.memory.storyboard import ShotEntry
from maestro.models.image_edit import MockImageEditClient
from maestro.types import AssetMemory, CandidateClip, Identity, ShotSpec


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 32)
    return path


def _entry(images=None, desc="shot 2: <the cat> trots on"):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description=desc)
    e.images = list(images or [])
    return e


class _Prev:
    def __init__(self, path):
        self.video_path = path


class _RecordingGen:
    def __init__(self):
        self.calls = []

    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_video", "ref_images"}

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        self.calls.append({"prompt": prompt,
                           "reference_images": reference_images,
                           "first_frame": first_frame})
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("MOCK VIDEO")
        return p


# ── ① 肖像绝不进素材目录/计划图 ──────────────────────────────────────

def test_asset_catalog_excludes_official_portraits(tmp_path):
    am = AssetMemory()
    photo = _png(tmp_path / "user_cat.png")
    port = _png(tmp_path / "portrait_cat.png")
    am.identity_anchors["cat"] = Identity(
        identity_id="cat", name="cat", source=str(photo),
        description="the user's orange cat photo")
    am.identity_anchors["portrait:the cat"] = Identity(
        identity_id="portrait:the cat", name="the cat", source=str(port),
        description="character: official portrait of the cat")
    paths = {row["path"] for row in wl._asset_catalog(am)}
    assert str(photo) in paths and str(port) not in paths
    # 全媒体目录(scene_write / image_plan 看的)同样排除
    paths2 = {row["path"] for row in wl._media_catalog(am)}
    assert str(port) not in paths2


def test_image_plan_guard_drops_planned_portrait(tmp_path):
    """检索绕不过目录,但用户素材=肖像同一文件(user_asset 来源)时仍可能
    命中 —— 出口守卫按路径兜底丢弃,计划如实降级。"""
    am = AssetMemory()
    photo = _png(tmp_path / "user_cat.png")
    am.identity_anchors["cat"] = Identity(
        identity_id="cat", name="cat", source=str(photo),
        description="orange cat trots amber eyes")
    decision = {"strategy": "single_reference",
                "images": [{"source": "asset_image",
                            "description": "orange cat amber eyes"}]}
    plan, imgs, degraded = wl._execute_image_plan(
        decision, _entry(), None, am, None, tmp_path / "kf",
        portrait_paths={str(photo.resolve())})
    assert plan == "none" and imgs == []
    assert degraded == "single_reference"


# ── ② 拆分器全角兼容 + 肖像背景按场景 ────────────────────────────────

def test_static_half_fullwidth_and_variants():
    assert wl._static_half(
        "static: 中年女子,齐肩棕发;dynamic: 姿势变化") == "中年女子,齐肩棕发"
    assert wl._static_half(
        "static: tall man in a coat; dynamic: pose") == "tall man in a coat"
    # 变体:全角冒号 / 只有 static 标签 / 无标签
    assert wl._static_half("static:红色斗篷 dynamic:飘动") == "红色斗篷"
    assert wl._static_half("static: red cloak") == "red cloak"
    assert wl._static_half("a plain red cloak") == "a plain red cloak"


def test_scrub_cast_labels_fullwidth():
    cast = {"客": "static: 中年女子;dynamic: 姿势"}
    out = wl._scrub_cast_labels("portrait of static: 中年女子;dynamic: 姿势",
                                cast)
    assert "static" not in out and "dynamic" not in out
    assert "中年女子" in out


def test_portrait_prompt_carries_film_setting(tmp_path):
    """肖像背景 = 影片场景(2026-07-31 裁决);标签绝不外漏。"""
    from maestro.memory.storyboard import StoryboardMemory

    sb = StoryboardMemory.from_outline(["shot 1: x"],
                                       path=tmp_path / "sb.json")
    sb.cast = {"the baker": "static: slender young man, white apron;"
                            "dynamic: pose"}
    sb.setting = "a warm street-corner bakery on a rainy morning"

    class _T2I:
        def __init__(self):
            self.prompts = []

        def text_to_image(self, prompt, out, seed=0):
            self.prompts.append(prompt)
            return _png(Path(out))

    gen = _T2I()
    wl._ensure_cast_portraits(sb, AssetMemory(), gen, tmp_path)
    p = gen.prompts[0]
    assert "street-corner bakery" in p          # 场景进背景
    assert "Background:" in p
    assert "static" not in p and "dynamic" not in p
    assert "the film scene" not in p            # 空话兜底已废除


# ── ① own 空 + 有肖像:菜单在、装配走、编号一致 ─────────────────────

def test_condition_menu_knows_portraits(tmp_path):
    gen = _RecordingGen()
    prev = _Prev(tmp_path / "prev.mp4")
    prev.video_path.write_text("MOCK")
    ports = {"the cat": str(_png(tmp_path / "p.png"))}
    names = {m["name"] for m in wl._condition_menu(_entry(), prev, gen,
                                                   portraits=ports)}
    assert {"t2v_own_refs", "ti2v_prev_plus_keyframe"} <= names
    # 无图无肖像 → 两条参考路线都不该出现
    names0 = {m["name"] for m in wl._condition_menu(_entry(), prev, gen)}
    assert "t2v_own_refs" not in names0
    assert "ti2v_prev_plus_keyframe" not in names0


def test_ti2v_portraits_ride_when_own_empty(tmp_path, monkeypatch):
    last = _png(tmp_path / "last.png")
    monkeypatch.setattr(wl, "_last_frame", lambda *a, **k: last)
    monkeypatch.setattr(wl, "_drop_first_frame",
                        lambda outp, cond, measured_prev=None: outp)
    gen = _RecordingGen()
    prev = _Prev(tmp_path / "prev.mp4")
    prev.video_path.write_text("MOCK")
    port = _png(tmp_path / "cat_portrait.png")
    entry = _entry()                              # 本镜没有自己的图
    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="the cat trots")
    _, cond = wl._generate_with_condition(
        "ti2v_prev_plus_keyframe", entry, prev, spec, gen,
        tmp_path / "s1", seed=0, fps=8, window_tail_s=2.0,
        portraits={"the cat": str(port)})
    call = gen.calls[-1]
    # 装配 = [上镜尾帧] + 肖像;不降级
    assert [str(p) for p in call["reference_images"]] == [str(last),
                                                          str(port)]
    assert cond["strategy"] == "ti2v_prev_plus_keyframe"
    assert "degraded_from" not in cond
    # 兜底模板编号与槽位清单一致:@Image2 = 肖像(身份参考,不许复刻构图)
    assert "@Image2 is the official portrait of the cat" in call["prompt"]
    rows = wl._slot_manifest("ti2v_prev_plus_keyframe", entry, prev,
                             portraits={"the cat": str(port)})
    assert rows[1]["slot"] == "@Image2"
    assert "official portrait of the cat" in rows[1]["content"]


def test_t2v_own_refs_portraits_only_no_degrade(tmp_path):
    gen = _RecordingGen()
    port = _png(tmp_path / "cat_portrait.png")
    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="the cat trots")
    _, cond = wl._generate_with_condition(
        "t2v_own_refs", _entry(), None, spec, gen, tmp_path / "s1",
        seed=0, fps=8, window_tail_s=2.0,
        portraits={"the cat": str(port)})
    call = gen.calls[-1]
    assert [str(p) for p in call["reference_images"]] == [str(port)]
    assert cond["strategy"] == "t2v_own_refs"
    assert "degraded_from" not in cond
    assert "@Image1 is the official portrait of the cat" in call["prompt"]


# ── ③ 裁决 1:ViMax 肖像替换修复 ─────────────────────────────────────

def _repair_clip(tmp_path):
    kf = _png(tmp_path / "kf.png")
    port = _png(tmp_path / "portrait.png")
    clip = CandidateClip(shot_idx=1, video_path=tmp_path / "v.mp4",
                         revision=0)
    clip.keyframes = [kf]
    clip.conditioning = {"images": [
        {"path": str(port), "role": "identity_portrait", "name": "the cat"}]}
    return clip, kf, port


def test_mock_image_edit_records_references(tmp_path):
    kf = _png(tmp_path / "kf.png")
    ref = _png(tmp_path / "ref.png")
    out = MockImageEditClient().edit(kf, "swap the person", tmp_path / "o.png",
                                     references=[ref])
    assert str(ref) in out.read_text(encoding="utf-8")


def test_menu_gates_identity_repair(tmp_path):
    clip, _kf, _p = _repair_clip(tmp_path)
    orch = OrchestratorAgent(image_edit=MockImageEditClient())
    names = [m["name"] for m in orch.available_actions(clip=clip)]
    assert names[0] == "repair_keyframe_identity"
    # 三重门控:无肖像 / 无关键帧 / 无编辑客户端 → 不出现
    bare = CandidateClip(shot_idx=1, video_path=tmp_path / "v.mp4",
                         revision=0)
    assert "repair_keyframe_identity" not in [
        m["name"] for m in orch.available_actions(clip=bare)]
    orch2 = OrchestratorAgent(image_edit=None)
    assert "repair_keyframe_identity" not in [
        m["name"] for m in orch2.available_actions(clip=clip)]


def test_execute_identity_repair_edits_then_regen(tmp_path):
    clip, kf, port = _repair_clip(tmp_path)
    assert _clip_portraits(clip) == {"the cat": str(port)}
    orch = OrchestratorAgent(image_edit=MockImageEditClient())
    seen = {}

    def regen_fn(seed, hint="", first_frame=None):
        seen.update(seed=seed, hint=hint, first_frame=first_frame)
        out = tmp_path / "regen.mp4"
        out.write_text("MOCK")
        return out, {"strategy": "i2v_keyframe"}

    class _Board:
        def review(self, cand, spec, asset_memory, fps):
            return None

    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="the cat trots")
    cand = orch.execute(
        {"tool": "repair_keyframe_identity",
         "args": {"character": "the cat"}},
        clip, spec, tmp_path, 1, _Board(), regen_fn=regen_fn)
    assert cand is not None and seen["first_frame"] is not None
    edited = Path(seen["first_frame"])
    assert edited.exists() and cand.keyframes == [edited]
    body = edited.read_text(encoding="utf-8")
    # 编辑指令 = 显式绑定(第一张=底、第二张=肖像)+ 换人保景 + 写实钉
    assert "official portrait of the cat" in body
    assert "SECOND image" in body and "photorealistic" in body
    assert str(port) in body
    # 名字对不上但只有一位肖像 → 唯一肖像即所指(别名兼容)
    seen.clear()
    cand2 = orch.execute(
        {"tool": "repair_keyframe_identity", "args": {"character": "猫"}},
        clip, spec, tmp_path, 2, _Board(), regen_fn=regen_fn)
    assert cand2 is not None and seen["first_frame"] is not None
