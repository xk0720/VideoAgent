"""百炼可灵策略池(2026-08-03 M1)回归:
① 能力标记分流:first_frame_plus_refs → 4 条新菜单;旧后端菜单不受扰;
② 槽位清单方言:<<<image_N>>>,i2v_first 的 refer 编号不含 first_frame;
③ 生成装配:ref2v 全参考;i2v_first 硬钉上镜末帧+参考同请求,接缝
   重复帧无条件切;无上镜时退用自有关键帧(不切);
④ 引用闸门认得 <<<image_N>>> 方言。全部离线。"""
from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.memory.storyboard import ShotEntry
from maestro.pipeline.ref_slots import validate_references
from maestro.types import ShotSpec


class _KlingGen:
    def __init__(self):
        self.calls = []

    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_images", "first_frame_plus_refs"}

    def ref_token(self, n: int) -> str:
        return f"<<<image_{n}>>>"

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        self.calls.append({"prompt": prompt, "first_frame": first_frame,
                           "reference_images": reference_images})
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("MOCK VIDEO")
        return p


def _png(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n" + b"\x00" * 32)
    return p


def _entry(images=None):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="shot 2: <the cat> trots on")
    e.images = list(images or [])
    return e


class _Prev:
    def __init__(self, path):
        self.video_path = path


def _spec():
    return ShotSpec(shot_idx=1, duration=5.0, prompt="the cat trots")


def test_menu_switches_on_capability_marker(tmp_path):
    gen = _KlingGen()
    prev = _Prev(tmp_path / "prev.mp4")
    prev.video_path.write_text("MOCK")
    ports = {"the cat": str(_png(tmp_path / "p.png"))}
    names = [m["name"] for m in wl._condition_menu(_entry(), prev, gen,
                                                   portraits=ports)]
    assert names == ["t2v", "ref2v", "i2v_first"]
    # 旧软钉/extend 系一律不出现
    assert not ({"extend_prev", "ti2v_prev_plus_keyframe",
                 "t2v_own_refs"} & set(names))


def test_manifest_dialect_and_numbering(tmp_path):
    gen = _KlingGen()
    prev = _Prev(tmp_path / "prev.mp4")
    prev.video_path.write_text("MOCK")
    ref = _png(tmp_path / "ref.png")
    e = _entry(images=[{"path": str(ref), "role": "reference",
                        "description": "a planned ref"}])
    ports = {"the cat": str(_png(tmp_path / "cat.png"))}
    rows = wl._slot_manifest("ref2v", e, prev, portraits=ports,
                             video_gen=gen)
    assert rows[0]["slot"] == "<<<image_1>>>"
    assert rows[1]["slot"] == "<<<image_2>>>"
    assert "official portrait of the cat" in rows[1]["content"]
    # i2v_first:FIRST_FRAME 不可引用;refer 编号从 1 起(不含首帧)
    rows2 = wl._slot_manifest("i2v_first", e, prev, portraits=ports,
                              video_gen=gen)
    assert rows2[0]["slot"] == "FIRST_FRAME"
    assert rows2[0]["referenceable"] is False
    assert rows2[1]["slot"] == "<<<image_1>>>"
    assert rows2[2]["slot"] == "<<<image_2>>>"


def test_ref2v_assembly(tmp_path):
    gen = _KlingGen()
    ref = _png(tmp_path / "ref.png")
    port = _png(tmp_path / "cat.png")
    e = _entry(images=[{"path": str(ref), "role": "reference",
                        "description": "a planned ref"}])
    _, cond = wl._generate_with_condition(
        "ref2v", e, None, _spec(), gen, tmp_path / "s", seed=0, fps=8,
        window_tail_s=2.0, portraits={"the cat": str(port)})
    call = gen.calls[-1]
    assert [str(p) for p in call["reference_images"]] == [str(ref),
                                                          str(port)]
    assert call["first_frame"] is None
    assert cond["anchoring"] == "ref2v"
    assert "<<<image_2>>> is the official portrait of the cat" \
        in call["prompt"]


def test_i2v_first_hard_pin_and_dedup(tmp_path, monkeypatch):
    last = _png(tmp_path / "last.png")
    monkeypatch.setattr(wl, "_last_frame", lambda *a, **k: last)
    dropped = []
    monkeypatch.setattr(wl, "_drop_first_frame",
                        lambda outp, cond, measured_prev=None:
                        (dropped.append(str(outp)) or outp))
    gen = _KlingGen()
    prev = _Prev(tmp_path / "prev.mp4")
    prev.video_path.write_text("MOCK")
    port = _png(tmp_path / "cat.png")
    _, cond = wl._generate_with_condition(
        "i2v_first", _entry(), prev, _spec(), gen, tmp_path / "s",
        seed=0, fps=8, window_tail_s=2.0,
        portraits={"the cat": str(port)})
    call = gen.calls[-1]
    assert str(call["first_frame"]) == str(last)      # 硬钉上镜末帧
    assert [str(p) for p in call["reference_images"]] == [str(port)]
    assert cond["anchoring"] == "hard_first_frame"
    assert dropped, "首帧=上镜末帧 → 接缝重复帧必须切"
    # 无上镜 → 退用自有关键帧,不切
    dropped.clear()
    kf = _png(tmp_path / "kf.png")
    e2 = _entry(images=[{"path": str(kf), "role": "first_frame",
                         "description": "opening"}])
    _, cond2 = wl._generate_with_condition(
        "i2v_first", e2, None, _spec(), gen, tmp_path / "s2",
        seed=0, fps=8, window_tail_s=2.0, portraits={})
    assert str(gen.calls[-1]["first_frame"]) == str(kf)
    assert not dropped


def test_ref_gate_accepts_kling_dialect():
    slots = [{"slot": "<<<image_1>>>", "referenceable": True,
              "content": "a planned ref"},
             {"slot": "<<<image_2>>>", "referenceable": True,
              "content": "official portrait of the cat"}]
    ok, audit = validate_references(
        "<<<image_1>>> trots ahead; <<<image_2>>> sets identity.", slots)
    assert audit["ok"] and ok
    bad, audit2 = validate_references("<<<image_9>>> appears.", slots)
    assert bad == "" and not audit2["ok"]
    assert "<<<image_9>>>" in audit2["unknown"]