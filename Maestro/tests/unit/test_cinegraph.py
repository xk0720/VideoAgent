"""cinegraph(ViMax 移植)单元测试:机位树校验、选图退化、注册表
降级、切点检测。"""
import json
import subprocess

from maestro.cinegraph.camera_tree import _validate_items, \
    construct_camera_tree
from maestro.cinegraph.first_frame_factory import _frame_after_cut
from maestro.cinegraph.interfaces import CGCamera, CGShot
from maestro.cinegraph.portrait_registry import build_portrait_registry, \
    registry_pairs
from maestro.cinegraph.reference_selector import select_reference_images


def _shots():
    return [CGShot(idx=0, cam_idx=0, visual_desc="大远景,二人对立",
                   ff_desc="a", lf_desc="b"),
            CGShot(idx=1, cam_idx=1, visual_desc="特写,甲的脸",
                   ff_desc="a", lf_desc="b")]


def _cams():
    return [CGCamera(idx=0, active_shot_idxs=[0]),
            CGCamera(idx=1, active_shot_idxs=[1])]


def test_tree_validation_rules():
    shots, cams = _shots(), _cams()
    ok = [{"parent_cam_idx": None, "parent_shot_idx": None, "reason": "root"},
          {"parent_cam_idx": 0, "parent_shot_idx": 0, "reason": "特写⊂远景",
           "is_parent_fully_covers_child": False,
           "missing_info": "甲的正脸"}]
    assert _validate_items(cams, shots, ok) is None
    assert "length" in _validate_items(cams, shots, ok[:1])
    bad_cycle = [{"parent_cam_idx": 1, "parent_shot_idx": 1, "reason": "x"},
                 {"parent_cam_idx": 0, "parent_shot_idx": 0, "reason": "x"}]
    assert _validate_items(cams, shots, bad_cycle) is not None
    two_roots = [{"parent_cam_idx": None, "reason": "r"},
                 {"parent_cam_idx": None, "reason": "r"}]
    assert "root" in _validate_items(cams, shots, two_roots)


def test_tree_llm_and_flat_degrade():
    shots, cams = _shots(), _cams()

    class _GoodLLM:
        def complete(self, prompt, **kw):
            return json.dumps({"camera_parent_items": [
                {"parent_cam_idx": None, "parent_shot_idx": None,
                 "reason": "root"},
                {"parent_cam_idx": 0, "parent_shot_idx": 0,
                 "reason": "close-up inside wide",
                 "is_parent_fully_covers_child": False,
                 "missing_info": "front face"}]})
    out = construct_camera_tree(_GoodLLM(), cams, shots)
    assert out[1].parent_cam_idx == 0 and out[0].parent_cam_idx is None

    class _BadLLM:
        def complete(self, prompt, **kw):
            return "not json at all"
    shots2, cams2 = _shots(), _cams()
    out2 = construct_camera_tree(_BadLLM(), cams2, shots2)
    # 两轮失败 → 平树降级:0 号根,其余挂 0 号
    assert out2[0].parent_cam_idx is None
    assert out2[1].parent_cam_idx == 0


def test_selector_retry_then_degrade():
    pairs = [(f"/p/{i}.png", f"desc {i}") for i in range(3)]

    class _OOBLLM:
        def complete(self, prompt, **kw):
            return json.dumps({"ref_image_indices": [9],
                               "text_prompt": "x"})
    out = select_reference_images(_OOBLLM(), pairs, "帧描述")
    # 越界两轮 → 诚实降级:近期全量 + 帧描述直出
    assert out["reference_image_path_and_text_pairs"] == pairs
    assert "帧描述" in out["text_prompt"]

    class _GoodLLM:
        def complete(self, prompt, **kw):
            return json.dumps({"ref_image_indices": [2, 0],
                               "text_prompt": "The man references Image 0."})
    out2 = select_reference_images(_GoodLLM(), pairs, "帧描述")
    assert [p for p, _ in out2["reference_image_path_and_text_pairs"]] == \
        ["/p/2.png", "/p/0.png"]


def test_registry_front_only_degrade(tmp_path):
    front = tmp_path / "甲.png"
    front.write_bytes(b"x")
    reg = build_portrait_registry({"甲": str(front), "缺": "/no/such.png"},
                                  image_edit=None, out_dir=tmp_path / "r")
    assert list(reg) == ["甲"]                    # 缺图角色被剔除
    assert list(reg["甲"]) == ["front"]           # 无编辑端 → front-only
    pairs = registry_pairs(reg, ["甲", "无此人"])
    assert len(pairs) == 1 and pairs[0][0] == str(front)


def test_frame_after_cut_detects_hard_cut(tmp_path):
    """合成 2s 双色视频(黑→白硬切)→ 必须取到切后(白)帧。"""
    v = tmp_path / "cut.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1",
         "-f", "lavfi", "-i", "color=c=white:s=160x90:d=1",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
         str(v)], check=True)
    out = _frame_after_cut(v, tmp_path / "cut.png")
    import cv2
    img = cv2.imread(str(out))
    assert img.mean() > 200                       # 白帧(切后)


def test_spaced_retry_rides_transient(monkeypatch):
    """图像端点分钟级抖动:失败一次后成功即返回;穷尽才上抛。"""
    from maestro.cinegraph import first_frame_factory as fff
    monkeypatch.setattr(fff.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("SSL EOF")
        return "ok"
    assert fff._spaced_retry(flaky, tag="t") == "ok" and len(calls) == 2

    def dead():
        raise RuntimeError("down")
    import pytest as _pt
    with _pt.raises(RuntimeError, match="spaced attempts"):
        fff._spaced_retry(dead, tag="t")
