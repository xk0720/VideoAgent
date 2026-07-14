"""GeminiVLM 原生视频评审(2026-07-14 裁决)——合并单次调用、条件对照、
定位输出映射、盲测 verify_pair、Verifier A/B 主闸。CPU-only,无网络:
_generate 全部打桩,校验的是【prompt 部件装配】和【输出→管线对象映射】。"""
import json
from pathlib import Path

from maestro.agents.verifier import VerifierAgent
from maestro.models.mllm_backends import GeminiVLM
from maestro.types import CandidateClip, PhysFailureMode, ShotSpec

_MP4_MAGIC = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _real_video(tmp_path, name="v.mp4"):
    p = tmp_path / name
    p.write_bytes(_MP4_MAGIC)
    return p


def _vlm(monkeypatch, reply: dict):
    vlm = GeminiVLM("gemini", {"api_key": "k"})
    vlm._captured = []

    def _fake_generate(parts):
        vlm._captured.append(parts)
        return json.dumps(reply)
    monkeypatch.setattr(vlm, "_generate", _fake_generate)
    return vlm


_REVIEW_REPLY = {
    "checks": [
        {"question": "Does the glass fall off the table?", "passed": True},
        {"question": "Does the subject match reference image 1?", "passed": False},
    ],
    "issues": [
        {"type": "segment", "time_start_s": 2.0, "time_end_s": 3.0,
         "category": "physics", "physics_mode": "object_permanence",
         "entity": "the glass", "severity": 0.7,
         "problem": "the glass vanishes mid-fall",
         "reason": "objects cannot disappear",
         "suggestion": "keep the glass visible through the fall",
         "check_ref": -1},
        {"type": "frame", "time_start_s": 0.5, "time_end_s": 0.7,
         "category": "condition", "entity": "the cat", "severity": 0.5,
         "problem": "the cat's fur color differs from the reference",
         "reason": "must match reference image 1",
         "suggestion": "orange tabby coat as in the reference",
         "check_ref": 1},
    ],
    "summary": "one physics break and one condition mismatch",
}


def test_review_shot_single_call_maps_to_pipeline_objects(tmp_path, monkeypatch):
    """一次上传 → checklist 项 + 物理 verdict;秒→帧换算;fix 落到失败项;
    条件图和参考视频以带角色标签的 parts 进入同一调用。"""
    vlm = _vlm(monkeypatch, _REVIEW_REPLY)
    clip = CandidateClip(shot_idx=0, video_path=_real_video(tmp_path))
    kf = tmp_path / "kf.png"; kf.write_bytes(b"\x89PNG\r\n" + b"\x00" * 16)
    ref = _real_video(tmp_path, "prev_tail.mp4")
    clip.conditioning = {"video_prompt": "the glass falls",
                         "images": [{"path": str(kf), "role": "reference"}],
                         "reference_video": str(ref)}
    spec = ShotSpec(shot_idx=0, duration=5.0, prompt="a glass falls off a table")

    sem = vlm.assess_semantic(clip, spec)
    phys = vlm.assess_physics(clip, spec, fps=24)

    assert len(vlm._captured) == 1                 # U6:两个评审共享一次上传
    parts = vlm._captured[0]
    texts = [p.get("text", "") for p in parts if "text" in p]
    assert any("THE SHOT VIDEO" in t for t in texts)
    assert any("REFERENCE image" in t for t in texts)      # 条件图在场
    assert any("REFERENCE VIDEO" in t for t in texts)      # 参考视频在场
    assert any("check_ref" in t for t in texts)            # 评审指令在场
    n_videos = sum(1 for p in parts
                   if p.get("inline_data", {}).get("mime_type") == "video/mp4")
    assert n_videos == 2                                    # 成片 + 参考视频

    # checks → checklist(失败项带定位与 fix,来自 check_ref 链接的 issue)
    assert (sem[0][0], sem[0][1]) == ("Does the glass fall off the table?", True)
    q, passed, fix, fr = sem[1]
    assert not passed and "orange tabby" in fix
    assert fr == (12, 17)                    # 0.5-0.7s @24fps,frame 窗口≥1
    # physics issue → verdict(秒→帧、mode 映射、建议成 intervention)
    assert len(phys) == 1
    v = phys[0]
    assert v.mode == PhysFailureMode.OBJECT_PERMANENCE
    assert v.frame_range == (48, 72) and v.entity == "the glass"
    assert "visible" in v.suggested_intervention


def test_review_shot_honest_on_stub_and_caches(tmp_path, monkeypatch):
    vlm = _vlm(monkeypatch, _REVIEW_REPLY)
    stub = tmp_path / "mock.mp4"; stub.write_text("MOCK VIDEO")
    clip = CandidateClip(shot_idx=0, video_path=stub)
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="x")
    assert vlm.assess_semantic(clip, spec) == []       # 桩 → 沉默,不上载
    assert vlm._captured == []
    clip2 = CandidateClip(shot_idx=0, video_path=_real_video(tmp_path))
    vlm.assess_semantic(clip2, spec)
    vlm.assess_physics(clip2, spec, fps=24)
    vlm.assess_semantic(clip2, spec)
    assert len(vlm._captured) == 1                     # 缓存:仍只一次上传


_PAIR_REPLY = {
    "dim_scores": {"semantic": 2, "physics": 5, "temporal": 0, "visual": -1},
    "notes": {"semantic": "s", "physics": "p", "temporal": "t", "visual": "v"},
    "score": 4,
    "defect_present": {"video1": True, "video2": False},
    "issues": ["video 1 still loses the glass mid-fall"],
    "summary": "video 2 fixes the disappearance",
}


def test_verify_pair_blind_remap_and_acceptance(tmp_path, monkeypatch):
    """盲测:随机槽位 + 分数按槽位回映射;accept 规则 = 总分≥+1 且无维度
    ≤-3(维度不回退守卫)。seed 固定保证可复现。"""
    vlm = _vlm(monkeypatch, _PAIR_REPLY)
    cand = _real_video(tmp_path, "cand.mp4")
    base = _real_video(tmp_path, "base.mp4")
    td = {"note": "object_permanence (vlm)", "fix_hint": "keep visible",
          "time_range_s": [2.0, 3.0], "entity": "the glass"}
    # seed=0 → candidate_is_v2 结果确定;两种槽位都验证符号回映射
    v = vlm.verify_pair(str(cand), str(base), "a glass falls",
                        repair_context={"target_defect": td}, seed=0)
    assert v is not None
    order = v["_order"]
    sign = 1 if order["video2"] == "candidate" else -1
    assert v["score"] == sign * 4
    assert v["dim_scores"]["physics"] == sign * 5
    if sign == 1:   # candidate=V2:defect 在 V1(baseline)→ target_fixed=True
        assert v["target_fixed"] is True
        assert v["conclusion"] == "accept"       # 4≥1 且 min(dim)=-1≥-2
        assert v["issues"] == []                 # 赢家不背输家的问题
    else:           # candidate=V1:回映射后 score=-4 → reject + issues 保留
        assert v["conclusion"] == "reject"
        assert v["issues"]


def test_verifier_ab_primary_gate_and_fallback(tmp_path, monkeypatch):
    """Verifier:评委有 verify_pair → 盲测主闸(accept/reject 按 conclusion,
    verdict 挂在候选上);verify_pair 返回 None → 落回指标闸。"""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="x")

    class _ABJudge:
        def __init__(self, verdict):
            self.verdict = verdict
            self.calls = []

        def verify_pair(self, c, b, prompt, repair_context=None, seed=None):
            self.calls.append(repair_context)
            return self.verdict

    def _clip(total, name):
        c = CandidateClip(shot_idx=0, video_path=tmp_path / name)
        c.metric_scores = {"weighted_total": total}
        return c

    ok = {"score": 3, "dim_scores": {"semantic": 3, "physics": 0,
                                     "temporal": 0, "visual": 0},
          "notes": {}, "issues": [], "summary": "better",
          "conclusion": "accept", "target_fixed": True, "_order": {}}
    j = _ABJudge(ok)
    v = VerifierAgent(judge=j)
    cand, best = _clip(0.1, "c.mp4"), _clip(0.9, "b.mp4")
    # 指标更低也能被 A/B 接受 —— A/B 是主闸,指标只观测
    assert v.is_better(cand, best, spec=spec,
                       repair_context={"tool": "edit_clip"}) is True
    assert cand.verifier_verdict["score"] == 3
    assert j.calls[0]["tool"] == "edit_clip"

    bad = dict(ok); bad["conclusion"] = "reject"; bad["score"] = 0
    assert VerifierAgent(judge=_ABJudge(bad)).is_better(
        _clip(0.9, "c2.mp4"), _clip(0.1, "b2.mp4"), spec=spec) is False

    # verify_pair 不可用(None)→ 指标闸兜底:0.9 > 0.1 → 接受
    class _DeadJudge:
        def verify_pair(self, *a, **k):
            return None
    assert VerifierAgent(judge=_DeadJudge()).is_better(
        _clip(0.9, "c3.mp4"), _clip(0.1, "b3.mp4"), spec=spec) is True
