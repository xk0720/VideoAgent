"""reward v3 判官层(2026-08-14):排名点数/并列、合成归一、文本判官
条件维、一致性 null 项、收集器 v3 接线(桩判官)。零 API、零 maestro
依赖。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl"))

from reward.judges import (ConsistencyChecker, TextJudge,  # noqa: E402
                           VideoRanker, compose_rewards, rank_to_points)


class _FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, content, retries=2):
        self.calls.append(content)
        return self.reply if isinstance(self.reply, str) \
            else self.reply.pop(0)


def test_rank_points_with_ties():
    # [B, [A,C], D]:B=3, A/C 并列 2、1 名 → 平均 1.5, D=0;除以 3 归一
    pts = rank_to_points(["B", ["A", "C"], "D"], 4)
    assert pts["B"] == 1.0
    assert abs(pts["A"] - 0.5) < 1e-9 and pts["A"] == pts["C"]
    assert pts["D"] == 0.0


def test_ranker_unshuffles_back_to_candidate_index():
    reply = json.dumps({"ranking": ["A", "B", "C", "D"],
                        "evidence": {}})
    rk = VideoRanker(_FakeChat(reply), rng_seed=7)
    vids = [f"v{i}.mp4" for i in range(4)]
    # 不真读文件:打桩 _video_part
    import reward.judges as J
    orig = J._video_part
    J._video_part = lambda p: {"type": "video_url",
                               "video_url": {"url": p}}
    try:
        res = rk.rank("action", {"shot_script": "x"}, vids)
    finally:
        J._video_part = orig
    # 展示序被打乱,但点数按【候选下标】返回:A(展示第一)= 打乱后
    # order[0] 那个候选拿满分
    assert set(res["points"]) == {0, 1, 2, 3}
    assert res["points"][res["order"][0]] == 1.0
    assert res["points"][res["order"][3]] == 0.0


def test_text_judge_conditional_dim_normalization():
    reply = json.dumps({"scores": {"script_faithfulness": 5,
                                   "visual_specificity": 4,
                                   "transition_continuity": None,
                                   "character_consistency": 3},
                        "rationale": {}})
    score, detail = TextJudge.__new__(TextJudge).__class__.score(
        _make_text_judge(reply), {"candidate_prompt": "x"})
    # null 维剔除:(5+4+3)/(5*3) = 0.8
    assert abs(score - 0.8) < 1e-9


def _make_text_judge(reply):
    tj = TextJudge.__new__(TextJudge)
    tj.client = _FakeChat(reply)
    from reward.judges import _skill
    tj.skill = _skill("prompt_review")
    return tj


def test_consistency_null_items_excluded():
    reply = json.dumps({"checks": [
        {"item": "portrait:A/clothing", "pass": True},
        {"item": "portrait:A/hairstyle", "pass": False},
        {"item": "space/wall", "pass": None, "note": "never in frame"}]})
    cc = ConsistencyChecker.__new__(ConsistencyChecker)
    cc.client = _FakeChat(reply)
    from reward.judges import _skill
    cc.skill = _skill("consistency_check")
    import reward.judges as J
    orig_v, orig_i = J._video_part, J._image_part
    J._video_part = J._image_part = lambda p: {"type": "text", "text": p}
    try:
        score, detail = cc.score("v.mp4", [], {})
    finally:
        J._video_part, J._image_part = orig_v, orig_i
    assert abs(score - 0.5) < 1e-9        # null 不计分母
    assert detail["n_null"] == 1


def test_compose_renormalizes_on_missing_components():
    fmt = [1.0, 1.0]
    text = [0.8, None]                     # 候选1 文本判官失败
    video_parts = {"action": {0: 1.0, 1: 0.0},
                   "physics": None,        # 物理判官整路失败
                   "camera": {0: 0.5, 1: 0.5},
                   "consistency": {0: 1.0, 1: 0.5}}
    out = compose_rewards(fmt, text, video_parts, 2)
    # 候选0:r_video = (0.30*1 + 0.15*0.5 + 0.30*1)/0.75 = 0.9
    assert abs(out[0]["r_video"] - 0.9) < 1e-9
    assert out[0]["dropped_components"] == ["physics"]
    # 候选1 无文本分:r = (0.15*1 + 0.5*r_video)/0.65
    rv1 = (0.30 * 0.0 + 0.15 * 0.5 + 0.30 * 0.5) / 0.75
    exp = (0.15 * 1.0 + 0.5 * rv1) / 0.65
    assert abs(out[1]["reward"] - exp) < 1e-3   # reward 落盘 round(4)
    assert out[1]["r_text"] is None


def test_collector_v3_overwrites_rewards(tmp_path):
    """桩判官走通收集器 v3 全链:案卷组装/覆写/明细留痕。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                            / "rl" / "collect"))
    import watch_online as W
    run = tmp_path / "movie_x"
    run.mkdir()
    for i in range(2):
        (run / f"v{i}.mp4").write_bytes(b"fake")
    (run / "storyboard.json").write_text(json.dumps({
        "entries": [{"shot_idx": 0, "camera_facing": "朝柜台",
                     "junction_meta": {"kind": "continue"}}],
        "portraits": {}}))
    g = {"kind": "condition_group", "run": "movie_x", "shot_idx": 0,
         "label": "s1", "junction_kind": "continue",
         "policy_version": "0", "group_size": 2,
         "menu": [{"name": "t2v"}],
         "context": {"shot": {"description": "A 走进店里。"},
                     "cast": {}, "cast_in_shot": [],
                     "prev_shot": {"end_state": "A 在门口"},
                     "storyboard": [], "slots_by_strategy": {"t2v": []}},
         "samples": [
             {"decision_id": "d0", "via": "llm", "chosen": True,
              "completion": json.dumps({"strategy": "t2v",
                                        "video_prompt": "p0"}),
              "weighted_total": 0.7, "video": str(run / "v0.mp4"),
              "metrics": {"m1_semantic": 0.7, "p1_physics": 0.7}},
             {"decision_id": "d1", "via": "llm", "chosen": False,
              "completion": json.dumps({"strategy": "t2v",
                                        "video_prompt": "p1"}),
              "weighted_total": 0.6, "video": str(run / "v1.mp4"),
              "metrics": {"m1_semantic": 0.6, "p1_physics": 0.6}}]}
    (run / "rl_steps.jsonl").write_text(json.dumps(g) + "\n")

    class _TJ:
        def score(self, case, tag=None):
            assert case["junction"]["continuity_applicable"] is True
            return (0.9 if case["candidate_prompt"] == "p0" else 0.4,
                    {"scores": {}})

    class _RK:
        def rank(self, dim, ctx, videos, tag=None):
            return {"points": {0: 1.0, 1: 0.0}, "order": [0, 1],
                    "evidence": {}}

    class _CC:
        def score(self, video, refs, ctx, tag=None):
            return 0.8, {"checks": []}

    groups = W.collect_run(run, set(), judges={
        "text": _TJ(), "ranker": _RK(), "consistency": _CC()})
    s0, s1 = groups[0]["samples"]
    assert s0["r_text"] == 0.9 and s1["r_text"] == 0.4
    # 无肖像/视图参考 → consistency 跳过并被剔除留痕
    assert "consistency" in s0["dropped_components"]
    assert s0["reward"] > s1["reward"]     # 排名+文本双赢 → 合成分更高


def test_batch_metrics_component_means():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                            / "rl" / "train"))
    from train_online import batch_metrics
    batch = [{"samples": [
        {"reward": 0.8, "r_format": 1.0, "r_text": 0.9, "r_video": 0.7,
         "advantage": 0.1,
         "video_detail": {"action": 1.0, "physics": 0.5}},
        {"reward": 0.4, "r_format": 1.0, "r_text": None, "r_video": 0.3,
         "advantage": -0.1,
         "video_detail": {"action": 0.0}}]}]
    m = batch_metrics(batch)
    assert m["reward/mean"] == 0.6
    assert m["reward/text"] == 0.9          # None 剔除后均值
    assert m["batch/judged_text_rate"] == 0.5
    assert m["video/action"] == 0.5
    assert m["video/physics"] == 0.5        # 只有一个样本有该维
    assert m["video/consistency"] is None   # 全缺 → None,不造 0
    assert abs(m["advantage/std"] - 0.1) < 1e-9


def test_group_rank_lines_per_candidate():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                            / "rl" / "train"))
    from train_online import group_rank_lines
    batch = [{"run": "movie_x", "label": "s1", "samples": [
        {"video_detail": {"action": 1.0, "physics": 2 / 3,
                          "camera": 1.0, "consistency": 0.85}},
        {"video_detail": {"action": 0.0, "physics": None,
                          "camera": 0.5}},
        {"video_detail": {}}]}]
    lines = group_rank_lines(batch)
    assert len(lines) == 1
    assert "c0 a/p/c=1.00/0.67/1.00 avg=0.89 con=0.85" in lines[0]
    assert "c1 a/p/c=0.00/--/0.50 avg=0.25" in lines[0]
    assert "c2" not in lines[0]               # 无排序分的候选不占行


def test_judge_log_written(tmp_path):
    """2026-08-19 用户令:每次评审结果留痕 JSONL(成败皆记)。"""
    from reward.judges import JudgeLog, TextJudge
    reply = json.dumps({"scores": {"script_faithfulness": 4,
                                   "visual_specificity": 4,
                                   "transition_continuity": None,
                                   "character_consistency": 5},
                        "rationale": {"script_faithfulness": "ok"}})
    tj = TextJudge.__new__(TextJudge)
    tj.client = _FakeChat(reply)
    from reward.judges import _skill
    tj.skill = _skill("prompt_review")
    tj.log = JudgeLog(tmp_path / "judge_calls.jsonl")
    tj.score({"candidate_prompt": "x"},
             tag={"run": "movie_x", "label": "s1", "candidate": 0})
    rec = json.loads((tmp_path / "judge_calls.jsonl").read_text())
    assert rec["judge"] == "text" and rec["error"] == ""
    assert rec["tag"]["label"] == "s1"
    assert rec["scores"]["character_consistency"] == 5
    assert "ts" in rec and rec["latency_s"] >= 0
