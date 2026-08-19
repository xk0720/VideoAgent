"""rl/env agent loop 单测(2026-08-19 用户令:loop 重建进 rl/ 后,
本文件针对 rl/env —— 桩件同时被 run_grpo.sh --smoke 复用)。
零 API:LLM/生成器/判官全打桩。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl"))

from env import loop as L                                     # noqa: E402


# ── 桩件(--smoke 复用)────────────────────────────────────────────
class FakeFrozenLLM:
    def complete(self, prompt, temperature=None, max_tokens=None):
        if '"cast"' in prompt[-3000:]:          # scene_write 契约尾
            return json.dumps({
                "cast": {"小明": "static: 蓝夹克短发男子; dynamic: none"},
                "setting": "深夜便利店,冷白灯光",
                "shots": [
                    {"description": "Shot 1: <小明>推门进入便利店",
                     "duration_s": 5, "end_state": "小明站在货架前",
                     "variation": "medium",
                     "opening_frame": "便利店门口静景", "bg": "bg_1"},
                    {"description": "Shot 2: <小明>拿起饭团走向收银台",
                     "duration_s": 5, "end_state": "小明到达收银台",
                     "variation": "medium", "bg": "bg_1",
                     "dialogue": {"speaker": "小明", "line": "就这个吧"}},
                    {"description": "Shot 3: <小明>走出店门",
                     "duration_s": 4, "end_state": "门关上",
                     "variation": "large", "bg": "bg_2"}]},
                ensure_ascii=False)
        return json.dumps({"prompt": "empty convenience store interior"})


class FakePolicy:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, temperature=None, max_tokens=None):
        self.calls.append(temperature)
        strat = "ref2v" if '"ref2v"' in prompt else "t2v"
        return json.dumps(
            {"strategy": strat, "reason": "test",
             "video_prompt": "<小明>在<<<image_1>>>所示空间行动"},
            ensure_ascii=False)


class FakeT2I:
    def text_to_image(self, prompt, out, seed=0):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"png")
        return out


class FakeKling:
    def generate(self, prompt, duration, out, first_frame=None,
                 reference_images=None, audio=False):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"mp4")
        return out


class FakeJudges(dict):
    def __init__(self):
        class T:
            def score(self, case, tag=None):
                return 0.8, {"scores": {}}

        class R:
            def rank(self, dim, ctx, videos, tag=None):
                return {"points": {i: 1.0 - 0.2 * i
                                   for i in range(len(videos))},
                        "order": list(range(len(videos))),
                        "evidence": {}}

        class C:
            def score(self, video, refs, ctx, tag=None):
                return 0.9, {"checks": []}
        super().__init__(text=T(), ranker=R(), consistency=C())


def _episode(tmp_path, group=4):
    run = tmp_path / "movie_test"
    pol = FakePolicy()
    L.run_episode(task_text="深夜便利店,店员和最后一位客人的十分钟",
                  run_dir=run, frozen_llm=FakeFrozenLLM(), policy=pol,
                  kling=FakeKling(), t2i=FakeT2I(), judges=FakeJudges(),
                  group=group, rl_temperature=0.9)
    recs = [json.loads(x) for x in
            (run / "rl_steps.jsonl").read_text().splitlines()]
    return run, pol, recs


def test_group_sampling_and_temperatures(tmp_path):
    """K 组采样:v0 默认温度(None → 客户端默认),其余带 rl 温度。"""
    _run, pol, recs = _episode(tmp_path)
    assert len(recs) == 3
    assert all(r["group_size"] == 4 and len(r["samples"]) == 4
               for r in recs)
    assert pol.calls[:4] == [None, 0.9, 0.9, 0.9]


def test_record_schema_and_trunk(tmp_path):
    """记录自包含:menu/context/completion/reward 字段齐;主干唯一,
    且 = reward argmax(桩判官给 c0 最高)。"""
    _run, _pol, recs = _episode(tmp_path)
    g = recs[1]
    for f in ("kind", "run", "shot_idx", "label", "junction_kind",
              "policy_version", "group_size", "menu", "context",
              "samples"):
        assert f in g, f
    s0 = g["samples"][0]
    for f in ("decision_id", "via", "completion", "raw", "usable",
              "strategy", "final_prompt", "video", "chosen", "reward",
              "r_format", "r_text", "r_video", "video_detail",
              "dropped_components"):
        assert f in s0, f
    assert sum(1 for s in g["samples"] if s["chosen"]) == 1
    assert s0["chosen"] is True
    comp = json.loads(s0["completion"])
    assert set(comp) == {"strategy", "reason", "video_prompt"}
    ctx = g["context"]
    assert set(ctx) == {"shot", "prompt_language", "prev_shot",
                        "junction", "cast", "setting", "cast_in_shot",
                        "slots_by_strategy", "storyboard",
                        "episode_guidance"}


def test_junction_and_menu_lock(tmp_path):
    """精简 junction:shot0=None(全菜单)、同 bg=continue、换 bg=cut;
    非首镜菜单锁 [ref2v](与生产菜单锁一致)。"""
    _run, _pol, recs = _episode(tmp_path)
    assert [r["junction_kind"] for r in recs] == [None, "continue", "cut"]
    assert [m["name"] for m in recs[0]["menu"]] == ["t2v", "ref2v"]
    assert [m["name"] for m in recs[1]["menu"]] == ["ref2v"]


def test_outgoing_prompt_chain():
    """出门链:剥标记 → 引用闸 → 名字终换 → 对白+无BGM 压制句;
    引用清单外编号 → 弃用整条落剧本兜底。"""
    entry = {"description": "Shot 2: <小明>拿起饭团",
             "dialogue": "就这个吧", "dialogue_speaker": "小明",
             "end_state": ""}
    slots = [{"slot": "<<<image_1>>>", "content": "master plate",
              "referenceable": True},
             {"slot": "<<<image_2>>>", "content": "portrait",
              "referenceable": True, "name": "小明"}]
    p, audio = L.outgoing_prompt(
        {"video_prompt": "<小明>在<<<image_1>>>前", "strategy": "ref2v"},
        entry, slots, {"小明": "蓝夹克男子"}, zh=True)
    assert audio and "无背景音乐" in p
    assert "<小明>" not in p and "<<<image_2>>>" in p     # 名字终换
    # 清单外编号 → 弃用,落剧本兜底(剥 Shot 前缀)
    p2, _ = L.outgoing_prompt(
        {"video_prompt": "看<<<image_9>>>", "strategy": "ref2v"},
        entry, slots, {}, zh=True)
    assert "<<<image_9>>>" not in p2 and "拿起饭团" in p2


def test_fallback_on_bad_policy_reply(tmp_path):
    """策略回复不可解析 → via=fallback、reward 只剩 format 差异,
    组照样成型(loop 不因坏回复卡死)。"""
    class BadPolicy:
        def complete(self, prompt, temperature=None, max_tokens=None):
            return "我拒绝输出 JSON"
    run = tmp_path / "movie_bad"
    L.run_episode(task_text="深夜便利店的十分钟", run_dir=run,
                  frozen_llm=FakeFrozenLLM(), policy=BadPolicy(),
                  kling=FakeKling(), t2i=FakeT2I(), judges=FakeJudges(),
                  group=3, rl_temperature=0.9)
    recs = [json.loads(x) for x in
            (run / "rl_steps.jsonl").read_text().splitlines()]
    assert all(s["via"] == "fallback" and s["r_format"] == 0.0
               for r in recs for s in r["samples"])


def test_collector_aggregates_and_skips_unjudged(tmp_path):
    """收集器 = 纯聚合:reward 内联的组直通;缺 reward 的旧格式组
    响亮跳过。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "rl/collect"))
    import watch_online as W
    run = tmp_path / "movie_c"
    run.mkdir(parents=True)
    good = {"kind": "condition_group", "run": "movie_c", "shot_idx": 0,
            "label": "s", "group_size": 2, "menu": [], "context": {},
            "samples": [{"completion": "{}", "reward": 0.7},
                        {"completion": "{}", "reward": 0.3}]}
    bad = {**good, "samples": [{"completion": "{}"},
                               {"completion": "{}"}]}
    (run / "rl_steps.jsonl").write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    seen = set()
    out = W.collect_run(run, seen)
    assert len(out) == 1 and out[0]["samples"][0]["reward"] == 0.7
    assert len(seen) == 2                     # 坏组也记书签,不反复读
