"""rl/env agent loop 单测(2026-08-19 用户令:训练=生产完全同构后,
针对全保真 driver;桩件同时被 run_grpo.sh --smoke 复用)。零 API:
LLM/生成器/判官/VLM/图像编辑全打桩,走完 §A0→§E 全流程。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl"))

from env import loop as L                                     # noqa: E402


# ── 桩件(--smoke 复用)────────────────────────────────────────────
_SB = {"cast": {"小明": "static: 蓝夹克短发男子; dynamic: none"},
       "setting": "深夜便利店,冷白灯光",
       "shots": [
           {"description": "Shot 1: <小明>推门进入便利店",
            "duration_s": 5, "end_state": "小明站在货架前,镜头静止",
            "variation": "medium", "opening_frame": "便利店门口静景",
            "bg": "bg_1"},
           {"description": "Shot 2: <小明>拿起饭团走向收银台",
            "duration_s": 5, "end_state": "小明到达收银台",
            "variation": "medium", "bg": "bg_1",
            "dialogue": {"speaker": "小明", "line": "就这个吧"}},
           {"description": "Shot 3: <小明>走出店门", "duration_s": 4,
            "end_state": "门关上", "variation": "large", "bg": "bg_2"}]}


class FakeFrozenLLM:
    def complete(self, prompt, temperature=None, max_tokens=None):
        if '"characters"' in prompt[-2000:]:      # character_extract
            return json.dumps({"characters": _SB["cast"]},
                              ensure_ascii=False)
        if '"cast"' in prompt[-3000:]:            # scene_write
            return json.dumps(_SB, ensure_ascii=False)
        # scene_image / 肖像翻译等 → 任意文本(调用方有确定性兜底)
        return "empty convenience store interior, cold white light"


class FakePolicy:
    def __init__(self):
        self.calls = []                            # (kind, temperature)

    def complete(self, prompt, temperature=None, max_tokens=None):
        if '"prev_end_cast"' in prompt:            # 交界人物判官
            return json.dumps({"prev_end_cast": ["小明"],
                               "cur_open_cast": ["小明"],
                               "reason": "同人"}, ensure_ascii=False)
        if '"view"' in prompt[-200:]:              # pick_space_view
            return json.dumps({"view": "master"})
        if "first_shot_desc" in prompt:            # 缝合师 → 判死退模板
            return "no"
        if "single_first_frame" in prompt:         # image plan
            self.calls.append(("plan", temperature))
            return json.dumps({"strategy": "none", "reason": "test"})
        self.calls.append(("cond", temperature))   # generation-condition
        strat = "ref2v" if '"ref2v"' in prompt else "t2v"
        return json.dumps(
            {"strategy": strat, "reason": "test",
             "video_prompt": "<小明>在<<<image_1>>>所示空间行动"},
            ensure_ascii=False)


class FakeT2I:
    def text_to_image(self, prompt, out, seed=0):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"\x89PNG fake" * 8)
        return Path(out)


class FakeKling:
    """生产 BailianKlingClient 的接口面桩件。"""

    def __init__(self):
        self.generate_audio = False
        self._t2i = FakeT2I()

    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_images",
                "first_frame_plus_refs", "t2i"}

    @staticmethod
    def ref_token(n):
        return f"<<<image_{n}>>>"

    def text_to_image(self, prompt, out, seed=0):
        return self._t2i.text_to_image(prompt, out, seed=seed)

    def generate(self, prompt, duration, out_path, fps=None, seed=None,
                 first_frame=None, reference_images=None,
                 reference_video=None):
        if reference_video is not None:
            raise RuntimeError("no reference-video channel")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake mp4 bytes " * 200)
        return Path(out_path)

    def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                       duration=None, seed=None, reference_images=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake mp4 bytes " * 200)
        return Path(out_path)


class FakeImageEdit:
    def edit(self, src, prompt, out, references=None):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"\x89PNG fake" * 8)
        return Path(out)


class FakeVLM:
    def caption_image(self, image_path):
        return "background: 测试图注,货架与收银台"


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


def _episode(tmp_path, group=4, policy=None):
    run = tmp_path / "movie_test"
    pol = policy or FakePolicy()
    res = L.run_episode(
        task_text="深夜便利店的十分钟",
        screenplay="深夜便利店。小明推门进店,拿起饭团说:\"就这个吧\","
                   "结账后走出店门。",
        run_dir=run, frozen_llm=FakeFrozenLLM(), policy=pol,
        video_gen=FakeKling(), image_edit=FakeImageEdit(),
        mllm=FakeVLM(), judges=FakeJudges(), group=group,
        rl_temperature=0.9)
    recs = [json.loads(x) for x in
            (run / "rl_steps.jsonl").read_text().splitlines()]
    return run, pol, recs, res


def test_group_sampling_and_temperatures(tmp_path):
    """K 组采样:v0 默认温度(None),其余带 rl 温度;image plan 单采。"""
    _run, pol, recs, _res = _episode(tmp_path)
    assert len(recs) == 3
    assert all(r["group_size"] == 4 and len(r["samples"]) == 4
               for r in recs)
    cond_temps = [t for k, t in pol.calls if k == "cond"][:4]
    assert cond_temps == [None, 0.9, 0.9, 0.9]
    assert sum(1 for k, _ in pol.calls if k == "plan") == 3  # 每镜一次


def test_record_schema_and_trunk(tmp_path):
    """记录自包含 + degraded_from 字段回归 + 主干 = reward argmax。"""
    _run, _pol, recs, _res = _episode(tmp_path)
    g = recs[1]
    for f in ("kind", "run", "shot_idx", "label", "junction_kind",
              "policy_version", "group_size", "menu", "context",
              "samples"):
        assert f in g, f
    s0 = g["samples"][0]
    for f in ("decision_id", "via", "completion", "raw", "usable",
              "strategy", "degraded_from", "final_prompt", "video",
              "chosen", "reward", "r_format", "r_text", "r_video",
              "video_detail", "dropped_components"):
        assert f in s0, f
    assert sum(1 for s in g["samples"] if s["chosen"]) == 1
    assert s0["chosen"] is True                    # 桩排名给 c0 最高
    comp = json.loads(s0["completion"])
    assert set(comp) == {"strategy", "reason", "video_prompt"}
    ctx = g["context"]
    assert set(ctx) == {"shot", "prompt_language", "prev_shot",
                        "junction", "cast", "setting", "cast_in_shot",
                        "slots_by_strategy", "storyboard",
                        "episode_guidance"}


def test_junction_fusion_routing(tmp_path):
    """三叉分诊(生产同构):同人同景 → derive(桩视频派生必败)→
    退 continue;换景 → cut;非首镜菜单锁 ref2v。"""
    run, _pol, recs, _res = _episode(tmp_path)
    assert [r["junction_kind"] for r in recs] == [None, "continue",
                                                  "cut"]
    assert [m["name"] for m in recs[1]["menu"]] == ["ref2v"]
    sb = json.loads((run / "storyboard.json").read_text())
    jm1 = sb["entries"][1]["junction_meta"]
    assert jm1["kind"] == "continue" \
        and jm1.get("fallback_to") == "continue"   # 派生失败留痕
    # 空间圣经:视图注册表已建(image_edit 兜底路),含 master
    assert "master" in (sb.get("spaces") or {}).get("bg_1", {})


def test_fallback_on_bad_policy_reply(tmp_path):
    """策略回复不可解析 → via=fallback、r_format=0,组照样成型。"""
    class BadPolicy:
        def complete(self, prompt, temperature=None, max_tokens=None):
            return "我拒绝输出 JSON"
    _run, _pol, recs, _res = _episode(tmp_path, group=3,
                                      policy=BadPolicy())
    assert recs and all(
        s["via"] == "fallback" and s["r_format"] == 0.0
        for r in recs for s in r["samples"])


def test_collector_aggregates_and_skips_unjudged(tmp_path):
    """收集器 = 纯聚合:reward 内联的组直通;缺 reward 的旧格式组
    响亮跳过。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "rl/collect"))
    import watch_online as Wc
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
    out = Wc.collect_run(run, seen)
    assert len(out) == 1 and out[0]["samples"][0]["reward"] == 0.7
    assert len(seen) == 2
