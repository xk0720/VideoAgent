"""RL 组采样(2026-08-10 用户令,semi-online GRPO):主干+单步分支。

断言:同 state 采 K 个条件决策(首个默认温度,其余带温);每变体各
生成一候选;组记录落 rl_steps.jsonl(decision_id/score/chosen);
主干 = 评审胜者;rl_group=0 时行为与旧版完全一致(单决策)。
"""
import json
import pathlib
from pathlib import Path

from maestro.agents.generator import GeneratorAgent
from maestro.agents.orchestrator import OrchestratorAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.models.llm import BaseLLMClient
from maestro.models.video_gen import MockVideoGenClient
from maestro.pipeline.window_loop import generate_movie_windowed


class _SamplingLLM(BaseLLMClient):
    """条件决策桩:逐次轮换策略(模拟温度采样的多样性);记录每次
    调用的 temperature 入参。"""

    _STRATS = ["i2v_keyframe", "t2v", "i2v_keyframe"]

    def __init__(self):
        self.cond_calls = []

    def complete(self, prompt: str, **kwargs) -> str:
        # 分支判据用【菜单内容】(prompt 里 menu JSON 必然带策略名):
        # image-plan 菜单含 single_first_frame;条件菜单含 t2v/i2v 等。
        if '"name": "single_first_frame"' in prompt:
            return json.dumps({
                "strategy": "single_first_frame",
                "images": [{"source": "t2i",
                            "description": "opening frame"}],
                "reason": "stub"})
        if '"name": "t2v"' in prompt:
            k = len(self.cond_calls)
            self.cond_calls.append(kwargs.get("temperature"))
            return json.dumps({
                "strategy": self._STRATS[k % len(self._STRATS)],
                "video_prompt": f"variant {k}: the cat walks",
                "reason": "stub"})
        return "not json"


class _VG(MockVideoGenClient):
    def __init__(self):
        super().__init__(name="mock-rl-gen")
        self.calls = []

    def capabilities(self):
        return {"t2v", "i2v", "t2i"}

    def text_to_image(self, prompt, out_path, seed=0):
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"PNG-STUB")
        return out_path

    def generate(self, prompt, duration, out_path, fps=8,
                 first_frame=None, reference_images=None, seed=0,
                 reference_video=None):
        self.calls.append({"prompt": prompt, "seed": seed})
        return super().generate(prompt, duration, out_path, fps=fps,
                                first_frame=first_frame, seed=seed)


def _components(video_gen):
    board = ReviewBoard([SemanticCritic(), PhysicsCritic(),
                         ConsistencyCritic(), RhythmCritic()])
    gen = GeneratorAgent(video_gen=video_gen)
    orch = OrchestratorAgent(generator=gen)
    return dict(board=board, generator=gen, refiner=RefinerAgent(),
                verifier=VerifierAgent(), orchestrator=orch)


def test_rl_group_samples_and_records(tmp_path, monkeypatch):
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    monkeypatch.setenv("MAESTRO_POLICY_VERSION", "7")
    vg = _VG()
    llm = _SamplingLLM()
    res = generate_movie_windowed(
        "a cat walks across the room", cache_dir=tmp_path,
        llm=llm, max_turns=1, n_candidates=1,
        rl_group=3, rl_temperature=0.8,
        **_components(vg))
    # 每镜:1 个默认温度 + 2 个带温采样
    n_shots = len(res.storyboard.entries)
    assert len(llm.cond_calls) == 3 * n_shots
    per_shot = llm.cond_calls[:3]
    assert per_shot[0] is None                # 首变体默认温度
    assert per_shot[1] == 0.8 and per_shot[2] == 0.8
    # 组记录落盘
    rec_path = tmp_path / "rl_steps.jsonl"
    assert rec_path.exists()
    recs = [json.loads(l) for l in rec_path.read_text().splitlines()]
    assert len(recs) == n_shots
    g = recs[0]
    assert g["kind"] == "condition_group"
    assert g["group_size"] == 3
    assert g["policy_version"] == "7"
    assert len(g["samples"]) == 3
    assert sum(1 for s_ in g["samples"] if s_["chosen"]) == 1
    for s_ in g["samples"]:
        assert s_["decision_id"]
        assert "weighted_total" in s_
    # 变体策略确实多样(轮换桩:两种策略都出现过)
    strats = {s_["strategy"] for s_ in g["samples"]}
    assert strats == {"i2v_keyframe", "t2v"}


def test_rl_group_off_is_legacy_behavior(tmp_path, monkeypatch):
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    vg = _VG()
    llm = _SamplingLLM()
    generate_movie_windowed(
        "a cat walks across the room", cache_dir=tmp_path,
        llm=llm, max_turns=1, n_candidates=1,
        **_components(vg))
    assert not (tmp_path / "rl_steps.jsonl").exists()
    assert all(t is None for t in llm.cond_calls)   # 无带温采样
