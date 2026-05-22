"""EditorEnv + ContextManager smoke tests."""
from __future__ import annotations

from training.env.context_manager import ContextManager


def test_context_manager_bounded_history():
    cm = ContextManager(max_turns=3)
    from training.env.base import EnvObservation
    for i in range(5):
        cm.push(action={"a": i}, observation=EnvObservation(state={"i": i}),
                reward=float(i))
    assert len(cm) == 3
    msgs = cm.to_messages("you are an agent")
    assert msgs[0]["role"] == "system"
    # 3 (action,obs) pairs = 6 messages + 1 system = 7
    assert len(msgs) == 7


def test_editor_env_reset_and_step(editor_env_factory):
    env = editor_env_factory()
    obs = env.reset()
    assert "segment_idx" in obs.state
    assert "retrieve" in obs.available_actions
    res = env.step({"action": "retrieve", "rationale": "test"})
    assert -1.0 < res.reward < 2.0
    assert "step" in res.info


def test_editor_env_unknown_action_penalised(editor_env_factory):
    env = editor_env_factory()
    env.reset()
    res = env.step({"action": "totally-invalid", "rationale": "x"})
    assert res.reward < 0
    assert "error" in res.info


def test_editor_env_terminates_on_acceptance(editor_env_factory):
    env = editor_env_factory()
    env.reset()
    terminated = False
    for _ in range(5):
        res = env.step({"action": "retrieve", "rationale": ""})
        if res.terminated or res.truncated:
            terminated = True
            break
    assert terminated, "EditorEnv must terminate within max_steps"


def test_editor_env_exposes_ors_tools(editor_env_factory):
    env = editor_env_factory()
    tools = env.tools()
    names = {t["function"]["name"] for t in tools}
    assert names == {"retrieve", "generate", "fallback"}
