"""EpisodeMemory (R2) — good/bad 蒸馏、检索、可执行 guidance。CPU-only。"""
from maestro.memory.episode_memory import EpisodeMemory
from maestro.memory.storyboard import StoryboardMemory


def _finished_storyboard(tmp_path, converged=(True, True)):
    sb = StoryboardMemory.from_outline(
        ["Shot 1: a glass falls off a table",
         "Shot 2: shards scatter on the floor"],
        path=tmp_path / "sb.json")
    for i, ok in enumerate(converged):
        sb.set_keyframe(i, tmp_path / f"kf{i}.png", "t2i")
        sb.set_condition(i, {"strategy": "flf2v_bridge" if i else "i2v_keyframe"})
        sb.add_review(i, {"weighted_total": 0.7 if ok else 0.3,
                          "n_failed": 0 if ok else 2,
                          "brief_headline": "" if ok else "gravity violated"})
        sb.set_result(i, tmp_path / f"v{i}.mp4", converged=ok,
                      repair_actions=[{"tool": "edit_clip",
                                       "outcome": "accepted" if ok else "rejected"}])
    return sb


def test_good_episode_all_strategies_into_replay(tmp_path):
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    rec = em.distill_episode("a glass falls off a table",
                             _finished_storyboard(tmp_path))
    assert rec.outcome == "good"
    assert len(rec.replay) == 2 and not rec.avoid
    assert rec.replay[1]["condition_strategy"] == "flf2v_bridge"
    assert rec.repair_tool_stats["edit_clip"] == [2, 0]


def test_bad_episode_splits_replay_and_avoid(tmp_path):
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    rec = em.distill_episode("a glass falls off a table",
                             _finished_storyboard(tmp_path, (True, False)))
    assert rec.outcome == "bad"
    # 收敛的 shot 策略仍进 replay(好的局部经验不陪葬);失败的进 avoid 带原因
    assert len(rec.replay) == 1 and rec.replay[0]["label"] == "scene 1 shot 1"
    assert len(rec.avoid) == 1
    assert rec.avoid[0]["condition_strategy"] == "flf2v_bridge"
    assert "gravity" in rec.avoid[0]["reason"]


def test_retrieve_by_keyword_overlap_and_persistence(tmp_path):
    p = tmp_path / "ep.jsonl"
    em = EpisodeMemory(p)
    em.distill_episode("a glass falls off a table",
                       _finished_storyboard(tmp_path))
    em.distill_episode("a rocket launches into space",
                       _finished_storyboard(tmp_path))
    em2 = EpisodeMemory(p)                             # 重开进程也能读回
    hits = em2.retrieve("the glass drops from the table edge")
    assert hits and "glass" in hits[0].user_prompt
    assert not em2.retrieve("完全无关的水下城市 neon jellyfish")


def test_guidance_is_executable_replay_plus_avoid(tmp_path):
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    em.distill_episode("a glass falls off a table",
                       _finished_storyboard(tmp_path, (True, False)))
    g = em.guidance_for("a glass falls from a shelf")
    assert g["n_episodes_matched"] == 1
    # bad episode:只有收敛 shot 进 replay 提示;失败策略进 avoid
    assert len(g["replay_hints"]) == 1 and g["replay_hints"][0]["converged"]
    assert g["avoid"] and g["avoid"][0]["condition_strategy"] == "flf2v_bridge"
    assert em.episodes[0].uses == 1                    # 使用记账
