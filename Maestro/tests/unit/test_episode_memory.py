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


def test_distill_film_level_dossier(tmp_path):
    """2026-08-13 用户令:episode 是剧本级档案 —— 剧本形状/参考库
    构建/逐镜蓝图(junction/朝向/prompt/引用数)全落记录;--no-review
    的占位评审行判 ungraded(replay 照进、avoid 不进、0 分不记)。"""
    from maestro.memory.episode_memory import EpisodeMemory
    from maestro.memory.storyboard import StoryboardMemory

    sb = StoryboardMemory.from_outline(
        ["scene 1 shot 1: <A>走进店里。", "shot 2: <A>付钱。"],
        path=tmp_path / "sb.json")
    sb.cast = {"A": "static: tall man"}
    sb.setting = "a bakery at dawn"
    sb.portraits = {"A": str(tmp_path / "a.png")}
    sb.backgrounds = {"bg_1": {"path": "x.png", "src": "t2i"}}
    sb.spaces = {"bg_1": {"master": {"path": "x.png", "src": "t2i"},
                          "new_0": {"path": "y.png", "src": "frame"}}}
    stub = {"revision": 0, "weighted_total": 0.0, "n_failed": 0,
            "review_evidence": {"checklist_items": 0},
            "stop_reason": "review_disabled"}
    for i, e in enumerate(sb.entries):
        e.video_path = f"s{i}.mp4"
        e.status = "generated_with_defects"
        e.reviews = [dict(stub)]
        e.bg_id = "bg_1"
        e.camera_facing = "朝柜台,中景"
        e.draft_prompt = "草稿"
        e.condition = {"strategy": "ref2v", "decided_strategy": "ref2v",
                       "final_prompt": "终稿",
                       "reference_images": ["p1", "p2"]}
        e.junction_meta = {"kind": "derive",
                           "space_view": {"view": "new_0"},
                           "stitcher": {"via": "agent"}}
    mem = EpisodeMemory(tmp_path / "ep.jsonl")
    rec = mem.distill_episode("面包店的清晨", sb)
    assert rec.outcome == "ungraded"          # 占位评审 ≠ 评过审
    assert len(rec.replay) == 2 and not rec.avoid
    assert rec.replay[0]["final_score"] is None   # 0.0 占位分不记账
    assert rec.screenplay_digest["cast"]["A"].startswith("static")
    assert rec.screenplay_digest["bgs"]["bg_1"] == [0, 1]
    assert rec.asset_build["spaces"]["bg_1"]["n_frame_views"] == 1
    plan = rec.shot_plans[0]
    assert plan["junction"]["space_view"] == "new_0"
    assert plan["camera_facing"] == "朝柜台,中景"
    assert plan["n_references"] == 2
    assert plan["prompt_final"] == "终稿"
    # ungraded 的 replay 参谋进 guidance
    g = mem.guidance_for("清晨的面包店故事")
    assert g["n_episodes_matched"] == 1
    assert g["replay_hints"]
