"""EpisodeMemory (R2) — 技能轨迹蒸馏、检索、guidance。CPU-only。"""
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


def test_good_episode_trajectory_steps(tmp_path):
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    rec = em.distill_episode("a glass falls off a table",
                             _finished_storyboard(tmp_path))
    assert rec.outcome == "good"
    assert len(rec.trajectory) == 2
    st = rec.trajectory[1]
    assert st["action"]["strategy"] == "flf2v_bridge"
    assert st["feedback"]["converged"] is True
    assert st["feedback"]["score"] == 0.7
    assert rec.repair_tool_stats["edit_clip"] == [2, 0]


def test_bad_episode_feedback_carries_vlm_verdict(tmp_path):
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    rec = em.distill_episode("a glass falls off a table",
                             _finished_storyboard(tmp_path, (True, False)))
    assert rec.outcome == "bad"
    ok_step, bad_step = rec.trajectory
    assert ok_step["feedback"]["converged"] is True
    assert bad_step["feedback"]["converged"] is False
    assert "gravity" in bad_step["feedback"]["vlm_headline"]


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


def test_guidance_derived_from_trajectory(tmp_path):
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    em.distill_episode("a glass falls off a table",
                       _finished_storyboard(tmp_path, (True, False)))
    g = em.guidance_for("a glass falls from a shelf")
    assert g["n_episodes_matched"] == 1
    # bad 片:收敛步进 replay 提示;失败步进 avoid 带 VLM 评语
    assert len(g["replay_hints"]) == 1 and g["replay_hints"][0]["converged"]
    assert g["avoid"] and g["avoid"][0]["condition_strategy"] == "flf2v_bridge"
    assert "gravity" in g["avoid"][0]["reason"]
    assert g["past_task_shapes"][0]["n_shots"] == 2


def test_trajectory_references_resolve_in_registry(tmp_path):
    """2026-08-13 用户令:轨迹里的引用必须自解释 —— new_0 是什么、
    图注是什么,在 header.reference_registry 内闭环;--no-review 片
    判 ungraded、feedback 诚实留空。"""
    sb = StoryboardMemory.from_outline(
        ["scene 1 shot 1: <A>走进店里。", "shot 2: <A>付钱。"],
        path=tmp_path / "sb.json")
    sb.cast = {"A": "static: tall man"}
    sb.setting = "a bakery at dawn"
    sb.portraits = {"A": str(tmp_path / "a.png")}
    sb.backgrounds = {"bg_1": {"path": "plate.png", "src": "t2i"}}
    sb.spaces = {"bg_1": {
        "master": {"path": "plate.png", "src": "t2i", "caption": "店堂全景"},
        "new_0": {"path": "y.png", "src": "frame",
                  "caption": "从入口反打:柜台居右,木架烤炉在后"}}}
    stub = {"revision": 0, "weighted_total": 0.0, "n_failed": 0,
            "review_evidence": {"checklist_items": 0},
            "stop_reason": "review_disabled"}
    for i, e in enumerate(sb.entries):
        e.video_path = f"s{i}.mp4"
        e.status = "generated_with_defects"
        e.reviews = [dict(stub)]
        e.bg_id = "bg_1"
        e.camera_facing = "朝柜台,中景"
        e.draft_prompt = "草稿全文"
        e.condition = {"strategy": "ref2v", "decided_strategy": "ref2v",
                       "final_prompt": "终稿全文",
                       "reference_images": [str(tmp_path / "a.png"),
                                            "y.png"]}
        e.junction_meta = {"kind": "derive",
                           "space_view": {"view": "new_0"},
                           "stitcher": {"via": "agent"}}
    mem = EpisodeMemory(tmp_path / "ep.jsonl")
    rec = mem.distill_episode("面包店的清晨", sb)
    assert rec.outcome == "ungraded"
    st = rec.trajectory[0]
    # 引用图例 → registry 键 → 图注可查(new_0 在记录内自解释)
    assert st["action"]["refs"] == {"image_1": "portrait:A",
                                    "image_2": "space_view:bg_1/new_0"}
    assert st["context"]["junction"]["space_view"] == "space_view:bg_1/new_0"
    reg = rec.header["reference_registry"]
    assert reg["space_view:bg_1/new_0"]["caption"].startswith("从入口反打")
    assert reg["portrait:A"]["desc"] == "static: tall man"
    # prompt 全文、feedback 诚实留空
    assert st["action"]["prompt"] == "终稿全文"
    assert st["feedback"]["score"] is None
    assert st["feedback"]["converged"] is None
    # ungraded 的步照进 replay 参谋
    g = mem.guidance_for("清晨的面包店故事")
    assert g["n_episodes_matched"] == 1 and g["replay_hints"]
    assert not g["avoid"]
