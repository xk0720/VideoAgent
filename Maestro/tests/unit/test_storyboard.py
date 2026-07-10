"""StoryboardMemory (R1) — 台账构建/更新/持久化/brain 视图。CPU-only。"""
import json

from maestro.memory.storyboard import StoryboardMemory


OUTLINE = [
    "Shot 1: scene 1 a glass sits on a table",
    "Shot 2: scene 1 the glass falls off the edge",
    "Shot 3: scene 2 shards scatter on the kitchen floor",
]


def test_from_outline_orders_and_labels_scene_shot(tmp_path):
    sb = StoryboardMemory.from_outline(OUTLINE, path=tmp_path / "sb.json")
    assert [e.shot_idx for e in sb.entries] == [0, 1, 2]
    assert sb.entries[0].label == "scene 1 shot 1"
    assert sb.entries[1].label == "scene 1 shot 2"
    assert sb.entries[2].label == "scene 2 shot 1"     # 场号从文本解析
    assert all(e.status == "pending" for e in sb.entries)


def test_updates_persist_and_reviews_append(tmp_path):
    p = tmp_path / "sb.json"
    sb = StoryboardMemory.from_outline(OUTLINE, path=p)
    sb.set_keyframe(0, tmp_path / "kf0.png", "t2i")
    sb.set_condition(0, {"strategy": "i2v_keyframe"})
    sb.add_review(0, {"revision": 0, "weighted_total": 0.4, "n_failed": 2})
    sb.add_review(0, {"revision": 1, "weighted_total": 0.6, "n_failed": 0})
    sb.set_result(0, tmp_path / "v0.mp4", converged=True,
                  repair_actions=[{"tool": "edit_clip", "outcome": "accepted"}])

    loaded = StoryboardMemory.load(p)                 # 崩溃后可恢复
    e = loaded.get(0)
    assert e.status == "verified"
    assert e.keyframe_source == "t2i"
    assert len(e.reviews) == 2                        # 追加式,评审轨迹不覆盖
    assert e.reviews[0]["weighted_total"] == 0.4
    assert e.repair_actions[0]["tool"] == "edit_clip"


def test_next_pending_and_prev_generated_drive_the_window(tmp_path):
    sb = StoryboardMemory.from_outline(OUTLINE, path=tmp_path / "sb.json")
    assert sb.next_pending().shot_idx == 0
    sb.set_result(0, tmp_path / "v0.mp4", converged=False)   # 未收敛也算已生成
    assert sb.next_pending().shot_idx == 1
    # 上一镜 = 最近【已生成】(哪怕带缺陷),它的尾帧才是正确续接点
    assert sb.prev_generated(1).shot_idx == 0
    assert sb.prev_generated(0) is None
    sb.set_result(1, tmp_path / "v1.mp4", converged=True)
    sb.set_result(2, tmp_path / "v2.mp4", converged=True)
    assert sb.next_pending() is None and sb.all_generated()


def test_brain_json_is_compact_and_honest(tmp_path):
    sb = StoryboardMemory.from_outline(OUTLINE, path=tmp_path / "sb.json")
    sb.add_review(1, {"weighted_total": 0.3, "n_failed": 3})
    sb.set_result(1, tmp_path / "v1.mp4", converged=False)
    rows = sb.to_brain_json()
    assert len(rows) == 3
    assert rows[1]["status"] == "generated_with_defects"   # 永不谎报 verified
    assert rows[1]["last_score"] == 0.3
    assert rows[1]["open_defects"] == 3
    # 大对象(评审全文/修复流水)不进 brain 视图
    assert "reviews" not in rows[1] and "repair_actions" not in rows[1]


def test_atomic_save_file_is_valid_json(tmp_path):
    p = tmp_path / "sb.json"
    sb = StoryboardMemory.from_outline(OUTLINE, path=p)
    for i in range(3):
        sb.set_result(i, tmp_path / f"v{i}.mp4", converged=True)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["rev"] >= 4 and len(data["entries"]) == 3
