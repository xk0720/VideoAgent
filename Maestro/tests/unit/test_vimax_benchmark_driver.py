"""vimax_benchmark 测试链(2026-08-13):适配器结构保真 + 预分镜法在场。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_vimax_benchmark import adapt_story  # noqa: E402


def test_adapt_story_keeps_every_shot_verbatim():
    d = {"story_overview": "两人下棋。",
         "scenes": [
             {"scene_num": 1, "shots": [
                 {"shot_id": 1, "first_frame": "咖啡馆全景,砖墙上三幅画。",
                  "video_prompt": "镜头缓推,A落子。"},
                 {"shot_id": 2, "first_frame": "近景棋盘。",
                  "video_prompt": "B皱眉思考。"}]},
             {"scene_num": 2, "shots": [
                 {"shot_id": 3, "first_frame": "同一咖啡馆,雨夜窗景。",
                  "video_prompt": "A伸手握手。"}]}]}
    txt = adapt_story(d)
    assert txt.startswith("故事总览:两人下棋。")
    # 镜头结构显式:场景N 镜头M 逐条在场,原文一字不丢
    assert "场景1 镜头1:【开场画面】咖啡馆全景,砖墙上三幅画。【本镜动作】镜头缓推,A落子。" in txt
    assert "场景2 镜头3:" in txt
    assert txt.count("【开场画面】") == 3 == txt.count("【本镜动作】")


def test_scene_write_carries_prestoryboard_law():
    sw = " ".join(Path(
        "src/maestro/skills/brain_skills/scene_write/SKILL.md"
    ).read_text().split())
    assert "PRE-STORYBOARDED SCRIPT LAW" in sw
    assert "no merging, no splitting, no reordering" in sw
    assert "ANNOTATION only" in sw


def test_benchmark_dataset_shape():
    idx = json.loads(Path("vimax_benchmark/benchmark_index.json"
                          ).read_text())
    assert idx["total_stories"] == 35
    one = json.loads(
        (Path("vimax_benchmark") / idx["stories"][0]["file"]).read_text())
    sh = one["scenes"][0]["shots"][0]
    assert {"shot_id", "first_frame", "video_prompt"} <= set(sh)
