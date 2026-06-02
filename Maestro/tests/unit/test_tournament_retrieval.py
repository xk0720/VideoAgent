from pathlib import Path

from maestro.critics.tournament import Tournament
from maestro.tools.retrieval_tool import RetrievalTool
from maestro.types import AssetMemory, CandidateClip, Identity, ShotSpec, StyleRef


def _clip(idx, total):
    c = CandidateClip(shot_idx=idx, video_path=Path("x.mp4"))
    c.metric_scores = {"weighted_total": total}
    return c


def test_tournament_picks_highest_and_is_debiased():
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="p")
    cands = [_clip(0, 0.3), _clip(0, 0.9), _clip(0, 0.5)]
    best = Tournament().select(cands, spec)
    assert best.metric_scores["weighted_total"] == 0.9


def test_tournament_single_candidate():
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="p")
    only = _clip(0, 0.4)
    assert Tournament().select([only], spec) is only


def test_retrieval_identity_and_style_refs():
    mem = AssetMemory(
        identity_anchors={"id_a": Identity("id_a", source="/data/a.png")},
        style_anchors=[StyleRef("style_a", source="/data/a.png")],
    )
    tool = RetrievalTool(mem)
    assert tool.retrieve_identity_refs(["id_a"]) == [Path("/data/a.png")]
    assert tool.retrieve_identity_refs(["missing"]) == []
    assert tool.retrieve_style_refs(["style_a"]) == [Path("/data/a.png")]


def test_retrieval_source_shots_semantic():
    from maestro.embeddings import embed_text
    from maestro.types import Shot

    mem = AssetMemory(video_shots={
        "s0": Shot("s0", "v.mp4", 0, 2, caption="a car chase in the city",
                   clip_embedding=embed_text("a car chase in the city")),
        "s1": Shot("s1", "v.mp4", 2, 4, caption="a quiet beach at sunset",
                   clip_embedding=embed_text("a quiet beach at sunset")),
    })
    hits = RetrievalTool(mem).retrieve_source_shots("car chase city", top_k=1)
    assert hits == ["s0"]
