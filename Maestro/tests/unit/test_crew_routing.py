"""ViMax 分工路由(2026-08-06 用户令):剧本扩写/分镜/视频 brain 各持
独立 LLM agent 实例——窗口管线必须把对应调用发给对应实例。"""
from pathlib import Path

from maestro.pipeline.window_loop import generate_movie_windowed

from test_window_loop import (_JsonLLM, _WindowVideoGen,  # noqa: E402
                              _components)


class _TaggedLLM(_JsonLLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def complete(self, prompt: str, **kw):
        self.prompts.append(prompt)
        return super().complete(prompt, **kw)


def test_crew_calls_route_to_their_agents(tmp_path, monkeypatch):
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    sw, sc, vb, default = (_TaggedLLM(), _TaggedLLM(), _TaggedLLM(),
                           _TaggedLLM())
    generate_movie_windowed(
        "a glass falls off a table", cache_dir=tmp_path, llm=default,
        crew={"screenwriter": sw, "scene_writer": sc, "video_brain": vb},
        max_turns=1, n_candidates=1, **_components(_WindowVideoGen()))
    # 分镜调用(Scene Write 技能全文开头)必须落在 scene_writer 实例
    assert any("Scene Write" in p[:300] for p in sc.prompts)
    assert not any("Scene Write" in p[:300] for p in vb.prompts)
    # 视频 brain 调用(Image Plan / Window Generation)必须落在 video_brain
    assert any("Image Plan" in p[:300] or "Window Generation" in p[:300]
               for p in vb.prompts)
    assert not any("Image Plan" in p[:300] for p in sc.prompts)
    # 三个实例互为独立对象
    assert len({id(sw), id(sc), id(vb), id(default)}) == 4
