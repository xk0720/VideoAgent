"""cinegraph 全链冒烟:全假后端(真 mp4/png 工件)跑通
剧本→分镜→机位树→注册表→首帧派生→逐镜视频→拼装,接线错误一网打尽。"""
import json
import subprocess
from pathlib import Path

import pytest

from maestro.cinegraph import generate_movie_cinegraph
from maestro.types import AssetMemory

_SCREENPLAY = ("雨夜暗巷。<黑帮老大>在车内点燃雪茄,雨滴敲击车窗声。"
               "<女子>神情恐惧,低沉心跳声。"
               "黑帮老大说:“今晚到此为止。”")


def _mk_mp4(out: Path, color: str = "gray", secs: float = 0.3):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={color}:s=160x90:d={secs}", str(out)], check=True)
    return out


def _mk_png(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=160x90:d=0.1", "-frames:v", "1",
         str(out)], check=True)
    return out


class _LLM:
    """按调用意图路由的多面假 brain。"""

    def complete(self, prompt, **kw):
        # 路由按【独有标记】判,顺序从特异到宽泛(scene_write prompt 里
        # 也含 characters/static:,宽泛判据会劫走调用 —— 冒烟实锤)。
        if "camera position tree" in prompt:
            return json.dumps({"camera_parent_items": [
                {"parent_cam_idx": None, "parent_shot_idx": None,
                 "reason": "root"},
                {"parent_cam_idx": 0, "parent_shot_idx": 0,
                 "reason": "close-up inside wide",
                 "is_parent_fully_covers_child": False,
                 "missing_info": "front face"}]})
        if "ref_image_indices" in prompt:
            return json.dumps({"ref_image_indices": [0],
                               "text_prompt": "参考 Image 0 生成该帧。"})
        if '"ok": true|false' in prompt:
            return json.dumps({"ok": True, "reason": "fine"})
        if "music_plan" in prompt:
            return json.dumps({"cast": {
                "黑帮老大": "static: tall; dynamic: suit",
                "女子": "static: slim; dynamic: dress"},
                "setting": "rainy alley at night",
                "shots": [
                    {"description": "Shot 1: 大远景,<黑帮老大>在车内点燃"
                                    "雪茄,雨滴敲击车窗声。",
                     "duration_s": 5, "end_state": "<黑帮老大>静止。",
                     "variation": "small", "camera": 0,
                     "opening_frame": "<黑帮老大>在车内。",
                     "dialogue": "", "bg": "bg_1"},
                    {"description": "Shot 2: 特写,<女子>神情恐惧,"
                                    "低沉心跳声。",
                     "duration_s": 5, "end_state": "<女子>静止。",
                     "variation": "small", "camera": 1,
                     "opening_frame": "<女子>特写。",
                     "dialogue": {"speaker": "黑帮老大",
                                  "line": "今晚到此为止。"},
                     "bg": "bg_1"}],
                "music_plan": {}}, ensure_ascii=False)
        if "characters" in prompt and "static:" in prompt:
            return json.dumps({"characters": {
                "黑帮老大": "static: tall; dynamic: suit",
                "女子": "static: slim; dynamic: dress"}},
                ensure_ascii=False)
        return "{}"


class _MLLM:
    def caption_identity(self, p):
        return "该角色穿深色西装。"

    def caption_image(self, p):
        return "画面与描述一致。"


class _VG:
    def __init__(self):
        self.generate_audio = False
        self.calls = []

    def capabilities(self):
        return {"t2i", "i2v", "flf2v", "ref_images"}

    def ref_token(self, n):
        return f"<<<image_{n}>>>"

    def text_to_image(self, prompt, out):
        return _mk_png(Path(out))

    def generate(self, prompt, duration, out_path, fps=24, seed=0,
                 first_frame=None, reference_images=None, **kw):
        self.calls.append({"kind": "generate", "prompt": prompt,
                           "first_frame": str(first_frame),
                           "refs": [str(r) for r in
                                    (reference_images or [])],
                           "audio": self.generate_audio})
        return _mk_mp4(Path(out_path))

    def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                       duration=5, seed=0):
        self.calls.append({"kind": "flf2v", "prompt": prompt})
        return _mk_mp4(Path(out_path))


class _Edit:
    def edit(self, keyframe, instruction, out_path, references=None):
        return _mk_png(Path(out_path))


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"],
                                   capture_output=True).returncode != 0,
                    reason="ffmpeg required")
def test_cinegraph_end_to_end_smoke(tmp_path):
    vg = _VG()
    res = generate_movie_cinegraph(
        "雨夜短片", cache_dir=tmp_path, llm=_LLM(), mllm=_MLLM(),
        video_gen=vg, image_edit=_Edit(), board=None,
        asset_memory=AssetMemory(), screenplay=_SCREENPLAY,
        given_characters=None, max_turns=1, enable_audio=True)
    # 成片存在
    assert res.movie_path and Path(res.movie_path).exists()
    # 机位树:1 号机位挂 0 号根
    assert res.cameras[0].parent_cam_idx is None
    assert res.cameras[1].parent_cam_idx == 0
    # 帧工件齐(镜 0/1 各有 ff)
    assert (tmp_path / "frames" / "shot000_ff.png").exists()
    assert (tmp_path / "frames" / "shot001_ff.png").exists()
    # 派生链走过(双镜切视频存在)
    assert (tmp_path / "frames" / "two_shot_cam1.mp4").exists()
    # 逐镜视频调用:对白镜带原生音频 + 台词逐字 + 压制句
    gen_calls = [c for c in vg.calls if c["kind"] == "generate"
                 and "shot" in str(c.get("first_frame", ""))]
    dial = [c for c in gen_calls if "今晚到此为止" in c["prompt"]]
    assert dial and dial[0]["audio"] is True
    assert "无背景音乐" in dial[0]["prompt"]
    # 状态工件落盘
    state = json.loads((tmp_path / "cinegraph_state.json").read_text())
    assert len(state["cameras"]) == 2 and len(state["shots"]) == 2
