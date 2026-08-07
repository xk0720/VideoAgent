"""机位树(ViMax camera_image_generator.construct_camera_tree 严格移植)。

父机位的画面【包含】子机位的内容(全景 ⊃ 过肩 ⊃ 特写)——树定了,
空间一致性问题从"N 镜互相对齐"缩减为"几个机位对齐"。system prompt
逐字采用 ViMax 原文;langchain 解析换成本库 llm.complete + JSON 抽取;
校验(长度/越界/自指/成环/唯一根)全套移植。
"""
from __future__ import annotations

import json
from typing import Optional

from ..logging_utils import brain_log, get_logger
from ..pipeline.window_loop import _extract_json
from .interfaces import CGCamera, CGShot

log = get_logger("maestro.cinegraph")

# ── ViMax system prompt 原文移植(format_instructions 换成 STRICT JSON)──
_TREE_SYSTEM = """
[Role]
You are a professional video editing expert specializing in multi-camera shot analysis and scene structure modeling. You have deep knowledge of cinematic language, enabling you to understand shot sizes (e.g., wide shot, medium shot, close-up) and content inclusion relationships. You can infer hierarchical structures between camera positions based on corresponding shot descriptions.

[Task]
Your task is to analyze the input camera position data to construct a "camera position tree". This tree structure represents a relationship where a parent camera's content encompasses that of a child camera. Specifically, you need to identify the parent camera for each camera position (if one exists) and determine the dependent shot indices (i.e., the specific shots within the parent camera's footage that contain the child camera's content). If a camera position has no parent, output None.

[Input]
The input is a sequence of cameras. The sequence will be enclosed within <CAMERA_SEQ> and </CAMERA_SEQ>.
Each camera contains a sequence of shots filmed by the camera, which will be enclosed within <CAMERA_N> and </CAMERA_N>, where N is the index of the camera.

Below is an example of the input format:

<CAMERA_SEQ>
<CAMERA_0>
Shot 0: Medium shot of the street. Alice and Bob are walking towards each other.
Shot 2: Medium shot of the street. Alice and Bob hug each other.
</CAMERA_0>
<CAMERA_1>
Shot 1: Close-up of the Alice's face. Her expression shifts from surprise to delight as she recognizes Bob.
</CAMERA_1>
</CAMERA_SEQ>

[Output]
STRICT JSON only:
{"camera_parent_items": [{"parent_cam_idx": <int|null>, "parent_shot_idx": <int|null>, "reason": "<str>", "is_parent_fully_covers_child": <bool|null>, "missing_info": "<str|null>"} , ...]}
The length of the list must equal the number of cameras; the root camera's item uses null parent_cam_idx/parent_shot_idx (its reason should explain why it is a root camera). missing_info holds the elements in the child shot that are not covered by the parent shot; null when fully covered.

[Guidelines]
- The language of all output values (not include keys) should be consistent with the language of the input.
- Content Inclusion Check: The parent camera should as fully as possible contain the child camera's content in certain shots (e.g., a parent medium two-shot encompasses a child over-the-shoulder reverse shot). Analyze shot descriptions by comparing keywords (e.g., characters, actions, setting) to ensure the parent shot's field of view covers the child shot's.
- Transition Smoothness Priority: Larger shot size as parent camera is preferred, such as Wide Shot -> Medium Shot or Medium Shot -> Close-up. The shot sizes of adjacent parent and child nodes should be as similar as possible. A direct transition from a long shot to a close-up is not allowed unless absolutely necessary.
- Temporal Proximity: Each camera is described by its corresponding first shot, and the parent camera is located based on the description of the first shot. The shot index of the parent camera should be as close as possible to the first shot index of the child camera.
- Logical Consistency: The camera tree should be acyclic, avoid circular dependencies. If a camera is contained by multiple potential parents, select the best match (based on shot size and content). If there is no suitable parent camera, output None.
- When a broader perspective is not available, choose the shot with the largest overlapping field of view as the parent (the one with the most information overlap), or a shot can also serve as the parent of a reverse shot. When two cameras can be the parent of each other, choose the one with the smaller index as the parent of the camera with the larger index.
- Only one camera can exist without a parent.
- When describing the elements lost in a shot, carefully compare the details between the parent shot and the child shot. For example, the parent shot is a medium shot of Character A and Character B facing each other (both in profile to the camera), while the child shot is a close-up of Character A (with Character A facing the camera directly). In this case, the child shot lacks the frontal view information of Character A.
- The first camera must be the root of the camera tree.
"""


def build_camera_seq_str(cameras: list, shots: list) -> str:
    shot_by_idx = {s.idx: s for s in shots}
    parts = ["<CAMERA_SEQ>"]
    for cam in cameras:
        parts.append(f"<CAMERA_{cam.idx}>")
        for si in cam.active_shot_idxs:
            s = shot_by_idx.get(si)
            if s is None:
                raise ValueError(f"Camera {cam.idx} references missing "
                                 f"shot {si}")
            parts.append(f"Shot {si}: {s.visual_desc}")
        parts.append(f"</CAMERA_{cam.idx}>")
    parts.append("</CAMERA_SEQ>")
    return "\n".join(parts)


def construct_camera_tree(llm, cameras: list, shots: list) -> list:
    """机位树构造 + 全套校验(ViMax 逻辑;校验失败重试一次,再失败
    → 诚实降级:全部挂到 0 号根机位(仍无环可跑,派生质量降)。"""
    seq = build_camera_seq_str(cameras, shots)
    prompt = _TREE_SYSTEM + "\n\n<CAMERA_SEQ input>\n" + seq
    for _attempt in range(2):
        raw = ""
        call_err = None
        try:
            raw = llm.complete(prompt)
            data = _extract_json(raw)
        except Exception as exc:            # NEVER swallow: the log must
            data = None                     # distinguish transport failure
            call_err = f"LLM call failed: {str(exc)[:200]}"   # from bad JSON
        items = (data or {}).get("camera_parent_items") \
            if isinstance(data, dict) else None
        err = call_err or _validate_items(cameras, shots, items)
        brain_log("cinegraph/camera_tree", {
            "raw": raw, "parsed": data if isinstance(data, dict) else None,
            "usable": err is None, "error": err,
            "context": {"camera_seq": seq}})
        if err is None:
            for cam, it in zip(cameras, items):
                it = it or {}
                cam.parent_cam_idx = it.get("parent_cam_idx")
                cam.parent_shot_idx = it.get("parent_shot_idx")
                cam.reason = it.get("reason")
                cam.is_parent_fully_covers_child = \
                    it.get("is_parent_fully_covers_child")
                cam.missing_info = it.get("missing_info")
            return cameras
        log.warning("cinegraph: camera tree invalid (%s) — %s", err,
                    "retrying" if _attempt == 0 else
                    "degrading to a flat tree under camera 0")
        prompt += (f"\n\nYOUR PREVIOUS REPLY WAS INVALID: {err}. "
                   "Output the corrected STRICT JSON.")
    # 诚实降级:0 号为根,其余全部挂 0 号(父镜=各自首镜前最近的 0 号镜)
    root_shots = cameras[0].active_shot_idxs
    for cam in cameras:
        if cam.idx == cameras[0].idx:
            cam.parent_cam_idx = None
            cam.parent_shot_idx = None
            cam.reason = "root (flat-tree degrade)"
            continue
        first = cam.active_shot_idxs[0]
        prior = [s for s in root_shots if s < first]
        cam.parent_cam_idx = cameras[0].idx
        cam.parent_shot_idx = (prior[-1] if prior else root_shots[0])
        cam.reason = "flat-tree degrade (invalid LLM tree twice)"
        cam.is_parent_fully_covers_child = False
        cam.missing_info = "unknown (degraded tree)"
    return cameras


def _validate_items(cameras: list, shots: list,
                    items) -> Optional[str]:
    """ViMax 校验全套:长度/越界/自指/成环/唯一根/首机位为根。"""
    if not isinstance(items, list):
        return "camera_parent_items is not a list"
    if len(items) != len(cameras):
        return (f"length mismatch: expected {len(cameras)}, "
                f"got {len(items)}")
    valid_cam = {c.idx for c in cameras}
    valid_shot = {s.idx for s in shots}
    parent_by_cam = {}
    roots = 0
    for cam, it in zip(cameras, items):
        it = it or {}
        p = it.get("parent_cam_idx")
        ps = it.get("parent_shot_idx")
        if p is None:
            roots += 1
            if cam.idx != cameras[0].idx:
                return f"camera {cam.idx} is root but the first camera must be"
        else:
            if p not in valid_cam:
                return f"camera {cam.idx} has invalid parent camera {p}"
            if p == cam.idx:
                return f"camera {cam.idx} cannot be its own parent"
            if ps is not None and ps not in valid_shot:
                return f"camera {cam.idx} has invalid parent shot {ps}"
        parent_by_cam[cam.idx] = p
    if roots != 1:
        return f"exactly one root required, got {roots}"
    for cam in cameras:
        seen = set()
        cur = cam.idx
        while parent_by_cam.get(cur) is not None:
            cur = parent_by_cam[cur]
            if cur in seen:
                return f"cycle involving camera {cam.idx}"
            seen.add(cur)
    return None
