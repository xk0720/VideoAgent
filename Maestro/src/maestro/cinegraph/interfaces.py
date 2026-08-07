"""cinegraph 数据结构(ViMax interfaces 的等价移植,pydantic → dataclass)。

CGShot ≈ ViMax ShotDescription(idx/cam_idx/visual_desc/ff_desc/lf_desc/
variation/motion),外加本库私有的台词与音效字段(ViMax 无音频纪律,
这是我们的护城河)。CGCamera ≈ ViMax Camera(机位树节点)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CGShot:
    idx: int
    cam_idx: int
    visual_desc: str                    # 分镜描述(带 <名字> 标记)
    ff_desc: str                        # 首帧描述(opening_frame / 描述头)
    lf_desc: str                        # 末帧描述(end_state)
    variation: str = "small"            # small|medium|large(是否需要末帧)
    duration: Optional[float] = None
    dialogue: str = ""                  # 台词(逐字,我们的法)
    speaker: str = ""
    sounds: list = field(default_factory=list)   # 剧本载明的环境声
    ff_chars: list = field(default_factory=list)  # 首帧出场角色名
    lf_chars: list = field(default_factory=list)


@dataclass
class CGCamera:
    idx: int
    active_shot_idxs: list
    parent_cam_idx: Optional[int] = None
    parent_shot_idx: Optional[int] = None
    reason: Optional[str] = None
    is_parent_fully_covers_child: Optional[bool] = None
    missing_info: Optional[str] = None
