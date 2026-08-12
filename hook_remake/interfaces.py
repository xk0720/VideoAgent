"""hook_remake 测试链路的数据契约(接口先行, 全部 pydantic)。

链路: 原片 → 切镜(Shot) → 按 hook 人数平均分(assignment) → 逐镜调
wan2.2-animate-move(ShotJob) → 对齐时长拼回(remake.mp4)。
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    has_audio: bool


class Shot(BaseModel):
    idx: int                      # 时间轴顺序, 从 0 起
    t0: float                     # 原片内起点(秒)
    t1: float                     # 原片内终点(秒)
    duration_s: float
    clip_path: str                # 精切出的原始片段(无音轨)
    chunk_of: Optional[int] = None  # 若为超长镜头切块, 记原镜头 idx


class HookAsset(BaseModel):
    slot: str                     # e.g. "person_hook_1"
    source: str                   # 原始 URL 或本地路径
    local_path: str = ""          # 下载/规范化后的本地文件
    oss_url: str = ""             # 上传百炼临时空间后的 oss:// 地址


JobStatus = Literal[
    "planned",        # 已排产, 未调用
    "skipped",        # 超出 --limit, 本轮不生成
    "succeeded",      # 生成成功并已下载
    "failed",         # 生成失败(成片中回退为原片段)
]


class ShotJob(BaseModel):
    shot_idx: int
    hook_slot: str                # 分到的 person_hook
    driving_path: str = ""        # 喂给模型的驱动视频(<2.1s 的镜头为回文补齐版)
    padded: bool = False          # 是否做过回文补齐
    driving_s: float = 0.0        # 驱动视频时长 ≈ 计费秒数
    status: JobStatus = "planned"
    task_id: str = ""
    gen_path: str = ""            # 模型输出(裁剪前)
    conform_path: str = ""        # 对齐到原镜头时长/分辨率/fps 后的片段
    error: str = ""


class Manifest(BaseModel):
    """一次运行的全量台账, 落盘 manifest.json。"""
    source: SourceInfo
    person_hooks: List[HookAsset]
    ignored_hooks: List[str] = Field(default_factory=list)  # object_hook_* 本版不处理
    shots: List[Shot]
    assignment: Dict[int, str]    # shot_idx → hook slot
    jobs: List[ShotJob]
    config: dict
    estimated_billed_s: float = 0.0
    final_video: str = ""
