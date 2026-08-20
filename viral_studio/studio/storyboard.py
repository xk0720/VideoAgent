"""分镜脚本契约(v2) —— Planner 的输出。

设计要点: 每段只声明 **skill_id + slots**, 不重复 skill 卡里已经写死的东西
(prompt 正文、流水线、时长)。slots 的 key 集合由该 skill 卡的 slots 定义决定,
所以校验器能按卡逐字段校验, 加新 skill 无需改这里。
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Part = Literal["opening", "body", "ending"]


class SegmentSpec(BaseModel):
    """一段 = 选定的 skill + **填好的完整 pipeline**(可直接执行)。

    pipeline 里 prompt 全文、素材绝对路径、时长/卡点数值都已就位; 只有两类
    占位符留到运行期: @id(本段前一步的产物) 与 @bgimg(缺背景图时 Act 前置生成)。
    """
    seg_id: str
    part: Part
    skill_id: str
    variant: Optional[str] = None                # closer 用: "3" / "2"
    hook_index: Optional[int] = None             # 该段用第几张人物图(1-based)
    duration_s: float = 0.0
    t0: float = 0.0
    t1: float = 0.0
    pipeline: List[dict] = Field(default_factory=list)     # ← 主产物
    fills: Dict[str, str] = Field(default_factory=dict)    # 可追溯: LLM 填的可变内容
    reason: str = ""


class Storyboard(BaseModel):
    product_name: str
    category: str
    person_count: int
    segments: List[SegmentSpec]
    overall_reason: str = ""

    def duration_hint(self, store) -> float:
        total = 0.0
        for s in self.segments:
            card = store.get(s.skill_id) or {}
            p = card.get("produces", {})
            if "duration_s" in p:
                total += float(p["duration_s"])
            elif s.variant and "variants" in p:
                total += float(p["variants"][s.variant]["duration_s"])
        return total


class Issue(BaseModel):
    seg_id: str = ""
    field: str = ""
    msg: str


class Report(BaseModel):
    ok: bool
    errors: List[Issue] = Field(default_factory=list)
    warnings: List[Issue] = Field(default_factory=list)
