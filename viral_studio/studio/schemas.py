"""viral_studio 数据契约(接口先行)。

流水线: ProductBrief → (策划) CreativeDirection → (导演) ShotScript → (执行) 工具调用。
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Role = Literal["hook", "swap", "talk", "detail", "tour", "outro"]
Mode = Literal["reuse_motion", "self_create", "self_create_multiwindow", "vo"]


class ProductBrief(BaseModel):
    """输入: 商品信息 + 参考素材。"""
    name: str
    description: str                      # 纯文本商品描述(卖点/材质/风格)
    selling_points: List[str] = Field(default_factory=list)
    category: str = ""
    product_images: List[str] = Field(default_factory=list)   # 商品图路径(可空)
    person_hooks: List[str] = Field(default_factory=list)     # 人物参考图路径
    duration_target_s: float = 20.0
    language: str = "en"                  # 烧字/口播语言


class SegmentBrief(BaseModel):
    """策划输出的段落草案(创意层, 不含执行细节)。"""
    role: Role
    duration_s: float
    idea: str                             # 这一段拍什么(中文)
    pattern_ref: Optional[str] = None     # 引用 memory/patterns/ 的 pattern_id
    asset_ref: Optional[str] = None       # 引用 memory/assets/ 的 asset_id
    reason: str                           # 为什么(须引用记忆库证据)


class CreativeDirection(BaseModel):
    """策划 agent 的产出。"""
    audience: str
    mood: str
    bgm_plan: str                         # 全片音乐思路(哪些段借资产BGM/口播段无BGM)
    structure: List[SegmentBrief]
    overall_reason: str


class WindowPlan(BaseModel):
    """multiwindow 模式下, 一次调用内部的时间窗。"""
    t0: float
    t1: float
    desc: str                             # 该窗口的画面(英文, 将进prompt)


class SegmentPlan(BaseModel):
    """导演输出的可执行段落。"""
    seg_id: str
    role: Role
    duration_s: float
    mode: Mode
    model: Literal["wan2.2-animate-move", "seedance_t2v"]
    asset_ref: Optional[str] = None       # reuse_motion 必填: 驱动素材卡 id
    person_hook_refs: List[str] = Field(default_factory=list)   # @ImageN 顺序
    product_image_refs: List[str] = Field(default_factory=list)
    prompt: str = ""                      # 英文prompt; reuse_motion 段=一句基本描述(仅台账)
    speech_text: Optional[str] = None     # vo 段台词(会写进prompt); reuse_motion 段必须为空
    bgm_source: Literal["asset_bgm", "none"] = "none"
    window_plan: Optional[List[WindowPlan]] = None
    decision_reason: str = ""

    # LLM 偏爱把空值写成 null——schema 层宽容归一, 语义问题留给确定性校验器
    @field_validator("person_hook_refs", "product_image_refs", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        if isinstance(v, str):        # "a.png, b.png" 逗号串 → 拆成列表; "[]"/"null" → 空
            s = v.strip()
            if s in ("[]", "null", "None"):
                return []
            return [x.strip() for x in s.split(",") if x.strip()]
        return v or []

    @field_validator("window_plan", mode="before")
    @classmethod
    def _window_str_to_none(cls, v):
        # 字符串形态的 window_plan 无法可靠解析 → 置空, 让语义校验器逼修复回路重写
        return None if isinstance(v, str) else v

    @field_validator("prompt", "decision_reason", mode="before")
    @classmethod
    def _none_to_str(cls, v):
        return v or ""

    @field_validator("bgm_source", mode="before")
    @classmethod
    def _none_to_none_bgm(cls, v):
        return v or "none"


class ShotScript(BaseModel):
    """导演 agent 的产出: 机器可执行分镜脚本。"""
    product_name: str
    total_duration_s: float
    segments: List[SegmentPlan]
    notes: str = ""


class ValidationReport(BaseModel):
    ok: bool
    errors: List[str] = Field(default_factory=list)     # 阻断项
    warnings: List[str] = Field(default_factory=list)   # 提醒项
