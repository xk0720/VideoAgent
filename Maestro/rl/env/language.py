# 2026-08-19 用户令【训练=生产完全同构 + rl/ 自包含】:本文件为
# src/maestro/language.py 的逐字拷贝,仅 import 行改指 rl/env 内部 shim。
# 改生产原件必须同步改这里(tests/unit/test_rl_env_parity.py 锁差异)。
"""全局输出语言策略(2026-08-05 用户令:动态语言)。

剧本/idea 是中文 → 本次运行【所有模型输出】(LLM 理由、VLM 图注、
出场矢量、评审文本、视频 prompt)一律中文;英文输入 → 不做约束
(维持英文)。唯一保留项:直发 flux 等英文偏置图像模型的 t2i 字符串
保持英文(2026-07-31 实测:中文描述画坏肖像)——但它们由 LLM 从中文
上下文翻译产出,中文绝不直通图像模型。

每次 generate_movie_windowed 起跑时 set_output_lang() 一次;各后端
指令懒读 output_lang(),零穿参。"""
from __future__ import annotations

_OUTPUT_LANG = "en"


def set_output_lang(lang: str) -> None:
    global _OUTPUT_LANG
    _OUTPUT_LANG = "zh" if str(lang).lower().startswith("zh") else "en"


def output_lang() -> str:
    return _OUTPUT_LANG


def zh() -> bool:
    return _OUTPUT_LANG == "zh"


def lang_clause(what: str = "all free-text values") -> str:
    """给英文指令模板追加的语言子句(zh 时非空)。"""
    if _OUTPUT_LANG == "zh":
        return (f" Write {what} in CHINESE (JSON keys and enum values "
                "stay English).")
    return ""
