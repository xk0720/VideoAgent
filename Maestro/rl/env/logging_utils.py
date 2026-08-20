# 2026-08-19 用户令【训练=生产完全同构 + rl/ 自包含】:本文件为
# src/maestro/logging_utils.py 的逐字拷贝,仅 import 行改指 rl/env 内部 shim。
# 改生产原件必须同步改这里(tests/unit/test_rl_env_parity.py 锁差异)。
"""Structured logging helper (named logging_utils to avoid shadowing stdlib)."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional

_CONFIGURED = False


def get_logger(name: str = "maestro", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
        )
        root = logging.getLogger("maestro")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("maestro") else f"maestro.{name}")


# ─────────────────────────────────────────────────────────────────────────
# Brain 决策日志(2026-07-14 用户令):debug 时要能看到 brain LLM 每次
# tool call / 策略决策的【原始输出】和解析结果(比如选 tiv2v_window 时
# 它到底输出了什么)。JSONL 逐行落盘 —— 路径由 set_brain_log() 或环境变量
# $MAESTRO_BRAIN_LOG 提供;未设置 = 只打终端 INFO,不落盘。绝不抛错。
# ─────────────────────────────────────────────────────────────────────────
_BRAIN_LOG_PATH: Optional[str] = None


def set_brain_log(path) -> None:
    """指定 brain 决策 JSONL 的输出文件(测试脚本接 <out_dir>/brain_calls.jsonl)。"""
    global _BRAIN_LOG_PATH
    _BRAIN_LOG_PATH = str(path) if path else None


def brain_log(stage: str, record: dict) -> str:
    """记一条 brain 决策:终端 INFO 一行(紧凑)+ JSONL 一行(全量,含 raw)。

    record 约定字段:label/shot_idx(定位哪一镜)、menu(可选项名)、
    raw(LLM 原始回复,完整保留)、parsed(校验后的决策 dict 或 None)、
    via(episode/llm/fallback/skill)。

    S0(RL 数据管道,2026-07-18):每条记录自动带 `decision_id`(uuid),
    并把它返回给调用方 —— 决策与它的延迟结局(repair/outcome、
    window/shot_outcome)靠这个 id 显式连接,训练数据不再靠时序猜。"""
    import uuid

    record.setdefault("decision_id", uuid.uuid4().hex[:16])
    log = get_logger("brain")
    try:
        compact = {k: v for k, v in record.items() if k != "raw"}
        log.info("brain[%s] %s", stage,
                 json.dumps(compact, ensure_ascii=False, default=str)[:800])
    except Exception:
        pass
    path = _BRAIN_LOG_PATH or os.getenv("MAESTRO_BRAIN_LOG")
    if not path:
        return record["decision_id"]
    try:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "stage": stage, **record},
                               ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        log.warning("brain_log write failed (%s): %s", path, exc)
    return record["decision_id"]
