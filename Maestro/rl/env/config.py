"""rl/ 自包含的配置工具(2026-08-19 用户令:rl/ 不调用文件夹以外的包)。

dotenv 语义与主仓一致:已存在的环境变量【赢过】文件(服务器导出的
key 永远优先);缺文件不报错。yaml 直接用 pyyaml(第三方库不算越界)。
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load_dotenv(path: Path = REPO / ".env", override: bool = False) -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            n += 1
    return n


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
