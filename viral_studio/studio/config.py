"""viral_studio 全局配置。独立项目铁律: 不 import 本目录之外的任何项目代码。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = PROJECT_ROOT / "memory"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# LLM(策划/导演): 百炼 OpenAI 兼容端点
LLM_BASE_URL = os.environ.get(
    "VS_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("VS_LLM_MODEL", "qwen-max")
# 多模态理解(离线入库): qwen3.5-omni-plus(已确认)
OMNI_MODEL = os.environ.get("VS_OMNI_MODEL", "qwen3.5-omni-plus")


def load_dotenv() -> None:
    """极简 .env 读取(项目根), 不覆盖已有环境变量。"""
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY(环境变量或项目根 .env)")
    return key
