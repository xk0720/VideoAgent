"""自带 LLM 客户端(百炼 OpenAI 兼容端点, 仅依赖 requests)。

约定: 所有 agent 走 chat_json() 拿严格 JSON——response_format 要求 json_object,
再宽容地剥离 ```json 围栏兜底。会话 trust_env=False(实测系统代理会掐长连接)。
"""
import json
import logging
import time

import requests

from .config import LLM_BASE_URL, LLM_MODEL, api_key

log = logging.getLogger("viral_studio")

_session = None


def _s() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.trust_env = False
        _session = s
    return _session


def chat_json(system: str, user: str, model: str = None,
              temperature: float = 0.6, max_retries: int = 3) -> dict:
    """一次结构化对话, 返回解析后的 dict; 解析失败自动重试并附错误提示。"""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    last_err = ""
    for attempt in range(1, max_retries + 1):
        r = _s().post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key()}",
                     "Content-Type": "application/json"},
            json={"model": model or LLM_MODEL, "messages": messages,
                  "temperature": temperature,
                  "response_format": {"type": "json_object"}},
            timeout=300)
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            log.warning("LLM 调用失败(第%d次): %s", attempt, last_err)
            time.sleep(2 * attempt)
            continue
        text = r.json()["choices"][0]["message"]["content"]
        try:
            return _parse(text)
        except Exception as e:                      # noqa: BLE001
            last_err = f"JSON 解析失败: {e}; 原文片段: {text[:200]}"
            log.warning("%s(第%d次)", last_err, attempt)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "上一条不是合法 JSON。只输出修正后的 JSON 本体, 不要任何多余文字。"})
    raise RuntimeError(f"LLM 连续 {max_retries} 次失败: {last_err}")


def _parse(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t[:4].lower() == "json" else t
        t = t.strip("` \n")
    return json.loads(t)
