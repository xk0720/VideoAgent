"""Act Agent —— 把分镜脚本变成可执行的工具调用计划, 并在失败时做决策。

分工(刻意设计):
  · 正常路径 = ActCompiler 确定性编译 —— 零 LLM、零幻觉、可完全审查与复现
  · 异常路径 = LLM 决策 —— 失败模式有限且已知(NoHuman/FullFace/审核/网络/超时),
    值得让模型判断该重试、换参数、降级还是跳过

校验器同样确定性: 工具是否注册、必填参数是否齐、@引用是否可达、有没有环。
"""
import json
import logging
from typing import Dict, List, Optional

from ..act_compiler import ActCompiler
from ..llm import chat_json

log = logging.getLogger("viral_studio")

# 工具契约: 必填参数 + 是否远程(计费)
TOOL_SPECS: Dict[str, dict] = {
    "animate_move":         {"required": ["ref", "driving", "mode"], "remote": True},
    "seedance_t2v":         {"required": ["prompt", "duration"],     "remote": True},
    "minimax_tts":          {"required": ["text"],                   "remote": True},
    "sonilo_text_to_music": {"required": ["prompt", "duration"],     "remote": True},
    "image_generation":     {"required": ["prompt"],                 "remote": True},
    "punch_up":             {"required": ["audio", "beats"],         "remote": False},
    "isolate_voice":        {"required": ["source"],                 "remote": False},
    "mix_audio":            {"required": ["video"],                  "remote": False},
    "burn_subtitle":        {"required": ["video", "text"],          "remote": False},
    "burn_text":            {"required": ["video", "text"],          "remote": False},
}

FAILURE_PROMPT = """# Act Agent — 失败决策

[Role]
你是生产执行的调度员。某次工具调用失败了, 你决定下一步怎么办。

[已知的失败模式与既定处置]
- InvalidVideo.NoHuman / InvalidVideo.FullFace: 前置检测确定性拒绝, **重试无用**
  (实测 pro/std 两轮逐镜同判) → 换路线或跳过
- 内容审核拒绝(censor): 不重试 → 跳过
- 网络类(ConnectionError/超时/5xx): 重试有效 → retry
- 参数非法(400 且响应体指明字段): 改参数后重试

[Output]
只输出 JSON:
{"action": "retry|retry_with|degrade|skip", "params_patch": {}, "reason": "中文一句"}
  retry       原样重试(仅网络类)
  retry_with  改参数重试, 把要覆盖的参数放进 params_patch
  degrade     放弃这个工具, 用更简单的方式产出(如视频段降级为静帧)
  skip        整段跳过, 成片少这一段
"""


class ActAgent:
    def __init__(self, store, brief: dict, bgm_source: Optional[str] = None):
        self.compiler = ActCompiler(store, brief, bgm_source)

    def plan(self, sb) -> tuple:
        plan = self.compiler.compile(sb)
        return plan, self.validate(plan)

    # ── 计划校验(确定性) ─────────────────────────────────
    @staticmethod
    def validate(plan: dict) -> List[str]:
        errs: List[str] = []
        for seg in plan["segments"]:
            sid = seg["seg_id"]
            if seg.get("error"):
                errs.append(f"[{sid}] {seg['error']}"); continue
            produced: List[str] = []
            for c in seg["calls"]:
                tool, cid = c["tool"], c["id"]
                spec = TOOL_SPECS.get(tool)
                if not spec:
                    errs.append(f"[{sid}.{cid}] 未注册的工具 '{tool}'"); continue
                for k in spec["required"]:
                    v = c["params"].get(k)
                    if v is None or v == "":
                        errs.append(f"[{sid}.{cid}] {tool} 缺必填参数 '{k}'")
                # @引用必须指向本段前面已产出的调用
                for k, v in c["params"].items():
                    for ref in _refs(v):
                        if ref not in produced:
                            errs.append(f"[{sid}.{cid}] 参数 {k} 引用了 @{ref}, "
                                        f"但它不在前序产物 {produced} 中")
                # 未解析的占位符不该漏进最终计划
                for k, v in c["params"].items():
                    if isinstance(v, str) and (v.startswith("$") or "{" in v):
                        errs.append(f"[{sid}.{cid}] 参数 {k} 未解析: {v[:40]}")
                produced.append(cid)
        return errs

    # ── 失败决策(LLM, 仅在执行期被调用) ──────────────────
    @staticmethod
    def on_failure(call: dict, error: str, attempt: int) -> dict:
        user = (f"## 失败的调用\n工具: {call['tool']}\n"
                f"参数: {json.dumps(call['params'], ensure_ascii=False)[:600]}\n"
                f"第 {attempt} 次尝试\n\n## 错误\n{error[:600]}\n\n请输出决策 JSON。")
        try:
            return chat_json(FAILURE_PROMPT, user, temperature=0.2)
        except Exception as e:                      # noqa: BLE001 决策失败 → 保守跳过
            log.warning("失败决策 LLM 不可用(%s), 保守 skip", str(e)[:80])
            return {"action": "skip", "params_patch": {}, "reason": "决策不可用"}


def _refs(v) -> List[str]:
    out = []
    if isinstance(v, str) and v.startswith("@"):
        out.append(v[1:])
    elif isinstance(v, list):
        for x in v:
            out += _refs(x)
    elif isinstance(v, dict):
        for x in v.values():
            out += _refs(x)
    return out
