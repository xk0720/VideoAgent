"""Act Agent —— 把分镜脚本变成可执行的工具调用计划, 并在失败时做决策。

分工(刻意设计):
  · 正常路径 = ActCompiler 确定性编译 —— 零 LLM、零幻觉、可完全审查与复现
  · 异常路径 = LLM 决策 —— 失败模式有限且已知(NoHuman/FullFace/审核/网络/超时),
    值得让模型判断该重试、换参数、降级还是跳过

校验器同样确定性: 工具是否注册、必填参数是否齐、@引用是否可达、有没有环。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..act_compiler import ActCompiler
from ..llm import chat_json

log = logging.getLogger("viral_studio")

# 工具契约: 必填参数 + 是否远程(计费)
TOOL_SPECS: Dict[str, dict] = {
    "assemble_slots":       {"required": ["videos", "durations"], "remote": False},
    "crop_ref":             {"required": ["image"],                "remote": False},
    "grab_frame":           {"required": ["video"],                "remote": False},
    "image_edit":           {"required": ["base_image", "prompt"], "remote": True},
    "animate_move":         {"required": ["ref", "driving", "mode"], "remote": True},
    "kling_omni_video":     {"required": ["prompt", "duration"],     "remote": True},
    "seedance_t2v":         {"required": ["prompt", "duration"],     "remote": True},
    "minimax_tts":          {"required": ["text"],                   "remote": True},
    "sonilo_text_to_music": {"required": ["prompt", "duration"],     "remote": True},
    "image_generation":     {"required": ["prompt"],                 "remote": True},
    "punch_up":             {"required": ["audio", "beats"],         "remote": False},
    "trim_audio":           {"required": ["audio", "duration"],      "remote": False},
    "isolate_voice":        {"required": ["source"],                 "remote": False},
    "mix_audio":            {"required": ["video"],                  "remote": False},
    "concat_av":            {"required": ["videos"],                 "remote": False},
    "concat_audio":         {"required": ["audios"],                 "remote": False},
    "burn_subtitle":        {"required": ["video"],                  "remote": False},
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


ACT_PROMPT = (Path(__file__).parents[1] / "prompts" / "act_agent.md").read_text(
    encoding="utf-8")


class ActAgent:
    """两条通路输出同一种格式:
      compiler —— 确定性编译, 零 LLM、可复现(默认)
      agent    —— 逐段喂给 LLM 编排, 面向未来更复杂的策略
    都过同一个确定性校验器; `both` 模式可 diff 出 agent 的自由发挥或错误。
    """

    def __init__(self, store, brief: dict, bgm_source: Optional[str] = None):
        self.store = store
        self.brief = brief
        self.compiler = ActCompiler(store, brief, bgm_source)

    def plan(self, sb, mode: str = "compiler") -> tuple:
        plan = self.compiler.compile(sb)
        if mode in ("agent", "both"):
            agent_plan = self.plan_by_agent(sb, plan)
            if mode == "agent":
                return agent_plan, self.validate(agent_plan)
            plan["_agent_segments"] = agent_plan["segments"]      # 供 diff
        return plan, self.validate(plan)

    # ── agent 通路: 逐段编排 ──────────────────────────────
    def plan_by_agent(self, sb, ref_plan: dict) -> dict:
        out = json.loads(json.dumps(ref_plan))                    # 复制骨架(时间轴/计费口径)
        for seg_out, seg in zip(out["segments"], sb.segments):
            card = self.store.get(seg.skill_id) or {}
            facts = self._facts(seg, seg_out, card)
            user = (f"## 本段事实(数值已算好, 照抄即可)\n{facts}\n\n"
                    f"## 该段 prompt 正文(照抄进视频调用的 prompt 参数)\n"
                    + (self._prompt_of(seg_out) or "(无, 该段由参考视频驱动)")
                    + "\n\n请输出这一段的调用 JSON。")
            try:
                raw = chat_json(ACT_PROMPT, user, temperature=0.2)
                calls = raw.get("calls") or []
                if calls:
                    seg_out["calls"] = calls
                    seg_out["agent_reason"] = raw.get("reason", "")
                    log.info("  %s agent 编排 %d 步: %s", seg.seg_id, len(calls),
                             " → ".join(c.get("tool", "?") for c in calls))
            except Exception as e:                                # noqa: BLE001
                log.warning("%s agent 编排失败(%s), 保留编译器结果",
                            seg.seg_id, str(e)[:100])
        return out

    def _facts(self, seg, seg_out: dict, card: dict) -> str:
        """喂给 agent 的事实卡 —— 所有派生量预先算好, 它只做取舍不做算术。"""
        hooks = self.brief.get("person_hooks") or []
        pv = (card.get("produces") or {}).get("variants", {}).get(seg.variant or "", {})
        dur = seg_out["duration_s"]
        beats = pv.get("beats") or []
        bgm_src = (card.get("audio") or {}).get("bgm_source", "none")

        # 背景图: 有现成的就给路径, 否则给 prompt 让它开一张
        bg_line = "本段不需要背景图, 视频调用不要填 first_frame"
        if card.get("needs_background"):
            comp_bg = None
            for c in seg_out["calls"]:
                if c["tool"] in ("kling_omni_video", "seedance_t2v"):
                    comp_bg = c["params"].get("first_frame")
            if comp_bg and not str(comp_bg).startswith("@"):
                bg_line = f"有现成背景图 → first_frame 直接填: {comp_bg}"
            else:
                bp = (card.get("background") or {}).get("background_prompt", "").strip()
                bg_line = ("无现成背景图 → 先开 image_generation(size 720*1280), "
                           f"视频的 first_frame 填 \"@bgimg\"。背景 prompt:\n{bp}")

        refer = []
        if seg.hook_index:
            i = seg.hook_index - 1
            refer = [hooks[i]] if i < len(hooks) else []
        elif card.get("kind") == "closer" or seg.variant:
            refer = hooks[:int(seg.variant or len(hooks))]

        music_line = "本段不需要生成音乐"
        if bgm_src == "generated":
            music_line = (f"需要生成音乐: duration={int(dur) + 2}s(段长+2, 给最后一击留衰减), "
                          f"随后 punch_up 强化落点 beats={beats}, gain=0.85, "
                          f"trim_to={dur + 0.5}s; 混音 duration={dur + 0.5}s\n"
                          f"音乐 prompt:\n"
                          + self._param_of(seg_out, "sonilo_text_to_music", "prompt"))

        pipe_steps = list(card.get("pipeline") or [])
        pipe = " → ".join(f"{s['id']}({s['tool']}{'|本地' if s.get('local') else ''})"
                          for s in pipe_steps)
        # 步骤清单前置 —— 实测: 清单埋在末尾时 agent 会漏步(只输出视频调用)
        roster = []
        if "@bgimg" in bg_line:
            roster.append(("bgimg", "image_generation", False))
        roster += [(x["id"], x["tool"], bool(x.get("local"))) for x in pipe_steps]
        if (card.get("title_overlay") or {}).get("enabled"):
            roster.append(("title", "burn_text", True))
        roster_txt = "\n".join(
            f"  {i}. id=\"{cid}\"  tool={tool}  local={str(loc).lower()}"
            for i, (cid, tool, loc) in enumerate(roster, 1))

        lines = [
            f"【本段必须输出这 {len(roster)} 个调用, id/tool/local 已定, "
            f"你只需填 params 并用 @id 串起依赖】\n{roster_txt}\n",
            f"segment: {seg.seg_id} / skill={seg.skill_id}"
            + (f" / variant={seg.variant}" if seg.variant else ""),
            f"时长: {dur}s" + (f" (+{seg_out['tail_s']}s 尾部余量)" if seg_out.get("tail_s") else ""),
            f"视频调用时长参数: duration={int(dur)}",
            f"背景图: {bg_line}",
            f"人物参考: {len(refer)} 张"
            + ("".join(f"\n    <<<image_{i+1}>>> = {p}" for i, p in enumerate(refer))
               if refer else " → 纯文生视频, 记得补 aspect_ratio: \"9:16\""),
            f"音频模式: {(card.get('produces') or {}).get('audio_mode','-')}"
            f" / bgm_source={bgm_src}",
            f"音乐: {music_line}",
            f"卡声明的流水线: {pipe}",
        ]
        # 其余非视频/音乐类调用的现成参数, 直接给它抄
        extras = [c for c in seg_out["calls"]
                  if c["tool"] not in ("image_generation", "kling_omni_video",
                                       "seedance_t2v", "sonilo_text_to_music", "punch_up")]
        if extras:
            lines.append("其余步骤的参数(照抄):\n" + json.dumps(
                [{"id": c["id"], "tool": c["tool"], "local": c.get("local", False),
                  "params": c["params"]} for c in extras], ensure_ascii=False, indent=2))
        return "\n".join(lines)

    @staticmethod
    def _prompt_of(seg_out: dict) -> Optional[str]:
        for c in seg_out["calls"]:
            if c["tool"] in ("kling_omni_video", "seedance_t2v"):
                return c["params"].get("prompt")
        return None

    @staticmethod
    def _param_of(seg_out: dict, tool: str, key: str) -> str:
        for c in seg_out["calls"]:
            if c["tool"] == tool:
                return str(c["params"].get(key, ""))
        return ""

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
