"""把 skill 卡 + 本段决策 渲染成【填好的完整 pipeline】。

这是 Planner 与 Act 的分界改动: 分镜脚本里每段直接给出可执行的调用序列
(prompt 全文、素材绝对路径、时长/卡点数值都已就位), Act 只剩运行时依赖
(@产物引用、缺背景图时前置一张)要处理。

占位符只剩一类留到运行期:
  @id      本段前一步的产物, executor 执行时替换
需要先生成背景图之类的前置步骤, 在 skill 卡的 pipeline 里显式写出来 ——
pipeline 所见即所得, 没有任何一层会暗中增删步骤。
"""
import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT

log = logging.getLogger("viral_studio")


def abs_path(rel) -> Optional[str]:
    """卡里的相对路径按 PROJECT_ROOT 解析; 存在才返回绝对路径。"""
    if not rel or not isinstance(rel, str):
        return None
    if len(rel) > 250 or "\n" in rel:      # 长文本(prompt 等)不是路径
        return None
    try:
        p = Path(rel)
        if p.is_absolute():
            return str(p) if p.exists() else None
        q = PROJECT_ROOT / rel
        return str(q) if q.exists() else None
    except OSError:
        return None


class Renderer:
    def __init__(self, card: dict, hooks: List[str], person_count: int,
                 hook_index: Optional[int] = None, bgm_source: Optional[str] = None,
                 t0: float = 0.0, t1: float = 0.0):
        self.card = card
        self.hooks = hooks
        self.n = person_count
        self.hook_index = hook_index
        self.bgm_source = bgm_source
        self.t0, self.t1 = t0, t1
        self._cache: Optional[Dict[str, str]] = None

    # ── prompt 正文: 按人数选版本, 再灌入可变内容 ──────────
    def prompt(self, fills: Dict[str, str]) -> str:
        """prompt 正文取哪一段, 由卡上的 prompt_source 声明(不在代码里猜)。

        索引口径按卡的 kind 分:
          · template 类(一人一段) → 用 **本段是第几个人**(hook_index): 第 N 人
            有自己的一套动作, 逐段递进
          · 其余(多人同框, 如 closer) → 用 **总人数**: 2 人版与 3 人版画面不同
        """
        if self.card.get("kind") == "template":
            return ""            # 多人卡: 每人一套模板 → 走 prompt_of_person()
        key = (self.card.get("prompt_source") or {}).get(str(self.n))
        if key is None and "prompt_source" in self.card:
            return ""                                   # 卡明确声明该人数无 prompt
        tpl = self.card.get(key) if key else (
            self.card.get(f"prompt_{self.n}p") or self.card.get("prompt_template") or "")
        return self._fill_text(tpl or "", fills).strip()

    def prompt_of_person(self, person: int, texts: Dict[str, str]) -> str:
        """第 person 人的完整 prompt: 取 prompt_source[person] 指向的模板, 填其台词。

        多人卡里每人一套模板(动作各异)、每人一组文案(键带 _N 后缀), 所以按人渲染。
        取不到该人的模板时返回空串(如 1 人场景问第 2 人)。
        """
        src = self.card.get("prompt_source") or {}
        key = src.get(str(person))
        if key is None:
            return ""
        tpl = self.card.get(key) or ""
        return self._fill_text(tpl, texts).strip()

    def _fill_text(self, text: str, fills: Dict[str, str]) -> str:
        if not text:
            return ""
        out = text
        for k, v in {**self._derived(), **fills}.items():
            out = out.replace("{" + k + "}", str(v))
        return out

    def _derived(self) -> Dict[str, str]:
        """派生量: 从卡与人数算出来的, 不问模型。
        注意 music_prompt 只做一次浅替换 —— 走 _fill_text 会与本函数互相递归。"""
        if self._cache is not None:
            return self._cache
        d: Dict[str, str] = {}
        beats = self._beats()
        if beats:
            d["beats_text"] = (", ".join(f"{b}s" for b in beats[:-1])
                               + f" and {beats[-1]}s") if len(beats) > 1 else f"{beats[0]}s"
        mp = self.card.get("music_prompt")
        if mp:
            d["music_prompt"] = mp.replace("{beats_text}", d.get("beats_text", "")).strip()
        self._cache = d
        return d

    def _beats(self) -> List[int]:
        for step in self.pipeline_steps():
            b = (step.get("params") or {}).get("beats")
            if isinstance(b, list):
                return b
        return []

    def pipeline_steps(self) -> List[dict]:
        pls = self.card.get("pipelines") or {}
        return copy.deepcopy(pls.get(str(self.n)) or [])

    # ── 完整 pipeline ────────────────────────────────────
    def pipeline(self, texts: Dict[str, str], prompt: str = "",
                 prompts: Optional[Dict[int, str]] = None) -> List[dict]:
        """texts   = pipeline 里 {文本参数} 的值
        prompt  = 单模板卡的完整正文($prompt)
        prompts = 多人卡里第 N 人各自的正文($prompt_N), 键为 1-based 序号"""
        self._prompts = prompts or {}
        self._prompt = prompt          # 供 $prompt 解析
        out: List[dict] = []
        for step in self.pipeline_steps():
            params = self._resolve(step.get("params") or {}, texts)
            tool = step["tool"]
            if tool == "kling_omni_video" and not params.get("first_frame") \
                    and not params.get("refer"):
                params.setdefault("aspect_ratio", "9:16")   # 纯文生视频: 可灵硬性要求
            rec = {"id": step["id"], "tool": tool,
                   "local": bool(step.get("local")), "params": params}
            for k in ("optional", "lenient"):          # 可失败步骤/容缺引用: 原样带给 executor
                if step.get(k):
                    rec[k] = True
            out.append(rec)
        return out

    def _resolve(self, val: Any, fills: Dict[str, str]) -> Any:
        if isinstance(val, dict):
            return {k: self._resolve(v, fills) for k, v in val.items()}
        if isinstance(val, list):
            return [self._resolve(v, fills) for v in val]
        if not isinstance(val, str):
            return val
        s = val.strip()
        if "{" in s:
            s = self._fill_text(s, fills)
            if "{" not in s:
                return s
        if s.startswith("@"):                      # 运行期产物引用, 原样留给 executor
            return s
        if not s.startswith("$"):
            return s
        return self._lookup(s)

    def _lookup(self, expr: str) -> Any:
        if expr == "$prompt":              # 单模板卡: 本段完整 prompt 正文
            return getattr(self, "_prompt", "")
        m = re.match(r"^\$prompt_(\d+)$", expr)
        if m:                              # 多人卡: 第 N 人有自己的模板与台词
            return (getattr(self, "_prompts", {}) or {}).get(int(m.group(1)), "")
        if expr == "$hook":
            i = (self.hook_index or 1) - 1
            return self.hooks[i] if i < len(self.hooks) else None
        m = re.match(r"^\$hook_(\d+)$", expr)
        if m:
            i = int(m.group(1)) - 1
            return self.hooks[i] if i < len(self.hooks) else None
        if expr == "$background":
            # 卡里给了现成图就用它; 没有则返回 None, 由校验器响亮报错 ——
            # 需要现生成背景的卡, 请在它自己的 pipeline 里显式写 image_generation 一步,
            # 不由 Act 暗中插入(pipeline 所见即所得)。
            bg = abs_path((self.card.get("background") or {}).get("default_image"))
            if not bg:
                log.warning("%s 声明 $background 但无现成图, 且 pipeline 未含生成步",
                            self.card.get("skill_id"))
            return bg
        if expr == "$bgm_slice":
            return ({"source": self.bgm_source, "t0": self.t0, "t1": self.t1}
                    if self.bgm_source else None)
        if expr.startswith("$skill."):
            cur: Any = self.card
            for part in expr[len("$skill."):].split("."):
                cur = (cur or {}).get(part)
            if isinstance(cur, str) and "{" in cur:      # 卡内文本(如 music_prompt)先填派生量
                cur = self._fill_text(cur, {})
            return abs_path(cur) or cur
        log.warning("未知占位符 %s(原样保留)", expr)
        return expr
